"""Streams raw BGR frames into ffmpeg (bundled via imageio-ffmpeg) and muxes
in the music track, looping/trimming it to exactly match the video length.
"""
import os
import re
import subprocess
import sys
import threading
import imageio_ffmpeg
import numpy as np

from src.config import resolve_write_path


def _codec_args(codec: str, preset: str):
    """Return encoder-specific quality/preset flags. Unknown codecs get none."""
    if codec == "libx264":
        return ["-preset", preset, "-crf", "18"]
    if codec == "h264_nvenc":
        return ["-preset", "p4", "-cq", "18"]
    if codec == "h264_qsv":
        # QSV rejects x264-style presets; use global_quality only.
        return ["-global_quality", "18"]
    if codec == "h264_amf":
        return ["-rc", "cqp", "-qp_i", "18", "-qp_p", "18"]
    return []


class FfmpegFrameWriter:
    """Pipe BGR frames to ffmpeg and optionally mux a looping music track.

    Callers should pass even width/height. Odd dimensions are rounded down to
    the nearest even size (minimum 2) because yuv420p requires even dims.
    """

    def __init__(self, output_path: str, width: int, height: int, fps: float, music_path: str = None, codec: str = "libx264", preset: str = "medium"):
        width = int(width)
        height = int(height)
        width &= ~1
        height &= ~1
        self.width = max(2, width)
        self.height = max(2, height)
        output_path = resolve_write_path(output_path)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-r", str(fps),
            "-i", "-",
        ]

        vcodec_args = ["-c:v", codec, "-pix_fmt", "yuv420p"] + _codec_args(codec, preset)

        if music_path:
            cmd.extend([
                "-stream_loop", "-1", "-i", music_path,
                "-map", "0:v:0", "-map", "1:a:0",
            ])
            cmd.extend(vcodec_args)
            cmd.extend([
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
            ])
        else:
            cmd.extend(["-map", "0:v:0"])
            cmd.extend(vcodec_args)
        cmd.append(output_path)
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self.failed = False
        self.error_message = None
        self._closed = False
        # Continuously drain stderr so a full PIPE cannot deadlock long encodes.
        self._stderr_chunks = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            while True:
                chunk = self._proc.stderr.read(4096)
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
        except Exception:
            pass

    def _fit_frame(self, frame_bgr):
        """Center-crop or pad frame to the even encoder size stored at init."""
        h, w = frame_bgr.shape[:2]
        if w == self.width and h == self.height:
            return frame_bgr
        out = np.zeros((self.height, self.width, 3), dtype=frame_bgr.dtype)
        # Source crop window (center)
        src_y0 = max(0, (h - self.height) // 2)
        src_x0 = max(0, (w - self.width) // 2)
        ch = min(h, self.height)
        cw = min(w, self.width)
        # Destination paste origin (center when padding)
        dst_y0 = max(0, (self.height - ch) // 2)
        dst_x0 = max(0, (self.width - cw) // 2)
        out[dst_y0:dst_y0 + ch, dst_x0:dst_x0 + cw] = frame_bgr[src_y0:src_y0 + ch, src_x0:src_x0 + cw]
        return out

    def write(self, frame_bgr):
        if self.failed or self._closed:
            return False
        try:
            fitted = self._fit_frame(frame_bgr)
            self._proc.stdin.write(fitted.tobytes())
            return True
        except (BrokenPipeError, OSError) as e:
            self.failed = True
            self.error_message = f"ffmpeg pipe broken during write: {e}"
            return False

    def _stderr_text(self):
        return b"".join(self._stderr_chunks).decode(errors="replace").strip()

    def close(self):
        if self._closed:
            if self.failed:
                raise RuntimeError(self.error_message or "ffmpeg writer already failed")
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            self.failed = True
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.terminate()
            self.failed = True
            self.error_message = "ffmpeg timed out during close"
            raise RuntimeError(self.error_message)
        if self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2)

        stderr_text = self._stderr_text()
        if self._proc.returncode not in (0, None):
            self.failed = True
            self.error_message = f"ffmpeg failed (code {self._proc.returncode}):\n{stderr_text}"
            raise RuntimeError(self.error_message)
        if self.failed:
            raise RuntimeError(self.error_message or f"ffmpeg failed:\n{stderr_text}")

    def terminate(self):
        if self._closed and self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        if getattr(self, "_stderr_thread", None) is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        self._closed = True
        self.failed = True


def get_default_audio_device():
    """Detect default microphone device for audio capture."""
    if sys.platform.startswith("win"):
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            dev = QMediaDevices.defaultAudioInput()
            if dev and not dev.isNull():
                desc = dev.description()
                if desc:
                    return desc
        except Exception:
            pass
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            proc = subprocess.run(
                [exe, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=5,
            )
            devices = re.findall(r'"([^"]+)"\s+\(audio\)', proc.stderr)
            if devices:
                return devices[0]
        except Exception:
            pass
    elif sys.platform.startswith("darwin"):
        return ":default"
    else:
        return "default"
    return None


def list_audio_input_devices():
    """Return a list of available microphone device names."""
    devices = []
    try:
        from PyQt6.QtMultimedia import QMediaDevices
        qt_devices = QMediaDevices.audioInputs()
        for d in qt_devices:
            if d and not d.isNull():
                desc = d.description()
                if desc and desc not in devices:
                    devices.append(desc)
    except Exception:
        pass

    if sys.platform.startswith("win") and not devices:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            proc = subprocess.run(
                [exe, "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=5,
            )
            dshow_devices = re.findall(r'"([^"]+)"\s+\(audio\)', proc.stderr)
            for d in dshow_devices:
                if d not in devices:
                    devices.append(d)
        except Exception:
            pass

    return devices


class FfmpegAudioRecorder:
    """Records microphone audio to a file in the background using ffmpeg."""

    def __init__(self, output_path: str, device_name: str = None):
        self.output_path = resolve_write_path(output_path)
        self.device_name = device_name or get_default_audio_device()
        self._proc = None
        self._closed = False
        self.failed = False
        self.error_message = None

    def start(self):
        if not self.device_name:
            self.failed = True
            self.error_message = "No audio input device detected."
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            if os.path.isfile(self.output_path):
                try:
                    os.remove(self.output_path)
                except Exception:
                    pass
        except Exception:
            pass

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if sys.platform.startswith("win"):
            input_args = ["-f", "dshow", "-i", f"audio={self.device_name}"]
        elif sys.platform.startswith("darwin"):
            input_args = ["-f", "avfoundation", "-i", str(self.device_name)]
        else:
            input_args = ["-f", "pulse" if "pulse" in str(self.device_name) else "alsa", "-i", str(self.device_name)]

        cmd = [
            ffmpeg_exe, "-y",
            "-loglevel", "error",
            *input_args,
            "-ac", "2",
            "-ar", "44100",
            self.output_path,
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return True
        except Exception as e:
            self.failed = True
            self.error_message = f"Failed to start audio recorder: {e}"
            return False

    def stop(self):
        if self._proc is None or self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.write(b"q")
                self._proc.stdin.flush()
                self._proc.stdin.close()
        except Exception:
            pass

        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.terminate()

    def terminate(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3)
        except Exception:
            pass
        self._closed = True
