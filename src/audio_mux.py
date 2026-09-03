"""Streams raw BGR frames into ffmpeg (bundled via imageio-ffmpeg) and muxes
in the music track, looping/trimming it to exactly match the video length.
"""
import subprocess

import imageio_ffmpeg


class FfmpegFrameWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float, music_path: str = None, codec: str = "libx264", preset: str = "medium"):
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]

        if music_path:
            cmd.extend([
                "-stream_loop", "-1", "-i", music_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", codec, "-pix_fmt", "yuv420p",
            ])
            if codec == "libx264":
                cmd.extend(["-preset", preset, "-crf", "18"])
            elif codec == "h264_nvenc":
                cmd.extend(["-preset", "p4", "-cq", "18"])

            cmd.extend([
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
            ])
        else:
            cmd.extend([
                "-map", "0:v:0",
                "-c:v", codec, "-pix_fmt", "yuv420p",
            ])
            if codec == "libx264":
                cmd.extend(["-preset", preset, "-crf", "18"])
            elif codec == "h264_nvenc":
                cmd.extend(["-preset", "p4", "-cq", "18"])
        cmd.append(output_path)
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self.failed = False
        self.error_message = None
        self._closed = False

    def write(self, frame_bgr):
        if self.failed or self._closed:
            return False
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
            return True
        except (BrokenPipeError, OSError) as e:
            self.failed = True
            self.error_message = f"ffmpeg pipe broken during write: {e}"
            return False

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
            _, stderr = self._proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.terminate()
            _, stderr = self._proc.communicate()
            self.failed = True
            self.error_message = "ffmpeg timed out during close"
            raise RuntimeError(self.error_message)

        stderr_text = (stderr or b"").decode(errors="replace").strip()
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
            self._proc.communicate(timeout=5)
        except Exception:
            pass
        self._closed = True
        self.failed = True
