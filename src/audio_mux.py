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
        
        # Some hardware encoders don't support the 'preset' flag in the same way, 
        # so we only use it for libx264 or known compatible ones, or we just pass it and hope it works.
        # NVENC uses p1-p7 or slow/medium/fast.
        
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

    def write(self, frame_bgr):
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, OSError):
            pass  # It will be handled in close()

    def close(self):
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        _, stderr = self._proc.communicate()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr.decode(errors='replace')}")
