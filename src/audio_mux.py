"""Streams raw BGR frames into ffmpeg (bundled via imageio-ffmpeg) and muxes
in the music track, looping/trimming it to exactly match the video length.
"""
import subprocess

import imageio_ffmpeg


class FfmpegFrameWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float, music_path: str = None):
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]
        if music_path:
            cmd += [
                "-stream_loop", "-1", "-i", music_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
            ]
        else:
            cmd += [
                "-map", "0:v:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
            ]
        cmd.append(output_path)
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def write(self, frame_bgr):
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        self._proc.stdin.close()
        _, stderr = self._proc.communicate()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr.decode(errors='replace')}")
