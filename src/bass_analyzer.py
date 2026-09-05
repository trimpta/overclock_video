import os
import subprocess
import numpy as np
import imageio_ffmpeg
import concurrent.futures

class BassAnalyzer:
    """Extracts a smoothed, bass-reactive energy curve from an audio file.
    
    Reads the audio via ffmpeg (mixed down to mono, 44.1kHz), applies a simple
    IIR lowpass filter to isolate the bass (~120Hz), computes per-frame RMS energy,
    normalizes it, and applies an exponential curve and envelope smoothing.
    """
    def __init__(self, audio_path: str, fps: float):
        self.audio_path = audio_path
        self.fps = fps if fps > 0 else 30.0
        self._energy_curve = []
        self._future = None
        
        if self.audio_path and os.path.isfile(self.audio_path):
            self._future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(self._analyze)
            
    def get_level(self, frame_idx: int) -> float:
        """Return smoothed, emphasis-curved bass level [0, 1] for a given frame."""
        if not self._future:
            return 0.0
            
        try:
            # Wait briefly if it's almost done, otherwise return 0 if still processing
            curve = self._future.result(timeout=0.01)
            if not curve:
                return 0.0
            idx = min(max(0, frame_idx), len(curve) - 1)
            return curve[idx]
        except concurrent.futures.TimeoutError:
            return 0.0
        except Exception:
            return 0.0

    def _analyze(self):
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-i", self.audio_path,
                "-vn",                 # No video
                "-ac", "1",            # Mono
                "-ar", "44100",        # 44.1kHz
                "-f", "s16le",         # 16-bit PCM little-endian
                "-loglevel", "quiet",
                "-"                    # output to stdout
            ]
            
            # Read all PCM data
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            pcm_data, _ = process.communicate()
            
            if not pcm_data:
                return []
                
            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            if len(samples) == 0:
                return []

            # 1. Lowpass filter (simple 2-pole IIR, approx 120Hz cutoff at 44.1kHz)
            # Alpha ~ 0.017 for 120Hz at 44100Hz (alpha = 2*pi*fc/fs)
            # Y[n] = alpha * X[n] + (1 - alpha) * Y[n-1] (applied twice for 2-pole)
            alpha = 0.017
            
            # Fast vectorized IIR filter using scipy
            from scipy import signal
            # Create a 2nd order Butterworth lowpass filter at 120Hz
            nyquist = 44100 / 2
            b, a = signal.butter(2, 120 / nyquist, btype='low')
            bass_samples = signal.lfilter(b, a, samples)

            # 2. Frame-level RMS energy
            samples_per_frame = int(44100 / self.fps)
            if samples_per_frame <= 0:
                return []
                
            num_frames = len(bass_samples) // samples_per_frame
            if num_frames == 0:
                return []
                
            # Truncate to exact frame boundaries and reshape
            bass_frames = bass_samples[:num_frames * samples_per_frame].reshape((num_frames, samples_per_frame))
            
            # Compute RMS
            energy = np.sqrt(np.mean(bass_frames**2, axis=1))
            
            # 3. Normalize to [0, 1]
            max_energy = np.max(energy)
            if max_energy > 0:
                energy = energy / max_energy
                
            # 4. Apply exponential emphasis curve (x^0.3)
            # Makes small bass increases register as large jumps
            energy = np.power(energy, 0.3)
            
            # 5. One-sided smooth (fast attack, slow decay)
            decay_factor = 0.85 # Adjust for decay speed
            smoothed = np.zeros_like(energy)
            smoothed[0] = energy[0]
            for i in range(1, len(energy)):
                smoothed[i] = max(energy[i], smoothed[i-1] * decay_factor)
                
            return smoothed.tolist()
            
        except Exception as e:
            print(f"BassAnalyzer error: {e}")
            return []
