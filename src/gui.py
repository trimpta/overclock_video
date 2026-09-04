import os
import shutil
import sys
import threading
import time
import traceback
import cv2
import numpy as np
import concurrent.futures

from PyQt6.QtCore import Qt, QThread, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QCheckBox,
    QGroupBox, QFormLayout, QComboBox, QMessageBox, QStyle, QProgressBar, QLineEdit, QDialog,
)

from src.audio_mux import FfmpegFrameWriter, FfmpegAudioRecorder, get_default_audio_device
from src.config import (
    MEDIA_DIR,
    OUTPUT_DIR,
    USER_DATA_DIR,
    ensure_app_dirs,
    load_last_run,
    resolve_existing_path,
    resolve_output_dir,
    resolve_output_file,
    save_last_run,
    session_compositing_settings,
)
from src.compositor import build_canvas, composite_frame, warp_composite_frame, load_image_bgra, make_greenscreen_bgra, make_grid_bgra, full_frame_placement
from src.hand_tracking import HandTracker
from src.smoothing import QuadTracker, order_points_tl_tr_br_bl

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hand_landmarker.task")
ensure_app_dirs()
TEMP_RECORDING = os.path.join(USER_DATA_DIR, "temp_recording.mp4")
RAW_TEMP_RECORDING = os.path.join(USER_DATA_DIR, "raw_temp.mp4")
RAW_TEMP_MIC_AUDIO = os.path.join(USER_DATA_DIR, "raw_mic.wav")


class PreviewLabel(QLabel):
    placement_changed = pyqtSignal(dict)
    size_changed = pyqtSignal(int, int)

    def __init__(self):
        super().__init__("Select a video source to begin.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #fafafa; border: 1px solid #ddd; color: #333;")
        self.setToolTip("Interactive Preview:\n- Click and drag the center to move the image.\n- Click and drag the edges to scale the image.")
        
        self.is_dragging = False
        self.drag_start_pos = None
        self.drag_mode = "translate" # "translate" or "scale"
        self.current_placement = {"x": 960, "y": 540, "scale": 1.0, "rotation_deg": 0.0}
        self.warp_mode = False
        self.frame_size = (1920, 1080) # default, updated by thread

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.size_changed.emit(self.width(), self.height())

    def update_placement(self, placement):
        self.current_placement.update(placement)

    def _content_rect(self):
        """Painted pixmap rect: QLabel centers an unscaled pixmap in contentsRect."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        cr = self.contentsRect()
        pw, ph = float(pm.width()), float(pm.height())
        if pw <= 0 or ph <= 0 or cr.width() <= 0 or cr.height() <= 0:
            return None
        x = cr.x() + (cr.width() - pw) / 2.0
        y = cr.y() + (cr.height() - ph) / 2.0
        return QRectF(x, y, pw, ph)

    def mousePressEvent(self, event):
        if self.warp_mode: return
        if event.button() == Qt.MouseButton.LeftButton:
            content = self._content_rect()
            if content is None or not content.contains(event.position()):
                return

            self.is_dragging = True
            self.drag_start_pos = event.position()

            # Determine drag mode in content coordinates (ignore letterbox bars)
            x = event.position().x() - content.x()
            y = event.position().y() - content.y()
            w, h = content.width(), content.height()
            margin_x, margin_y = w * 0.2, h * 0.2

            # Outer 20% border = scale; inner 60% = translate.
            if x < margin_x or x > w - margin_x or y < margin_y or y > h - margin_y:
                self.drag_mode = "scale"
            else:
                self.drag_mode = "translate"

    def mouseMoveEvent(self, event):
        if not self.is_dragging or self.warp_mode: return

        content = self._content_rect()
        if content is None:
            return

        pos = event.position()
        delta = pos - self.drag_start_pos
        self.drag_start_pos = pos

        scale_x = self.frame_size[0] / max(1.0, content.width())
        scale_y = self.frame_size[1] / max(1.0, content.height())

        if self.drag_mode == "translate":
            dx = delta.x() * scale_x
            dy = delta.y() * scale_y
            self.current_placement["x"] += dx
            self.current_placement["y"] += dy
        elif self.drag_mode == "scale":
            # Dragging up/right increases scale, down/left decreases
            scale_delta = (delta.x() - delta.y()) * 0.005
            self.current_placement["scale"] = max(0.01, self.current_placement["scale"] + scale_delta)

        self.placement_changed.emit(self.current_placement)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False


class VideoProcessorThread(QThread):
    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)
    duration_changed = pyqtSignal(int) # total frames
    position_changed = pyqtSignal(int) # current frame
    resolution_changed = pyqtSignal(int, int)
    recording_finished = pyqtSignal()
    raw_recording_ready = pyqtSignal()
    export_finished = pyqtSignal()
    endfade_scan_progress = pyqtSignal(int, int)
    endfade_scan_complete = pyqtSignal(int, np.ndarray)
    endfade_scan_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self.video_path = None
        self.image_path = None
        self.music_path = None
        self.is_webcam = False
        
        self.codec = "libx264"
        self.preset = "medium"
        
        self.warp_mode = False
        self.placement = {"x": 960, "y": 540, "scale": 1.0, "rotation_deg": 0.0}
        self.feather = 9
        self.coast_limit = 15
        self.show_debug = False
        self.render_debug = False

        self._running = False
        self._paused = False
        self._seek_request = None
        self._needs_restart = False
        self._force_preview_frame = False
        
        self.endfade_mode = False
        self._scanning_endfade = False
        self.endfade_trigger_frame = None
        self.endfade_base_quad = None
        self.endfade_base_quad_proxy = None
        self.endfade_duration_frames = 60
        self.endfade_offset = 0

        self._recording = False
        self._exporting = False
        self.export_path = None
        
        self.target_gui_size = (640, 360)
        
        self.cap = None
        self.tracker = None
        self.quad_tracker = None
        self.writer = None
        self.image_bgra = None
        
        self._cached_canvas_bgr = None
        self._cached_canvas_alpha = None
        self._last_placement = None
        self._cached_canvas_size = None
        self._executor = None
        self._current_qimage_ref = None
        self._gui_frame_pending = False
        
        self.proxy_scale = 0.5
        self._raw_writer = None
        self._render_params_snapshot = None
        self._rendering_webcam = False
        self._last_frame_emit_time = None
        self.record_mic = True
        self._mic_recorder = None

    def _set_flag(self, name, value):
        with self._lock:
            setattr(self, name, value)

    def _get_flag(self, name):
        with self._lock:
            return getattr(self, name)

    def ack_frame(self):
        self._gui_frame_pending = False

    @staticmethod
    def _drain_future(future):
        """Wait for an in-flight tracker future so process/reset stay single-threaded."""
        if future is None:
            return None
        try:
            future.result()
        except Exception:
            pass
        return None

    def request_webcam_render(self):
        self._set_flag('_rendering_webcam', True)

    def start_recording(self):
        self._set_flag('_recording', True)
        if self.is_webcam and not self.music_path and self.record_mic:
            try:
                self._mic_recorder = FfmpegAudioRecorder(RAW_TEMP_MIC_AUDIO)
                self._mic_recorder.start()
            except Exception as e:
                print(f"Warning: Failed to start microphone recording: {e}")
                self._mic_recorder = None

    def stop_recording(self):
        self._set_flag('_recording', False)
        if self._mic_recorder is not None:
            try:
                self._mic_recorder.stop()
            except Exception:
                pass
            self._mic_recorder = None
        
    def start_export(self, export_path):
        self.export_path = export_path
        self._set_flag('_exporting', True)

    def update_params(self, params):
        if "record_mic" in params: self.record_mic = bool(params["record_mic"])
        if "warp_mode" in params: self.warp_mode = params["warp_mode"]
        if "feather" in params: self.feather = params["feather"]
        if "coast_limit" in params: self.coast_limit = params["coast_limit"]
        if "show_debug" in params: self.show_debug = params["show_debug"]
        if "render_debug" in params: self.render_debug = bool(params["render_debug"])
        if "placement" in params:
            self.placement.update(params["placement"])
        if "codec" in params: self.codec = params["codec"]
        if "endfade_mode" in params:
            new_mode = params["endfade_mode"]
            if new_mode and not self.endfade_mode and not self.is_webcam:
                self._set_flag('_scanning_endfade', True)
            self.endfade_mode = new_mode
        if "endfade_offset" in params: self.endfade_offset = params["endfade_offset"]
        if "endfade_duration" in params: self.endfade_duration_frames = params["endfade_duration"]

    def set_gui_size(self, w, h):
        self.target_gui_size = (w, h)

    def set_media(self, video_path, image_path, music_path, is_webcam=False):
        self.video_path = video_path
        self.image_path = image_path
        self.music_path = music_path
        self.is_webcam = is_webcam
        self._cached_canvas_bgr = None
        self._cached_canvas_alpha = None
        self._last_placement = None
        self._cached_canvas_size = None
        # Restart in-place inside run() — never block the GUI thread with wait().
        self._set_flag('_needs_restart', True)

    def request_seek(self, frame_idx):
        self._last_frame_emit_time = None
        self._set_flag('_seek_request', frame_idx)
        
    def set_paused(self, paused):
        self._last_frame_emit_time = None
        self._set_flag('_paused', paused)

    def _quad_pts_for_mode(self, smoothed, warp_mode=None):
        """Warp uses TL/TR/BR/BL order; stencil fillPoly uses angle-sorted winding."""
        use_warp = self.warp_mode if warp_mode is None else warp_mode
        if use_warp:
            pts = QuadTracker.ordered_quad_points(smoothed)
            if pts is None:
                return None
            return np.round(pts).astype(np.int32)
        return QuadTracker.quad_points(smoothed)

    @staticmethod
    def _clamp_fps(fps, default=30.0):
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return default
        if fps <= 0 or fps > 120:
            return default
        return fps

    @staticmethod
    def _measured_fps(frame_count, t0, fallback=30.0):
        if t0 is None or frame_count <= 0:
            return fallback
        elapsed = time.perf_counter() - t0
        if elapsed <= 1e-6:
            return fallback
        return VideoProcessorThread._clamp_fps(frame_count / elapsed, fallback)

    def _snapshot_render_params(self, fps=None):
        snap = {
            "warp_mode": self.warp_mode,
            "placement": self.placement.copy(),
            "feather": self.feather,
            "coast_limit": self.coast_limit,
            "endfade_mode": self.endfade_mode,
            "endfade_offset": self.endfade_offset,
            "endfade_duration_frames": self.endfade_duration_frames,
            "endfade_base_quad": self.endfade_base_quad.copy() if self.endfade_base_quad is not None else None,
            "endfade_base_quad_proxy": self.endfade_base_quad_proxy.copy() if self.endfade_base_quad_proxy is not None else None,
            "endfade_trigger_frame": self.endfade_trigger_frame,
            "codec": self.codec,
            "render_debug": self.render_debug,
            "record_mic": self.record_mic,
            "is_webcam": self.is_webcam,
        }
        if fps is not None:
            snap["fps"] = self._clamp_fps(fps)
        return snap

    def _abort_raw_writer(self, message=None):
        if message:
            self.error_occurred.emit(message)
        if self._mic_recorder is not None:
            try:
                self._mic_recorder.terminate()
            except Exception:
                pass
            self._mic_recorder = None
        if self._raw_writer is not None:
            try:
                self._raw_writer.terminate()
            except Exception:
                pass
            self._raw_writer = None
        self._set_flag('_recording', False)

    def _open_raw_writer(self, frame_w, frame_h, fps):
        fps = self._clamp_fps(fps)
        self._raw_writer = FfmpegFrameWriter(
            RAW_TEMP_RECORDING, frame_w, frame_h, fps, codec="libx264", preset="ultrafast",
        )
        return fps

    def _write_raw_frames(self, frames):
        if self._raw_writer is None:
            return False
        for frame in frames:
            if not self._raw_writer.write(frame) or self._raw_writer.failed:
                err = self._raw_writer.error_message or "Raw recording write failed."
                self._abort_raw_writer(err)
                return False
        return True

    def _clear_canvas_cache(self):
        self._cached_canvas_bgr = None
        self._cached_canvas_alpha = None
        self._last_placement = None
        self._cached_canvas_size = None

    def _reset_endfade_state(self):
        """Drop prior-clip endfade so a new source cannot inherit trigger/quads."""
        self.endfade_trigger_frame = None
        self.endfade_base_quad = None
        self.endfade_base_quad_proxy = None
        self._set_flag('_scanning_endfade', False)

    def _end_session(self):
        """Release capture/tracker for the current source without stopping the thread."""
        if getattr(self, '_mic_recorder', None) is not None:
            try:
                self._mic_recorder.terminate()
            except Exception:
                pass
            self._mic_recorder = None
        if getattr(self, '_raw_writer', None):
            try:
                self._raw_writer.terminate()
            except Exception:
                pass
            self._raw_writer = None
        if getattr(self, 'writer', None):
            try:
                self.writer.terminate()
            except Exception:
                pass
            self.writer = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception:
                pass
            self.tracker = None
        self.quad_tracker = None

    def _begin_session(self):
        """Open video source, load media, create tracker. Returns session tuple or None."""
        self._last_frame_emit_time = None
        if self.video_path is None:
            self.error_occurred.emit("No video source selected.")
            return None

        if not os.path.isfile(MODEL_PATH):
            self.error_occurred.emit(f"Hand tracking model not found:\n{MODEL_PATH}")
            return None

        self._reset_endfade_state()

        if self.is_webcam:
            cam_idx = int(self.video_path) if str(self.video_path).isdigit() else 0
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = cv2.VideoCapture(cam_idx)
            else:
                self.cap = cv2.VideoCapture(cam_idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.error_occurred.emit(f"Could not open video source: {self.video_path}")
            self._end_session()
            return None

        if self.is_webcam:
            # CAP_PROP_FPS is often 0/wrong for cameras; measure while recording (#51).
            fps = 30.0
        else:
            fps = self._clamp_fps(self.cap.get(cv2.CAP_PROP_FPS), 30.0)
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = None if self.is_webcam else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_w < 2 or frame_h < 2:
            self.error_occurred.emit(
                f"Invalid capture size {frame_w}x{frame_h} (need at least 2x2)."
            )
            self._end_session()
            return None

        longest = max(frame_w, frame_h)
        self.proxy_scale = min(0.5, max(0.25, 640.0 / longest))
        self.resolution_changed.emit(frame_w, frame_h)

        if not self.is_webcam and total_frames:
            self.duration_changed.emit(total_frames)

        proxy_w = max(2, int(frame_w * self.proxy_scale))
        proxy_h = max(2, int(frame_h * self.proxy_scale))

        self.full_grid_bgra = make_grid_bgra(frame_w, frame_h)
        self.grid_bgra = make_grid_bgra(proxy_w, proxy_h)

        if self.image_path and os.path.isfile(self.image_path):
            self.full_image_bgra = load_image_bgra(self.image_path)
            self.image_bgra = cv2.resize(self.full_image_bgra, (0, 0), fx=self.proxy_scale, fy=self.proxy_scale, interpolation=cv2.INTER_AREA)
            self.has_custom_image = True
        else:
            self.full_image_bgra = make_greenscreen_bgra(frame_w, frame_h)
            self.image_bgra = make_greenscreen_bgra(proxy_w, proxy_h)
            self.has_custom_image = False

        self.tracker = HandTracker(MODEL_PATH, num_hands=2)
        self.quad_tracker = QuadTracker(coast_limit=self.coast_limit)
        self._gui_frame_pending = False
        self._clear_canvas_cache()
        # Keep endfade_mode (UI checkbox); rescan the new clip if still enabled.
        if self.endfade_mode and not self.is_webcam:
            self._set_flag('_scanning_endfade', True)
        return fps, frame_w, frame_h, total_frames, proxy_w, proxy_h

    def _ensure_canvas(self, active_image_bgra, placement, cache_w, cache_h, proxy_scale=1.0, warp_mode=None):
        cache_key_size = (cache_w, cache_h)
        use_warp = self.warp_mode if warp_mode is None else warp_mode
        if not getattr(self, "has_custom_image", True) and not use_warp:
            if self._cached_canvas_bgr is None or self._cached_canvas_size != cache_key_size:
                self._cached_canvas_bgr = active_image_bgra[:, :, :3].copy()
                self._cached_canvas_alpha = active_image_bgra[:, :, 3].copy()
                self._cached_canvas_size = cache_key_size
            return
        if self._last_placement != placement or self._cached_canvas_bgr is None or self._cached_canvas_size != cache_key_size:
            canvas_placement = placement.copy()
            if proxy_scale != 1.0:
                canvas_placement["x"] = canvas_placement["x"] * proxy_scale
                canvas_placement["y"] = canvas_placement["y"] * proxy_scale
            self._cached_canvas_bgr, self._cached_canvas_alpha = build_canvas(active_image_bgra, canvas_placement, cache_w, cache_h)
            self._last_placement = placement.copy()
            self._cached_canvas_size = cache_key_size

    def _compute_endfade_quad(self, prev_frame_idx, quad_pts, frame_w, frame_h, proxy_w, proxy_h, use_proxy=False,
                               endfade_base_quad=None, endfade_base_quad_proxy=None, params=None):
        p = params if params is not None else {}
        endfade_mode = p.get("endfade_mode", self.endfade_mode)
        trigger_frame = p.get("endfade_trigger_frame", self.endfade_trigger_frame)
        offset = p.get("endfade_offset", self.endfade_offset)
        duration = p.get("endfade_duration_frames", self.endfade_duration_frames)
        is_webcam = p.get("is_webcam", self.is_webcam)
        if not endfade_mode or trigger_frame is None or is_webcam:
            return quad_pts
        trigger = trigger_frame + offset
        if prev_frame_idx < trigger:
            return quad_pts
        progress = min(1.0, (prev_frame_idx - trigger) / max(1, duration))
        progress = progress * progress * (3 - 2 * progress)
        if use_proxy:
            start_quad = p.get("endfade_base_quad_proxy", endfade_base_quad_proxy)
            start_quad = start_quad if start_quad is not None else (quad_pts if quad_pts is not None else np.zeros((4, 2)))
            target_quad = np.array([[0, 0], [proxy_w, 0], [proxy_w, proxy_h], [0, proxy_h]], dtype=np.float32)
        else:
            start_quad = p.get("endfade_base_quad", endfade_base_quad)
            start_quad = start_quad if start_quad is not None else (quad_pts if quad_pts is not None else np.zeros((4, 2)))
            target_quad = np.array([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]], dtype=np.float32)
        active_quad_pts = (start_quad * (1.0 - progress) + target_quad * progress).astype(np.int32)
        if active_quad_pts is not None and quad_pts is None:
            return active_quad_pts
        return active_quad_pts

    def _emit_gui_frame(self, out_frame, fps, force=False):
        ow, oh = out_frame.shape[1], out_frame.shape[0]
        gw, gh = self.target_gui_size
        if gw > 0 and gh > 0:
            scale = min(gw / ow, gh / oh)
            dw, dh = max(1, int(ow * scale)), max(1, int(oh * scale))
            gui_frame = cv2.resize(out_frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            gui_frame = out_frame
        gui_frame = cv2.cvtColor(gui_frame, cv2.COLOR_BGR2RGB)
        if self.is_webcam:
            dbg = "ON" if self.show_debug else "OFF"
            cv2.putText(gui_frame, f"LIVE (WEBCAM) | Debug: {dbg}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
        if force or not self._gui_frame_pending:
            h, w, ch = gui_frame.shape
            bytes_per_line = ch * w
            self._current_qimage_ref = gui_frame
            qimg = QImage(gui_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self._gui_frame_pending = True
            self.frame_ready.emit(qimg)
        if not force and not self.is_webcam:
            target_interval = 1.0 / max(fps, 1.0)
            now = time.perf_counter()
            if self._last_frame_emit_time is not None:
                elapsed = now - self._last_frame_emit_time
                remaining = target_interval - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)
            self._last_frame_emit_time = time.perf_counter()
        else:
            self._last_frame_emit_time = None

    def run(self):
        self._set_flag('_running', True)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            while self._get_flag('_running'):
                # Consume restart request from set_media without blocking the GUI thread.
                self._set_flag('_needs_restart', False)
                session = self._begin_session()
                if session is None:
                    while self._get_flag('_running') and not self._get_flag('_needs_restart'):
                        time.sleep(0.05)
                    if self._get_flag('_needs_restart'):
                        continue
                    break

                fps, frame_w, frame_h, total_frames, proxy_w, proxy_h = session
                future_tracker = None
                prev_frame = None
                proxy_prev_frame = None
                prev_frame_idx = 0
                preview_frame_idx = 0
                webcam_frame_idx = 0
                rec_t0 = None
                rec_n = 0
                rec_pending = []
                rec_writer_fps = None
                webcam_t0 = None

                try:
                    while self._get_flag('_running') and not self._get_flag('_needs_restart'):
                        if self._get_flag('_exporting') and not self.is_webcam:
                            future_tracker = self._drain_future(future_tracker)
                            prev_frame = None
                            proxy_prev_frame = None
                            try:
                                self._run_export_loop(frame_w, frame_h, fps)
                                self.export_finished.emit()
                            except Exception as e:
                                self.error_occurred.emit(str(e))
                            finally:
                                self._set_flag('_exporting', False)
                                self._clear_canvas_cache()
                            continue

                        if self._get_flag('_rendering_webcam') and self.is_webcam:
                            future_tracker = self._drain_future(future_tracker)
                            prev_frame = None
                            proxy_prev_frame = None
                            self._run_webcam_render_loop(frame_w, frame_h, fps)
                            self._set_flag('_rendering_webcam', False)
                            self.recording_finished.emit()
                            continue

                        seek_req = self._get_flag('_seek_request')
                        if seek_req is not None and not self.is_webcam:
                            future_tracker = self._drain_future(future_tracker)
                            seek_req = int(seek_req)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_req)
                            self._set_flag('_seek_request', None)
                            prev_frame = None
                            proxy_prev_frame = None
                            preview_frame_idx = seek_req
                            self.quad_tracker.reset()
                            self.tracker.reset()
                            if self._get_flag('_paused'):
                                self._set_flag('_force_preview_frame', True)

                        if self._get_flag('_paused') and not self.is_webcam and not self._get_flag('_force_preview_frame'):
                            time.sleep(0.05)
                            continue

                        if self._get_flag('_scanning_endfade') and not self.is_webcam:
                            # Drain any in-flight tracker work before synchronous endfade scan (#21/#22).
                            future_tracker = self._drain_future(future_tracker)
                            prev_frame = None
                            proxy_prev_frame = None
                            self.tracker.reset()
                            self.quad_tracker.reset()

                            last_valid_frame = -1
                            last_quad = None
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0
                            while self._get_flag('_running') and self._get_flag('_scanning_endfade') and not self._get_flag('_needs_restart'):
                                ok, f = self.cap.read()
                                if not ok: break
                                proxy_f = cv2.resize(f, (proxy_w, proxy_h), interpolation=cv2.INTER_LINEAR)
                                t_ms = int((frame_idx / fps) * 1000)
                                hr = self.tracker.process(proxy_f, t_ms)
                                sm = self.quad_tracker.update(hr.roles)
                                qp = QuadTracker.ordered_quad_points(sm)
                                if qp is not None:
                                    last_valid_frame = frame_idx
                                    last_quad = qp.copy()

                                frame_idx += 1
                                if frame_idx % 10 == 0:
                                    self.endfade_scan_progress.emit(frame_idx, total_frames)

                            self._set_flag('_scanning_endfade', False)
                            if not self._get_flag('_running') or self._get_flag('_needs_restart'):
                                continue
                            if last_valid_frame != -1 and last_quad is not None:
                                # Match endfade start corners to TL/TR/BR/BL target winding (#24).
                                last_quad = np.round(order_points_tl_tr_br_bl(last_quad)).astype(np.int32)
                                self.endfade_trigger_frame = last_valid_frame
                                self.endfade_base_quad_proxy = last_quad.copy()
                                scale_x = frame_w / proxy_w
                                scale_y = frame_h / proxy_h
                                self.endfade_base_quad = (last_quad * [scale_x, scale_y]).astype(np.int32)
                                self.endfade_scan_complete.emit(last_valid_frame, self.endfade_base_quad)
                            else:
                                self.endfade_mode = False
                                self.endfade_scan_failed.emit("Endfade Scan failed: No hands detected in the video.")

                            # Clear endfade scan timestamp state before returning to preview (#22).
                            self.tracker.reset()
                            self.quad_tracker.reset()

                            if last_valid_frame != -1:
                                resume_idx = max(0, last_valid_frame - int(fps))
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, resume_idx)
                                preview_frame_idx = resume_idx
                            else:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                preview_frame_idx = 0
                            continue

                        ok, frame = self.cap.read()
                        if not ok:
                            if self.is_webcam:
                                time.sleep(0.01)
                                continue
                            else:
                                future_tracker = self._drain_future(future_tracker)
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                preview_frame_idx = 0
                                prev_frame = None
                                proxy_prev_frame = None
                                self.quad_tracker.reset()
                                self.tracker.reset()
                                continue

                        if self.is_webcam:
                            # Track unflipped; mirror only for display/record (#46).
                            current_frame_idx = webcam_frame_idx
                            if webcam_t0 is None:
                                webcam_t0 = time.perf_counter()
                            timestamp_ms = int((time.perf_counter() - webcam_t0) * 1000)
                            webcam_frame_idx += 1
                            record_frame = cv2.flip(frame, 1)
                        else:
                            current_frame_idx = preview_frame_idx
                            timestamp_ms = int((preview_frame_idx / fps) * 1000)
                            preview_frame_idx += 1
                            record_frame = None

                        proxy_frame = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_LINEAR)

                        if self._get_flag('_recording') and self.is_webcam:
                            if rec_t0 is None:
                                rec_t0 = time.perf_counter()
                                rec_n = 0
                                rec_pending = []
                            rec_n += 1
                            if self._raw_writer is None:
                                rec_pending.append(record_frame)
                                dt = time.perf_counter() - rec_t0
                                nbuf = len(rec_pending)
                                if (dt >= 0.4 and nbuf >= 8) or nbuf >= 45 or dt >= 1.5:
                                    rec_writer_fps = self._measured_fps(nbuf, rec_t0, fallback=30.0)
                                    try:
                                        rec_writer_fps = self._open_raw_writer(frame_w, frame_h, rec_writer_fps)
                                    except Exception as e:
                                        self._abort_raw_writer(f"Failed to start raw recording: {e}")
                                        rec_t0 = None
                                        rec_n = 0
                                        rec_pending = []
                                    else:
                                        if not self._write_raw_frames(rec_pending):
                                            rec_t0 = None
                                            rec_n = 0
                                        rec_pending = []
                            else:
                                if not self._write_raw_frames([record_frame]):
                                    rec_t0 = None
                                    rec_n = 0
                                    rec_pending = []
                        elif self.is_webcam and (self._raw_writer is not None or rec_pending):
                            if self._raw_writer is None and rec_pending:
                                rec_writer_fps = self._measured_fps(len(rec_pending), rec_t0, fallback=30.0)
                                try:
                                    rec_writer_fps = self._open_raw_writer(frame_w, frame_h, rec_writer_fps)
                                except Exception as e:
                                    self._abort_raw_writer(f"Failed to start raw recording: {e}")
                                    rec_t0 = None
                                    rec_n = 0
                                    rec_pending = []
                                    continue
                                if not self._write_raw_frames(rec_pending):
                                    rec_t0 = None
                                    rec_n = 0
                                    rec_pending = []
                                    continue
                                rec_pending = []
                            overall_fps = self._measured_fps(rec_n, rec_t0, fallback=rec_writer_fps or 30.0)
                            rec_t0 = None
                            rec_n = 0
                            rec_pending = []
                            rec_writer_fps = None
                            try:
                                self._raw_writer.close()
                            except Exception as e:
                                self.error_occurred.emit(f"Failed to finalize recording: {e}")
                                self._raw_writer = None
                                continue
                            self._raw_writer = None
                            if not os.path.isfile(RAW_TEMP_RECORDING) or os.path.getsize(RAW_TEMP_RECORDING) == 0:
                                self.error_occurred.emit("Recording file missing or empty after stop.")
                                continue
                            self._render_params_snapshot = self._snapshot_render_params(fps=overall_fps)
                            self.raw_recording_ready.emit()

                        current_future = self._executor.submit(self.tracker.process, proxy_frame, timestamp_ms)

                        force_preview = self._get_flag('_force_preview_frame')
                        if force_preview and prev_frame is None:
                            # Seek-while-paused: composite this read immediately (#32).
                            prev_frame = frame
                            proxy_prev_frame = proxy_frame
                            future_tracker = current_future
                            prev_frame_idx = current_frame_idx

                        if prev_frame is not None and future_tracker is not None:
                            if not self.is_webcam:
                                self.position_changed.emit(prev_frame_idx)

                            hand_result = future_tracker.result()

                            self.quad_tracker.coast_limit = self.coast_limit
                            smoothed = self.quad_tracker.update(hand_result.roles)
                            quad_pts = self._quad_pts_for_mode(smoothed)

                            active_quad_pts = self._compute_endfade_quad(
                                prev_frame_idx, quad_pts, frame_w, frame_h, proxy_w, proxy_h,
                                use_proxy=True, endfade_base_quad_proxy=self.endfade_base_quad_proxy,
                            )

                            active_image_bgra = self.image_bgra
                            if not getattr(self, "has_custom_image", True) and self.warp_mode:
                                active_image_bgra = self.grid_bgra

                            if self.warp_mode:
                                out_frame = warp_composite_frame(proxy_prev_frame, active_quad_pts, active_image_bgra, feather=self.feather)
                            else:
                                self._ensure_canvas(active_image_bgra, self.placement, proxy_w, proxy_h, proxy_scale=self.proxy_scale)
                                out_frame = composite_frame(proxy_prev_frame, active_quad_pts, self._cached_canvas_bgr, self._cached_canvas_alpha, feather=self.feather)

                            if self.show_debug:
                                from src.debug_draw import draw_debug_overlay
                                out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, active_quad_pts)

                            if self.is_webcam:
                                out_frame = cv2.flip(out_frame, 1)

                            self._emit_gui_frame(out_frame, fps, force=force_preview)
                            if force_preview:
                                self._set_flag('_force_preview_frame', False)
                                prev_frame = frame
                                proxy_prev_frame = proxy_frame
                                future_tracker = None
                                prev_frame_idx = current_frame_idx
                                continue

                        prev_frame = frame
                        proxy_prev_frame = proxy_frame
                        future_tracker = current_future
                        prev_frame_idx = current_frame_idx

                except Exception as e:
                    traceback.print_exc()
                    self.error_occurred.emit(str(e))
                finally:
                    future_tracker = self._drain_future(future_tracker)
                    self._end_session()

                if self._get_flag('_needs_restart'):
                    continue
                break
        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            if getattr(self, '_raw_writer', None):
                try:
                    self._raw_writer.terminate()
                except Exception:
                    pass
                self._raw_writer = None
            if getattr(self, 'writer', None):
                try:
                    self.writer.terminate()
                except Exception:
                    pass
                self.writer = None
            self._end_session()
            self._set_flag('_running', False)

    def _run_export_loop(self, frame_w, frame_h, fps):
        """Fast headless render loop for exporting a video file"""
        params = self._snapshot_render_params()
        params["is_webcam"] = False
        local_warp_mode = params["warp_mode"]
        local_placement = params["placement"]
        local_feather = params["feather"]
        local_coast_limit = params["coast_limit"]
        local_render_debug = params.get("render_debug", self.render_debug)
        local_codec = params.get("codec", self.codec)

        self._clear_canvas_cache()
        self.tracker.reset()
        self.quad_tracker.reset()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        temp_export_path = os.path.join(USER_DATA_DIR, "temp_export.mp4")
        if os.path.exists(temp_export_path):
            try:
                os.remove(temp_export_path)
            except Exception:
                pass

        writer = FfmpegFrameWriter(temp_export_path, frame_w, frame_h, fps, self.music_path, codec=local_codec, preset=self.preset)
        
        future_tracker = None
        prev_frame = None
        prev_frame_idx = 0
        current_frame_idx = 0

        def process_prev():
            nonlocal prev_frame_idx
            hand_result = future_tracker.result()
            self.quad_tracker.coast_limit = local_coast_limit
            smoothed = self.quad_tracker.update(hand_result.roles)
            quad_pts = self._quad_pts_for_mode(smoothed, warp_mode=local_warp_mode)

            active_quad_pts = self._compute_endfade_quad(
                prev_frame_idx, quad_pts, frame_w, frame_h, frame_w, frame_h,
                use_proxy=False, endfade_base_quad=params.get("endfade_base_quad"),
                params=params,
            )

            active_image_bgra = self.full_image_bgra
            if not getattr(self, "has_custom_image", True) and local_warp_mode:
                active_image_bgra = self.full_grid_bgra

            if local_warp_mode:
                out_frame = warp_composite_frame(prev_frame, active_quad_pts, active_image_bgra, feather=local_feather)
            else:
                self._ensure_canvas(active_image_bgra, local_placement, frame_w, frame_h, warp_mode=local_warp_mode)
                out_frame = composite_frame(prev_frame, active_quad_pts, self._cached_canvas_bgr, self._cached_canvas_alpha, feather=local_feather)
                
            if local_render_debug:
                from src.debug_draw import draw_debug_overlay
                out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, active_quad_pts)
                
            if not writer.write(out_frame) or writer.failed:
                raise RuntimeError(writer.error_message or "Export write failed.")
            
            if prev_frame_idx % 10 == 0:
                self.position_changed.emit(prev_frame_idx)
        
        export_success = False
        try:
            while self._get_flag('_exporting') and self._get_flag('_running'):
                ok, frame = self.cap.read()
                
                if ok:
                    timestamp_ms = int((current_frame_idx / fps) * 1000)
                    current_future = self._executor.submit(self.tracker.process, frame, timestamp_ms)
                else:
                    current_future = None
                    
                if prev_frame is not None and future_tracker is not None:
                    process_prev()
                
                if not ok and prev_frame is None:
                    break
                    
                prev_frame = frame if ok else None
                future_tracker = current_future if ok else None
                prev_frame_idx = current_frame_idx
                if ok:
                    current_frame_idx += 1

            if prev_frame is not None and future_tracker is not None:
                process_prev()
            export_success = True
                    
        finally:
            try:
                writer.close()
            except Exception as e:
                export_success = False
                raise RuntimeError(f"Failed to finalize export: {e}") from e
            finally:
                if export_success and not writer.failed:
                    self.duration_changed.emit(current_frame_idx)
                    self.position_changed.emit(current_frame_idx)
                    dest_dir = os.path.dirname(self.export_path)
                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                    if os.path.exists(self.export_path):
                        try:
                            os.remove(self.export_path)
                        except Exception:
                            pass
                    shutil.move(temp_export_path, self.export_path)
                else:
                    if os.path.exists(temp_export_path):
                        try:
                            os.remove(temp_export_path)
                        except Exception:
                            pass
            self._clear_canvas_cache()

    def _run_webcam_render_loop(self, frame_w, frame_h, fps):
        """Offline full-res render loop for webcam recordings"""
        if self._render_params_snapshot is None:
            return
            
        raw_path = RAW_TEMP_RECORDING
        if not os.path.isfile(raw_path):
            self.error_occurred.emit("Raw recording file missing.")
            return
            
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            self.error_occurred.emit("Could not read raw recording.")
            return

        self.tracker.reset()
        self.quad_tracker.reset()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            self.duration_changed.emit(total_frames)

        params = self._render_params_snapshot
        fps = self._clamp_fps(params.get("fps", fps), fps)
        local_codec = params.get("codec", self.codec)
        local_warp_mode = params.get("warp_mode", False)
        local_placement = params.get("placement", {"x": 960, "y": 540, "scale": 1.0, "rotation_deg": 0.0})
        local_feather = params.get("feather", 9)
        local_coast_limit = params.get("coast_limit", 15)
        local_endfade_mode = params.get("endfade_mode", False)
        local_endfade_trigger_frame = params.get("endfade_trigger_frame", None)
        local_endfade_offset = params.get("endfade_offset", 0)
        local_endfade_duration_frames = params.get("endfade_duration_frames", 60)
        local_endfade_base_quad = params.get("endfade_base_quad", None)
        local_render_debug = params.get("render_debug", self.render_debug)
        local_record_mic = params.get("record_mic", self.record_mic)

        active_audio = self.music_path
        if not active_audio and local_record_mic and os.path.isfile(RAW_TEMP_MIC_AUDIO) and os.path.getsize(RAW_TEMP_MIC_AUDIO) > 100:
            active_audio = RAW_TEMP_MIC_AUDIO

        writer = FfmpegFrameWriter(TEMP_RECORDING, frame_w, frame_h, fps, active_audio, codec=local_codec, preset=self.preset)

        future_tracker = None
        prev_frame = None
        prev_frame_idx = 0
        current_frame_idx = 0

        local_cached_canvas_bgr = None
        local_cached_canvas_alpha = None
        local_last_placement = None
        local_cached_canvas_size = None

        def ensure_local_canvas(active_image_bgra):
            nonlocal local_cached_canvas_bgr, local_cached_canvas_alpha, local_last_placement, local_cached_canvas_size
            cache_key_size = (frame_w, frame_h)
            if not getattr(self, "has_custom_image", True) and not local_warp_mode:
                if local_cached_canvas_bgr is None or local_cached_canvas_size != cache_key_size:
                    local_cached_canvas_bgr = active_image_bgra[:, :, :3].copy()
                    local_cached_canvas_alpha = active_image_bgra[:, :, 3].copy()
                    local_cached_canvas_size = cache_key_size
                return
            if local_last_placement != local_placement or local_cached_canvas_bgr is None or local_cached_canvas_size != cache_key_size:
                local_cached_canvas_bgr, local_cached_canvas_alpha = build_canvas(active_image_bgra, local_placement, frame_w, frame_h)
                local_last_placement = local_placement.copy()
                local_cached_canvas_size = cache_key_size

        def compute_local_endfade(prev_idx, quad_pts):
            if not local_endfade_mode or local_endfade_trigger_frame is None:
                return quad_pts
            trigger = local_endfade_trigger_frame + local_endfade_offset
            if prev_idx < trigger:
                return quad_pts
            progress = min(1.0, (prev_idx - trigger) / max(1, local_endfade_duration_frames))
            progress = progress * progress * (3 - 2 * progress)
            start_quad = local_endfade_base_quad if local_endfade_base_quad is not None else (quad_pts if quad_pts is not None else np.zeros((4, 2)))
            target_quad = np.array([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]], dtype=np.float32)
            active_quad_pts = (start_quad * (1.0 - progress) + target_quad * progress).astype(np.int32)
            if active_quad_pts is not None and quad_pts is None:
                return active_quad_pts
            return active_quad_pts

        def process_prev():
            nonlocal prev_frame_idx
            hand_result = future_tracker.result()
            self.quad_tracker.coast_limit = local_coast_limit
            smoothed = self.quad_tracker.update(hand_result.roles)
            quad_pts = self._quad_pts_for_mode(smoothed, warp_mode=local_warp_mode)

            active_quad_pts = compute_local_endfade(prev_frame_idx, quad_pts)

            active_image_bgra = self.full_image_bgra
            if not getattr(self, "has_custom_image", True) and local_warp_mode:
                active_image_bgra = self.full_grid_bgra

            if local_warp_mode:
                out_frame = warp_composite_frame(prev_frame, active_quad_pts, active_image_bgra, feather=local_feather)
            else:
                ensure_local_canvas(active_image_bgra)
                out_frame = composite_frame(prev_frame, active_quad_pts, local_cached_canvas_bgr, local_cached_canvas_alpha, feather=local_feather)
                        
            if local_render_debug:
                from src.debug_draw import draw_debug_overlay
                out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, active_quad_pts)

            out_frame = cv2.flip(out_frame, 1)
            if not writer.write(out_frame) or writer.failed:
                raise RuntimeError(writer.error_message or "Webcam render write failed.")
            
            if prev_frame_idx % 10 == 0:
                self.position_changed.emit(prev_frame_idx)
        
        try:
            success = False
            while self._get_flag('_running'):
                ok, frame = cap.read()
                
                if ok:
                    frame = cv2.flip(frame, 1)
                    timestamp_ms = int((current_frame_idx / fps) * 1000)
                    current_future = self._executor.submit(self.tracker.process, frame, timestamp_ms)
                else:
                    current_future = None
                    
                if prev_frame is not None and future_tracker is not None:
                    process_prev()
                
                if not ok and prev_frame is None:
                    break
                    
                prev_frame = frame if ok else None
                future_tracker = current_future if ok else None
                prev_frame_idx = current_frame_idx
                if ok:
                    current_frame_idx += 1

            if prev_frame is not None and future_tracker is not None:
                process_prev()
            success = True
                    
        finally:
            try:
                writer.close()
            except Exception as e:
                success = False
                self.error_occurred.emit(f"Failed to finalize webcam render: {e}")
            cap.release()
            if success:
                self.duration_changed.emit(current_frame_idx)
                self.position_changed.emit(current_frame_idx)
                try:
                    os.remove(raw_path)
                except Exception:
                    pass


    def stop(self):
        self._set_flag('_running', False)
        self._set_flag('_needs_restart', False)
        if getattr(self, '_raw_writer', None):
            try:
                self._raw_writer.terminate()
            except Exception:
                pass
            self._raw_writer = None
        if getattr(self, 'writer', None):
            try:
                self.writer.terminate()
            except Exception:
                pass
            self.writer = None
        self.wait()


class ImageAdjustDialog(QDialog):
    def __init__(self, parent, frame_size, placement, warp_mode):
        super().__init__(parent)
        self.setWindowTitle("Image Adjust Editor")
        self.setMinimumSize(820, 500)
        layout = QVBoxLayout(self)
        self.preview = PreviewLabel()
        self.preview.setMinimumSize(800, 450)
        self.preview.frame_size = frame_size
        self.preview.warp_mode = warp_mode
        self.preview.update_placement(placement)
        self.preview.setText("")
        layout.addWidget(self.preview)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overclock Video")
        self.setMinimumSize(1000, 750)
        self.apply_clean_white_theme()

        self.processor = VideoProcessorThread()
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.error_occurred.connect(self.show_error)
        self.processor.duration_changed.connect(self.update_timeline_range)
        self.processor.position_changed.connect(self.update_timeline_pos)
        self.processor.raw_recording_ready.connect(self.on_raw_recording_ready)
        self.processor.recording_finished.connect(self.on_recording_finished)
        self.processor.export_finished.connect(self.on_export_finished)
        self.processor.endfade_scan_progress.connect(self.on_endfade_progress)
        self.processor.endfade_scan_complete.connect(self.on_endfade_complete)
        self.processor.endfade_scan_failed.connect(self.on_endfade_scan_failed)
        self.processor.resolution_changed.connect(self.on_resolution_changed)
        
        self.init_ui()
        self._is_webcam = False
        self._user_is_seeking = False
        self._image_adjust_dialog = None
        self._last_preview_pixmap = None
        self._last_frame_size = None
        self._pending_restore_placement = None
        self._try_restore_session()

    def apply_clean_white_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow { background-color: white; color: black; }
            QLabel { color: black; font-size: 14px; }
            QCheckBox { color: black; font-size: 14px; }
            QPushButton { 
                background-color: #f0f0f0; border: 1px solid #ccc; 
                border-radius: 4px; padding: 6px; color: black; font-size: 14px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #d0d0d0; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 6px; margin-top: 10px; padding-top: 10px; color: black; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
        """)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left Panel (Settings)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setSpacing(15)

        media_group = QGroupBox("Media Inputs")
        media_layout = QFormLayout(media_group)
        
        self.btn_webcam = QPushButton("Use Webcam")
        self.btn_webcam.clicked.connect(self.select_webcam)
        media_layout.addRow("Live:", self.btn_webcam)

        self.btn_video = QPushButton("Select Video File")
        self.btn_video.clicked.connect(self.select_video)
        media_layout.addRow("Video:", self.btn_video)

        self.btn_image = QPushButton("Select Image")
        self.btn_image.clicked.connect(self.select_image)
        media_layout.addRow("Image:", self.btn_image)

        self.btn_music = QPushButton("Select Music (Optional)")
        self.btn_music.clicked.connect(self.select_music)
        media_layout.addRow("Music:", self.btn_music)

        self.cb_record_mic = QCheckBox("Record Mic (when no music)")
        self.cb_record_mic.setToolTip("When recording with webcam, captures microphone audio if no music track is selected.")
        self.cb_record_mic.setChecked(True)
        self.cb_record_mic.stateChanged.connect(self.on_settings_changed)
        media_layout.addRow("Mic:", self.cb_record_mic)

        left_panel.addWidget(media_group)

        comp_group = QGroupBox("Compositing")
        comp_layout = QFormLayout(comp_group)

        self.cb_warp = QCheckBox("Enable Warp Mode")
        self.cb_warp.setToolTip("Morphs and stretches the image dynamically to precisely match the 3D contour of your hands.\nIf disabled, the image acts as a static background layer (Standard Mode).")
        self.cb_warp.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_warp)

        self.cb_debug = QCheckBox("Show Tracking Debug")
        self.cb_debug.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_debug)
        
        self.cb_endfade = QCheckBox("Enable Endfade Mode")
        self.cb_endfade.setToolTip("Automatically identifies when your hands leave the frame near the end of the video,\nand transitions the image to fill the screen.")
        self.cb_endfade.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_endfade)

        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setToolTip("Applies a soft blur to the mask edges to blend the image seamlessly with the video.")
        self.slider_feather.setRange(0, 31)
        self.slider_feather.setValue(9)
        self.slider_feather.valueChanged.connect(self.on_settings_changed)
        comp_layout.addRow("Feather:", self.slider_feather)

        self.slider_coast = QSlider(Qt.Orientation.Horizontal)
        self.slider_coast.setToolTip("How many frames the image persists after tracking is lost.\nUseful to cover up brief moments where MediaPipe fails to see your hands.")
        self.slider_coast.setRange(0, 60)
        self.slider_coast.setValue(15)
        self.slider_coast.valueChanged.connect(self.on_settings_changed)
        comp_layout.addRow("Coast Limit:", self.slider_coast)

        left_panel.addWidget(comp_group)
        
        self.endfade_group = QGroupBox("Endfade Editor")
        endfade_layout = QFormLayout(self.endfade_group)
        self.endfade_group.hide()
        
        self.slider_ef_offset = QSlider(Qt.Orientation.Horizontal)
        self.slider_ef_offset.setRange(-120, 120)
        self.slider_ef_offset.setValue(0)
        self.slider_ef_offset.valueChanged.connect(self.on_settings_changed_slider)
        endfade_layout.addRow("Trigger Offset:", self.slider_ef_offset)
        
        self.slider_ef_dur = QSlider(Qt.Orientation.Horizontal)
        self.slider_ef_dur.setRange(15, 180)
        self.slider_ef_dur.setValue(60)
        self.slider_ef_dur.valueChanged.connect(self.on_settings_changed_slider)
        endfade_layout.addRow("Duration (frames):", self.slider_ef_dur)
        
        self.lbl_ef_status = QLabel("Ready")
        endfade_layout.addRow(self.lbl_ef_status)
        
        left_panel.addWidget(self.endfade_group)

        place_group = QGroupBox("Static Image Placement")
        place_layout = QFormLayout(place_group)

        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(1, 300)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.on_settings_changed_slider)
        place_layout.addRow("Scale %:", self.slider_scale)

        self.slider_rot = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot.setRange(-180, 180)
        self.slider_rot.setValue(0)
        self.slider_rot.valueChanged.connect(self.on_settings_changed_slider)
        place_layout.addRow("Rotation:", self.slider_rot)
        
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setRange(0, 1920)
        self.slider_x.setValue(960)
        self.slider_x.valueChanged.connect(self.on_settings_changed_slider)
        place_layout.addRow("X Offset:", self.slider_x)

        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setRange(0, 1080)
        self.slider_y.setValue(540)
        self.slider_y.valueChanged.connect(self.on_settings_changed_slider)
        place_layout.addRow("Y Offset:", self.slider_y)

        self.btn_image_editor = QPushButton("Open Image Adjust Editor")
        self.btn_image_editor.clicked.connect(self.open_image_adjust_editor)
        place_layout.addRow(self.btn_image_editor)

        left_panel.addWidget(place_group)

        enc_group = QGroupBox("Encoding")
        enc_layout = QFormLayout(enc_group)
        self.cb_codec = QComboBox()
        self.cb_codec.setToolTip(
            "Select the hardware encoder for exporting:\n"
            "- libx264 (CPU): Slowest but highest compatibility. Uses your processor.\n"
            "- h264_nvenc (NVIDIA): Blazing fast GPU acceleration for NVIDIA graphics cards.\n"
            "- h264_qsv (Intel): Quick Sync Video for Intel integrated graphics.\n"
            "- h264_amf (AMD): Hardware encoding for AMD Radeon GPUs."
        )
        self.cb_codec.addItems(["libx264 (CPU)", "h264_nvenc (NVIDIA)", "h264_qsv (Intel)", "h264_amf (AMD)"])
        self.cb_codec.currentIndexChanged.connect(self.on_settings_changed)
        enc_layout.addRow("Codec:", self.cb_codec)
        left_panel.addWidget(enc_group)

        left_panel.addStretch()

        # Right Panel (Video Preview & Controls)
        right_panel = QVBoxLayout()
        
        self.video_label = PreviewLabel()
        self.video_label.placement_changed.connect(self.on_interactive_placement)
        self.video_label.size_changed.connect(self.processor.set_gui_size)
        right_panel.addWidget(self.video_label, 1)
        
        # Bottom Controls
        self.bottom_controls = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_controls)
        
        # Video mode controls
        self.btn_play_pause = QPushButton("Pause")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.hide()
        bottom_layout.addWidget(self.btn_play_pause)
        
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.sliderPressed.connect(self.on_timeline_pressed)
        self.timeline.sliderReleased.connect(self.on_timeline_released)
        self.timeline.hide()
        bottom_layout.addWidget(self.timeline)
        
        # Webcam mode controls
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold;")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.hide()
        bottom_layout.addWidget(self.btn_record)
        
        # Render Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        bottom_layout.addWidget(self.progress_bar)
        
        # Export Options
        self.export_options_group = QGroupBox("Export Options")
        export_opt_layout = QVBoxLayout(self.export_options_group)
        
        dir_layout = QHBoxLayout()
        default_out_file = os.path.join(OUTPUT_DIR, "output.mp4")
        self.le_out_dir = QLineEdit(default_out_file)
        self.btn_out_dir = QPushButton("Browse...")
        self.btn_out_dir.clicked.connect(self.select_out_dir)
        dir_layout.addWidget(QLabel("Output File:"))
        dir_layout.addWidget(self.le_out_dir)
        dir_layout.addWidget(self.btn_out_dir)
        export_opt_layout.addLayout(dir_layout)
        
        self.cb_render_debug = QCheckBox("Render Debug Overlays")
        self.cb_render_debug.setToolTip("Renders the skeletal tracking and quad corners into the final exported video.")
        self.cb_render_debug.stateChanged.connect(self.on_settings_changed)
        export_opt_layout.addWidget(self.cb_render_debug)
        
        # Unified Export button
        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self.do_export)
        self.btn_export.setEnabled(False) # Enabled when video loaded, or webcam recording finished
        export_opt_layout.addWidget(self.btn_export)
        
        right_panel.addWidget(self.bottom_controls)
        right_panel.addWidget(self.export_options_group)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

        self.setCentralWidget(main_widget)
        
        self.video_path = None
        self.image_path = None
        self.music_path = None
        
        self._prevent_slider_feedback = False

    def _save_session(self):
        save_last_run({
            "video": self.video_path,
            "image": self.image_path,
            "music": self.music_path,
            "is_webcam": self._is_webcam,
            "record_mic": self.cb_record_mic.isChecked(),
            "out_file": resolve_output_file(self.le_out_dir.text()),
            "placement": {
                "x": self.slider_x.value(),
                "y": self.slider_y.value(),
                "scale": self.slider_scale.value() / 100.0,
                "rotation_deg": self.slider_rot.value(),
            },
            "warp_mode": self.cb_warp.isChecked(),
            "endfade_mode": self.cb_endfade.isChecked(),
            "endfade_offset": self.slider_ef_offset.value(),
            "endfade_duration": self.slider_ef_dur.value(),
            "feather": self.slider_feather.value(),
            "coast_limit": self.slider_coast.value(),
            "codec": self.cb_codec.currentText().split(" ")[0],
        })

    def _apply_restored_compositing(self, settings):
        widgets = (
            self.cb_warp, self.cb_endfade, self.cb_record_mic, self.cb_codec,
            self.slider_feather, self.slider_coast,
            self.slider_ef_offset, self.slider_ef_dur,
            self.slider_scale, self.slider_rot,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            if "record_mic" in settings:
                self.cb_record_mic.setChecked(bool(settings["record_mic"]))
            if "warp_mode" in settings:
                self.cb_warp.setChecked(bool(settings["warp_mode"]))
            if "endfade_mode" in settings:
                self.cb_endfade.setChecked(bool(settings["endfade_mode"]))
            if "feather" in settings:
                self.slider_feather.setValue(max(0, min(31, int(settings["feather"]))))
            if "coast_limit" in settings:
                self.slider_coast.setValue(max(0, min(60, int(settings["coast_limit"]))))
            if "endfade_offset" in settings:
                self.slider_ef_offset.setValue(max(-120, min(120, int(settings["endfade_offset"]))))
            if "endfade_duration" in settings:
                self.slider_ef_dur.setValue(max(15, min(180, int(settings["endfade_duration"]))))
            placement = settings.get("placement")
            if isinstance(placement, dict):
                scale_pct = int(round(float(placement.get("scale", 1.0)) * 100))
                rot = int(round(float(placement.get("rotation_deg", 0.0))))
                self.slider_scale.setValue(max(1, min(300, scale_pct)))
                self.slider_rot.setValue(max(-180, min(180, rot)))
            codec = settings.get("codec")
            if codec:
                for i in range(self.cb_codec.count()):
                    if self.cb_codec.itemText(i).split(" ")[0] == codec:
                        self.cb_codec.setCurrentIndex(i)
                        break
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _try_restore_session(self):
        last_run = load_last_run()
        if not last_run or not last_run.get("video"):
            return
        reply = QMessageBox.question(
            self, "Restore Session",
            "Restore settings from your last session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if last_run.get("out_file"):
            self.le_out_dir.setText(resolve_output_file(last_run["out_file"]))
        elif last_run.get("out_dir"):
            self.le_out_dir.setText(resolve_output_file(os.path.join(resolve_output_dir(last_run["out_dir"]), "output.mp4")))
        if last_run.get("is_webcam") or str(last_run.get("video")) in ("0", 0):
            self._is_webcam = True
            self.video_path = "0"
            self.btn_webcam.setText("Webcam: Active")
        else:
            self._is_webcam = False
            self.video_path = last_run["video"]
            if os.path.isfile(self.video_path):
                self.btn_video.setText(os.path.basename(self.video_path))
        if last_run.get("image") and os.path.isfile(last_run["image"]):
            self.image_path = last_run["image"]
            self.btn_image.setText(os.path.basename(self.image_path))
        if last_run.get("music") and os.path.isfile(last_run["music"]):
            self.music_path = last_run["music"]
            self.btn_music.setText(os.path.basename(self.music_path))
        settings = session_compositing_settings(last_run)
        self._apply_restored_compositing(settings)
        if "placement" in settings:
            self._pending_restore_placement = settings["placement"]
        if self.video_path:
            self.update_controls_mode()
            self.start_playback()

    def open_image_adjust_editor(self):
        if self._image_adjust_dialog is not None and self._image_adjust_dialog.isVisible():
            self._image_adjust_dialog.raise_()
            self._image_adjust_dialog.activateWindow()
            return
        placement = {
            "x": self.slider_x.value(),
            "y": self.slider_y.value(),
            "scale": self.slider_scale.value() / 100.0,
            "rotation_deg": self.slider_rot.value(),
        }
        self._image_adjust_dialog = ImageAdjustDialog(
            self, self.video_label.frame_size, placement, self.cb_warp.isChecked(),
        )
        if self._last_preview_pixmap is not None:
            self._image_adjust_dialog.preview.setPixmap(self._last_preview_pixmap)
        self._image_adjust_dialog.preview.placement_changed.connect(self.on_interactive_placement)
        self._image_adjust_dialog.finished.connect(self._on_image_editor_closed)
        self._image_adjust_dialog.show()

    def _on_image_editor_closed(self):
        if self._image_adjust_dialog:
            try:
                self._image_adjust_dialog.preview.placement_changed.disconnect(self.on_interactive_placement)
            except (TypeError, RuntimeError):
                pass
            try:
                self._image_adjust_dialog.finished.disconnect(self._on_image_editor_closed)
            except (TypeError, RuntimeError):
                pass
            self._image_adjust_dialog = None

    @pyqtSlot(int, int)
    def on_resolution_changed(self, frame_w, frame_h):
        self.video_label.frame_size = (frame_w, frame_h)
        self.slider_x.setRange(0, frame_w)
        self.slider_y.setRange(0, frame_h)
        size_changed = self._last_frame_size != (frame_w, frame_h)
        if size_changed:
            pending = self._pending_restore_placement
            self._pending_restore_placement = None
            self._prevent_slider_feedback = True
            if pending:
                x = int(round(max(0, min(frame_w, float(pending.get("x", frame_w / 2.0))))))
                y = int(round(max(0, min(frame_h, float(pending.get("y", frame_h / 2.0))))))
                scale = float(pending.get("scale", 1.0))
                rot = float(pending.get("rotation_deg", 0.0))
                self.slider_x.setValue(x)
                self.slider_y.setValue(y)
                self.slider_scale.setValue(max(1, min(300, int(round(scale * 100)))))
                self.slider_rot.setValue(max(-180, min(180, int(round(rot)))))
                placement = {
                    "x": x,
                    "y": y,
                    "scale": self.slider_scale.value() / 100.0,
                    "rotation_deg": float(self.slider_rot.value()),
                }
            else:
                cx, cy = frame_w // 2, frame_h // 2
                self.slider_x.setValue(cx)
                self.slider_y.setValue(cy)
                placement = {"x": cx, "y": cy}
            self._prevent_slider_feedback = False
            self.processor.update_params({"placement": placement})
            self.video_label.update_placement(placement)
            self._last_frame_size = (frame_w, frame_h)
        if self._image_adjust_dialog and self._image_adjust_dialog.isVisible():
            self._image_adjust_dialog.preview.frame_size = (frame_w, frame_h)
            if size_changed:
                self._image_adjust_dialog.preview.update_placement({"x": frame_w // 2, "y": frame_h // 2})

    def select_webcam(self):
        self._is_webcam = True
        self.video_path = "0"
        self.btn_video.setText("Select Video File")
        self.btn_webcam.setText("Webcam: Active")
        self.update_controls_mode()
        self.start_playback()
        self._save_session()

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", MEDIA_DIR, "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self._is_webcam = False
            self.video_path = path
            self.btn_video.setText(os.path.basename(path))
            self.btn_webcam.setText("Use Webcam")
            self.update_controls_mode()
            self.start_playback()
            self._save_session()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", MEDIA_DIR, "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.image_path = path
            self.btn_image.setText(os.path.basename(path))
            self.start_playback()
            self._save_session()

    def select_music(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Music", MEDIA_DIR, "Audio Files (*.mp3 *.wav *.m4a *.flac *.aac)")
        if path:
            self.music_path = path
            self.btn_music.setText(os.path.basename(path))
            self.start_playback()
            self._save_session()

    def update_controls_mode(self):
        if self._is_webcam:
            self.timeline.hide()
            self.btn_play_pause.hide()
            self.btn_record.show()
            self.btn_export.setEnabled(os.path.exists(TEMP_RECORDING))
        else:
            self.btn_record.hide()
            self.timeline.show()
            self.btn_play_pause.show()
            self.btn_export.setEnabled(True)

    def start_playback(self):
        if not self.video_path:
            return
            
        self.btn_webcam.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_image.setEnabled(False)
        self.btn_music.setEnabled(False)
        QApplication.processEvents()
        
        self.processor.set_media(self.video_path, self.image_path, self.music_path, is_webcam=self._is_webcam)
        
        self.btn_webcam.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_image.setEnabled(True)
        self.btn_music.setEnabled(True)
        self.on_settings_changed()
        if not self.processor.isRunning():
            self.processor.start()

    def on_settings_changed_slider(self):
        if not self._prevent_slider_feedback:
            self.on_settings_changed()
        placement = {
            "x": self.slider_x.value(),
            "y": self.slider_y.value(),
            "scale": self.slider_scale.value() / 100.0,
            "rotation_deg": self.slider_rot.value(),
        }
        self.video_label.update_placement(placement)
        if self._image_adjust_dialog and self._image_adjust_dialog.isVisible():
            self._image_adjust_dialog.preview.update_placement(placement)

    def on_settings_changed(self):
        self.video_label.warp_mode = self.cb_warp.isChecked()
        if self._image_adjust_dialog and self._image_adjust_dialog.isVisible():
            self._image_adjust_dialog.preview.warp_mode = self.cb_warp.isChecked()
        codec_str = self.cb_codec.currentText().split(" ")[0]
        
        is_endfade = self.cb_endfade.isChecked()
        if is_endfade and not self._is_webcam and not self.processor.endfade_mode:
            self.lbl_ef_status.setText("Scanning video...")
            self.endfade_group.show()
        elif not is_endfade:
            self.endfade_group.hide()
            
        params = {
            "record_mic": self.cb_record_mic.isChecked(),
            "warp_mode": self.cb_warp.isChecked(),
            "feather": self.slider_feather.value(),
            "coast_limit": self.slider_coast.value(),
            "show_debug": self.cb_debug.isChecked(),
            "render_debug": self.cb_render_debug.isChecked(),
            "endfade_mode": is_endfade,
            "endfade_offset": self.slider_ef_offset.value(),
            "endfade_duration": self.slider_ef_dur.value(),
            "codec": codec_str,
            "placement": {
                "x": self.slider_x.value(),
                "y": self.slider_y.value(),
                "scale": self.slider_scale.value() / 100.0,
                "rotation_deg": self.slider_rot.value(),
            }
        }
        self.processor.update_params(params)
        
    @pyqtSlot(int, int)
    def on_endfade_progress(self, current, total):
        pct = int((current / total) * 100) if total > 0 else 0
        self.lbl_ef_status.setText(f"Scanning... {pct}%")
        
    @pyqtSlot(int, np.ndarray)
    def on_endfade_complete(self, trigger_frame, last_quad):
        self.lbl_ef_status.setText(f"Trigger: Frame {trigger_frame}")

    @pyqtSlot(str)
    def on_endfade_scan_failed(self, msg):
        self.cb_endfade.blockSignals(True)
        self.cb_endfade.setChecked(False)
        self.cb_endfade.blockSignals(False)
        self.endfade_group.hide()
        self.lbl_ef_status.setText("Scan failed")
        QMessageBox.critical(self, "Error", msg)

    def on_interactive_placement(self, placement):
        self._prevent_slider_feedback = True
        self.slider_x.setValue(int(placement["x"]))
        self.slider_y.setValue(int(placement["y"]))
        self.slider_scale.setValue(int(placement["scale"] * 100))
        self.slider_rot.setValue(int(placement["rotation_deg"]))
        self._prevent_slider_feedback = False
        
        self.processor.update_params({"placement": placement})
        self.video_label.update_placement(placement)
        if self._image_adjust_dialog and self._image_adjust_dialog.isVisible():
            self._image_adjust_dialog.preview.update_placement(placement)

    def toggle_play_pause(self):
        if self.processor._get_flag('_paused'):
            self.processor.set_paused(False)
            self.btn_play_pause.setText("Pause")
        else:
            self.processor.set_paused(True)
            self.btn_play_pause.setText("Play")
            
    def toggle_recording(self):
        if self.processor._get_flag('_recording'):
            self.processor.stop_recording()
        else:
            self.processor.start_recording()
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: #ffcccc; color: #cc0000; font-weight: bold; border: 1px solid #cc0000;")

    def on_raw_recording_ready(self):
        self.btn_record.setEnabled(False)
        self.btn_record.setText("Rendering...")
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.processor.request_webcam_render()
        
    def on_recording_finished(self):
        self.btn_record.setEnabled(True)
        self.btn_record.setText("Start Recording")
        self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold;")
        self.btn_export.setEnabled(True)
        self.progress_bar.hide()
        
        if os.path.exists(TEMP_RECORDING):
            default_out = os.path.join(PROJECT_ROOT, "output", "webcam_recording.mp4")
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Recorded Video",
                default_out,
                "Video Files (*.mp4);;All Files (*.*)",
            )
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    shutil.copy2(TEMP_RECORDING, save_path)
                    QMessageBox.information(self, "Recording Saved", f"Saved recording to:\n{save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save recording:\n{e}")

    def select_out_dir(self):
        current_path = self.le_out_dir.text() or os.path.join(OUTPUT_DIR, "output.mp4")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            current_path,
            "Video Files (*.mp4);;All Files (*.*)",
        )
        if file_path:
            self.le_out_dir.setText(resolve_output_file(file_path))

    def do_export(self):
        out_path = resolve_output_file(self.le_out_dir.text())
        self.le_out_dir.setText(out_path)
            
        self.processor.update_params({"render_debug": self.cb_render_debug.isChecked()})
        
        if self._is_webcam:
            # We already recorded to TEMP_RECORDING, just copy it
            if os.path.exists(TEMP_RECORDING):
                try:
                    shutil.copy2(TEMP_RECORDING, out_path)
                    QMessageBox.information(self, "Export Complete", f"Saved to {out_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", str(e))
        else:
            # Render the video in background
            self.btn_export.setEnabled(False)
            self.btn_export.setText("Exporting...")
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.processor.start_export(out_path)
            
    def on_export_finished(self):
        self.btn_export.setEnabled(True)
        self.btn_export.setText("Export")
        self.progress_bar.hide()
        QMessageBox.information(self, "Export Complete", "Video rendering finished successfully.")

    def on_timeline_pressed(self):
        self._user_is_seeking = True

    def on_timeline_released(self):
        self._user_is_seeking = False
        self.processor.request_seek(self.timeline.value())

    @pyqtSlot(int)
    def update_timeline_range(self, total_frames):
        max_idx = max(0, total_frames - 1) if total_frames > 0 else 0
        self.timeline.setRange(0, max_idx)
        self.progress_bar.setRange(0, max(1, total_frames))

    @pyqtSlot(int)
    def update_timeline_pos(self, frame_idx):
        if not self._user_is_seeking:
            self.timeline.blockSignals(True)
            self.timeline.setValue(frame_idx)
            self.timeline.blockSignals(False)
        self.progress_bar.setValue(frame_idx)

    @pyqtSlot(QImage)
    def update_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self._last_preview_pixmap = pixmap
        self.video_label.setPixmap(pixmap)
        if self._image_adjust_dialog and self._image_adjust_dialog.isVisible():
            self._image_adjust_dialog.preview.setPixmap(pixmap)
        self.processor.ack_frame()

    @pyqtSlot(str)
    def show_error(self, msg):
        self.btn_export.setEnabled(not self._is_webcam or os.path.exists(TEMP_RECORDING))
        self.btn_export.setText("Export")
        self.progress_bar.hide()
        self.btn_record.setEnabled(True)
        self.btn_record.setText("Start Recording")
        self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold;")
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        self._save_session()
        self.processor.stop()
        for temp_path in (TEMP_RECORDING, RAW_TEMP_RECORDING):
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        event.accept()

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
