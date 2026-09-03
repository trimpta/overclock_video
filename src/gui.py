import os
import shutil
import sys
import time
import traceback
import cv2
import numpy as np
import concurrent.futures

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QSlider, QCheckBox, 
    QGroupBox, QFormLayout, QComboBox, QMessageBox, QStyle
)

from src.audio_mux import FfmpegFrameWriter
from src.compositor import build_canvas, composite_frame, warp_composite_frame, load_image_bgra, make_greenscreen_bgra, make_grid_bgra, full_frame_placement
from src.hand_tracking import HandTracker
from src.smoothing import QuadTracker

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hand_landmarker.task")
TEMP_RECORDING = "temp_recording.mp4"


class PreviewLabel(QLabel):
    placement_changed = pyqtSignal(dict)
    size_changed = pyqtSignal(int, int)

    def __init__(self):
        super().__init__("Select a video source to begin.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #fafafa; border: 1px solid #ddd; color: #333;")
        
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

    def mousePressEvent(self, event):
        if self.warp_mode: return
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = event.pos()
            
            # Determine if we are dragging the center or the edges
            w, h = self.width(), self.height()
            x, y = event.pos().x(), event.pos().y()
            margin_x, margin_y = w * 0.2, h * 0.2
            
            # If clicked in the outer 20% border, we scale. If inner 60%, we translate.
            if x < margin_x or x > w - margin_x or y < margin_y or y > h - margin_y:
                self.drag_mode = "scale"
            else:
                self.drag_mode = "translate"

    def mouseMoveEvent(self, event):
        if not self.is_dragging or self.warp_mode: return
        
        delta = event.pos() - self.drag_start_pos
        self.drag_start_pos = event.pos()
        
        # Heuristic coordinate mapping
        scale_x = self.frame_size[0] / max(1, self.width())
        scale_y = self.frame_size[1] / max(1, self.height())
        
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
    recording_finished = pyqtSignal()
    raw_recording_ready = pyqtSignal()
    export_finished = pyqtSignal()
    endfade_scan_progress = pyqtSignal(int, int)
    endfade_scan_complete = pyqtSignal(int, np.ndarray)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.image_path = None
        self.music_path = None
        self.is_webcam = False
        
        self.codec = "libx264"
        self.preset = "medium"
        
        self.warp_mode = False
        self.placement = {"x": 960, "y": 540, "scale": 1.0, "rotation_deg": 0.0}
        self.feather = 9
        self.coast_limit = 0
        self.show_debug = False

        self._running = False
        self._paused = False
        self._seek_request = None
        
        self.endfade_mode = False
        self._scanning_endfade = False
        self.endfade_trigger_frame = None
        self.endfade_base_quad = None
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
        self._executor = None
        self._current_qimage_ref = None
        
        self.proxy_scale = 0.5
        self._raw_writer = None
        self._render_params_snapshot = None
        self._rendering_webcam = False

    def start_recording(self):
        self._recording = True

    def stop_recording(self):
        self._recording = False
        
    def start_export(self, export_path):
        self.export_path = export_path
        self._exporting = True

    def update_params(self, params):
        if "warp_mode" in params: self.warp_mode = params["warp_mode"]
        if "feather" in params: self.feather = params["feather"]
        if "coast_limit" in params: self.coast_limit = params["coast_limit"]
        if "show_debug" in params: self.show_debug = params["show_debug"]
        if "placement" in params:
            self.placement.update(params["placement"])
        if "codec" in params: self.codec = params["codec"]
        if "endfade_mode" in params:
            new_mode = params["endfade_mode"]
            if new_mode and not self.endfade_mode and not self.is_webcam:
                self._scanning_endfade = True
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
        if self._running:
            self.stop()
            self.start()

    def request_seek(self, frame_idx):
        self._seek_request = frame_idx
        
    def set_paused(self, paused):
        self._paused = paused

    def run(self):
        if self.video_path is None: return
        self._running = True
        
        if self.is_webcam:
            self.cap = cv2.VideoCapture(int(self.video_path) if str(self.video_path).isdigit() else 0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.error_occurred.emit(f"Could not open video source: {self.video_path}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 120: fps = 30.0
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = None if self.is_webcam else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not self.is_webcam and total_frames:
            self.duration_changed.emit(total_frames)

        proxy_w = int(frame_w * self.proxy_scale)
        proxy_h = int(frame_h * self.proxy_scale)

        if self.image_path and os.path.isfile(self.image_path):
            full_img = load_image_bgra(self.image_path)
            self.image_bgra = cv2.resize(full_img, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
            self.has_custom_image = True
        else:
            self.image_bgra = make_greenscreen_bgra(proxy_w, proxy_h)
            self.grid_bgra = make_grid_bgra(proxy_w, proxy_h)
            self.has_custom_image = False

        self.tracker = HandTracker(MODEL_PATH, num_hands=2)
        self.quad_tracker = QuadTracker(coast_limit=self.coast_limit)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        start_time = time.time()
        
        future_tracker = None
        prev_frame = None
        proxy_prev_frame = None
        prev_frame_idx = 0
        
        try:
            while self._running:
                # Handle Export for Video Mode
                if self._exporting and not self.is_webcam:
                    self._run_export_loop(frame_w, frame_h, fps)
                    self._exporting = False
                    self.export_finished.emit()
                    continue
                    
                # Handle Offline Rendering for Webcam Mode
                if self._rendering_webcam and self.is_webcam:
                    self._run_webcam_render_loop(frame_w, frame_h, fps)
                    self._rendering_webcam = False
                    self.recording_finished.emit()
                    continue
                
                # Handle Seek
                if self._seek_request is not None and not self.is_webcam:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_request)
                    self._seek_request = None
                    future_tracker = None
                    prev_frame = None
                    
                # Handle Pause
                if self._paused and not self.is_webcam:
                    time.sleep(0.05)
                    continue

                # Handle Endfade Pre-Pass
                if self._scanning_endfade and not self.is_webcam:
                    last_valid_frame = -1
                    last_quad = None
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    while self._running and self._scanning_endfade:
                        ok, f = self.cap.read()
                        if not ok: break
                        proxy_f = cv2.resize(f, (proxy_w, proxy_h), interpolation=cv2.INTER_LINEAR)
                        t_ms = int((frame_idx / fps) * 1000)
                        hr = self.tracker.process(proxy_f, t_ms)
                        sm = self.quad_tracker.update(hr.roles)
                        qp = QuadTracker.quad_points(sm)
                        if qp is not None:
                            last_valid_frame = frame_idx
                            last_quad = qp.copy()
                            
                        frame_idx += 1
                        if frame_idx % 10 == 0:
                            self.endfade_scan_progress.emit(frame_idx, total_frames)
                            
                    self._scanning_endfade = False
                    if last_valid_frame != -1 and last_quad is not None:
                        self.endfade_trigger_frame = last_valid_frame
                        scale_x = frame_w / proxy_w
                        scale_y = frame_h / proxy_h
                        self.endfade_base_quad = (last_quad * [scale_x, scale_y]).astype(np.int32)
                        self.endfade_scan_complete.emit(last_valid_frame, self.endfade_base_quad)
                    else:
                        self.endfade_mode = False
                        self.error_occurred.emit("Endfade Scan failed: No hands detected in the video.")
                        
                    # Reset playhead to loop region
                    if last_valid_frame != -1:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, last_valid_frame - int(fps)))
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                ok, frame = self.cap.read()
                if not ok:
                    if self.is_webcam:
                        time.sleep(0.01)
                        continue
                    else:
                        # Auto-loop video. If endfade mode, loop only the end region
                        if self.endfade_mode and self.endfade_trigger_frame is not None:
                            loop_start = max(0, self.endfade_trigger_frame + self.endfade_offset - int(fps))
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start)
                        else:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        future_tracker = None
                        prev_frame = None
                        continue
                
                if self.is_webcam:
                    frame = cv2.flip(frame, 1)
                
                # Create proxy frame
                proxy_frame = cv2.resize(frame, (proxy_w, proxy_h), interpolation=cv2.INTER_LINEAR)
                
                current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if not self.is_webcam else 0
                timestamp_ms = int((time.time() - start_time) * 1000)
                
                # Handle Raw Recording for Webcam Mode
                if self._recording and self.is_webcam:
                    if self._raw_writer is None:
                        try:
                            # Write raw full-res frames
                            self._raw_writer = FfmpegFrameWriter("raw_temp.mp4", frame_w, frame_h, fps, codec="libx264", preset="ultrafast")
                        except Exception as e:
                            self.error_occurred.emit(f"Failed to start raw recording: {e}")
                            self._recording = False
                    
                    if self._raw_writer:
                        self._raw_writer.write(frame)
                else:
                    if self._raw_writer is not None:
                        self._raw_writer.close()
                        self._raw_writer = None
                        self._render_params_snapshot = {
                            "warp_mode": self.warp_mode,
                            "placement": self.placement.copy(),
                            "feather": self.feather,
                            "endfade_mode": self.endfade_mode,
                            "endfade_offset": self.endfade_offset,
                            "endfade_duration_frames": self.endfade_duration_frames,
                            "endfade_base_quad": self.endfade_base_quad.copy() if self.endfade_base_quad is not None else None,
                            "endfade_trigger_frame": self.endfade_trigger_frame,
                        }
                        self.raw_recording_ready.emit()
                
                # Submit tracking to background executor
                current_future = self._executor.submit(self.tracker.process, proxy_frame, timestamp_ms)

                # Process the previous frame whose tracking is now complete
                if prev_frame is not None and future_tracker is not None:
                    if not self.is_webcam:
                        self.position_changed.emit(prev_frame_idx)
                        
                    hand_result = future_tracker.result()
                    
                    self.quad_tracker.coast_limit = self.coast_limit
                    smoothed = self.quad_tracker.update(hand_result.roles)
                    quad_pts = QuadTracker.quad_points(smoothed)

                    active_quad_pts = quad_pts
                    if self.endfade_mode and self.endfade_trigger_frame is not None and not self.is_webcam:
                        trigger = self.endfade_trigger_frame + self.endfade_offset
                        if prev_frame_idx >= trigger:
                            progress = min(1.0, (prev_frame_idx - trigger) / max(1, self.endfade_duration_frames))
                            progress = progress * progress * (3 - 2 * progress) # ease-in-out
                            
                            start_quad = self.endfade_base_quad if self.endfade_base_quad is not None else (quad_pts if quad_pts is not None else np.zeros((4,2)))
                            target_quad = np.array([
                                [0, 0], [proxy_w, 0], [proxy_w, proxy_h], [0, proxy_h]
                            ], dtype=np.float32)
                            
                            active_quad_pts = (start_quad * (1.0 - progress) + target_quad * progress).astype(np.int32)
                            
                            # In endfade, we override missing quad_pts so it still renders
                            if active_quad_pts is not None and quad_pts is None:
                                quad_pts = active_quad_pts

                    active_image_bgra = self.image_bgra
                    if not getattr(self, "has_custom_image", True) and self.warp_mode:
                        active_image_bgra = getattr(self, "grid_bgra", self.image_bgra)

                    if self.warp_mode:
                        out_frame = warp_composite_frame(proxy_prev_frame, active_quad_pts, active_image_bgra, feather=self.feather)
                    else:
                        if self._last_placement != self.placement or self._cached_canvas_bgr is None:
                            proxy_placement = self.placement.copy()
                            proxy_placement["x"] = proxy_placement["x"] * self.proxy_scale
                            proxy_placement["y"] = proxy_placement["y"] * self.proxy_scale
                            self._cached_canvas_bgr, self._cached_canvas_alpha = build_canvas(active_image_bgra, proxy_placement, proxy_w, proxy_h)
                            self._last_placement = self.placement.copy()
                        out_frame = composite_frame(proxy_prev_frame, active_quad_pts, self._cached_canvas_bgr, self._cached_canvas_alpha, feather=self.feather)

                    if self.show_debug:
                        from src.debug_draw import draw_debug_overlay
                        out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, active_quad_pts)

                    # Fast resize for GUI on background thread
                    gw, gh = self.target_gui_size
                    if gw > 0 and gh > 0:
                        gui_frame = cv2.resize(out_frame, (gw, gh), interpolation=cv2.INTER_LINEAR)
                    else:
                        gui_frame = out_frame

                    # In-place color conversion and zero-copy QImage
                    cv2.cvtColor(gui_frame, cv2.COLOR_BGR2RGB, dst=gui_frame)
                    h, w, ch = gui_frame.shape
                    bytes_per_line = ch * w
                    self._current_qimage_ref = gui_frame # Retain reference to prevent garbage collection
                    qimg = QImage(gui_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(qimg)

                    if not self.is_webcam:
                        # Very simple fps sync
                        time.sleep(1.0 / fps)

                # Advance pipeline
                prev_frame = frame
                proxy_prev_frame = proxy_frame
                future_tracker = current_future
                prev_frame_idx = current_frame_idx

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            if self._executor:
                self._executor.shutdown(wait=False)
            if self.writer:
                self.writer.close()
                self.writer = None
            if self.cap:
                self.cap.release()
            if self.tracker:
                self.tracker.close()

    def _run_export_loop(self, frame_w, frame_h, fps):
        """Fast headless render loop for exporting a video file"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = FfmpegFrameWriter(self.export_path, frame_w, frame_h, fps, self.music_path, codec=self.codec, preset=self.preset)
        
        future_tracker = None
        prev_frame = None
        prev_frame_idx = 0
        current_frame_idx = 0
        
        try:
            while self._exporting and self._running:
                ok, frame = self.cap.read()
                
                # Submit current frame if available
                if ok:
                    timestamp_ms = int((current_frame_idx / fps) * 1000)
                    current_future = self._executor.submit(self.tracker.process, frame, timestamp_ms)
                else:
                    current_future = None
                    
                # Process previous frame
                if prev_frame is not None and future_tracker is not None:
                    hand_result = future_tracker.result()
                    smoothed = self.quad_tracker.update(hand_result.roles)
                    quad_pts = QuadTracker.quad_points(smoothed)

                    active_quad_pts = quad_pts
                    if self.endfade_mode and self.endfade_trigger_frame is not None and not self.is_webcam:
                        trigger = self.endfade_trigger_frame + self.endfade_offset
                        if prev_frame_idx >= trigger:
                            progress = min(1.0, (prev_frame_idx - trigger) / max(1, self.endfade_duration_frames))
                            progress = progress * progress * (3 - 2 * progress)
                            start_quad = self.endfade_base_quad if self.endfade_base_quad is not None else (quad_pts if quad_pts is not None else np.zeros((4,2)))
                            target_quad = np.array([[0,0], [frame_w,0], [frame_w,frame_h], [0,frame_h]], dtype=np.float32)
                            active_quad_pts = (start_quad * (1.0 - progress) + target_quad * progress).astype(np.int32)

                    active_image_bgra = self.image_bgra
                    if not getattr(self, "has_custom_image", True) and self.warp_mode:
                        active_image_bgra = getattr(self, "grid_bgra", self.image_bgra)

                    if self.warp_mode:
                        out_frame = warp_composite_frame(prev_frame, active_quad_pts, active_image_bgra, feather=self.feather)
                    else:
                        if self._last_placement != self.placement or self._cached_canvas_bgr is None:
                            self._cached_canvas_bgr, self._cached_canvas_alpha = build_canvas(active_image_bgra, self.placement, frame_w, frame_h)
                            self._last_placement = self.placement.copy()
                        out_frame = composite_frame(prev_frame, active_quad_pts, self._cached_canvas_bgr, self._cached_canvas_alpha, feather=self.feather)
                        
                    writer.write(out_frame)
                    
                    # Signal progress
                    if prev_frame_idx % 10 == 0:
                        self.position_changed.emit(prev_frame_idx)
                
                if not ok and prev_frame is None:
                    # Finished processing all frames
                    break
                    
                prev_frame = frame if ok else None
                future_tracker = current_future if ok else None
                prev_frame_idx = current_frame_idx
                if ok:
                    current_frame_idx += 1
                    
        finally:
            writer.close()

    def _run_webcam_render_loop(self, frame_w, frame_h, fps):
        """Offline full-res render loop for webcam recordings"""
        if self._render_params_snapshot is None:
            return
            
        raw_path = "raw_temp.mp4"
        if not os.path.isfile(raw_path):
            self.error_occurred.emit("Raw recording file missing.")
            return
            
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            self.error_occurred.emit("Could not read raw recording.")
            return

        writer = FfmpegFrameWriter(TEMP_RECORDING, frame_w, frame_h, fps, self.music_path, codec=self.codec, preset=self.preset)
        
        future_tracker = None
        prev_frame = None
        prev_frame_idx = 0
        current_frame_idx = 0
        
        params = self._render_params_snapshot
        local_warp_mode = params.get("warp_mode", False)
        local_placement = params.get("placement", {"x": 960, "y": 540, "scale": 1.0, "rotation_deg": 0.0})
        local_feather = params.get("feather", 9)
        local_endfade_mode = params.get("endfade_mode", False)
        local_endfade_trigger_frame = params.get("endfade_trigger_frame", None)
        local_endfade_offset = params.get("endfade_offset", 0)
        local_endfade_duration_frames = params.get("endfade_duration_frames", 60)
        local_endfade_base_quad = params.get("endfade_base_quad", None)
        
        local_cached_canvas_bgr = None
        local_cached_canvas_alpha = None
        local_last_placement = None
        
        try:
            while self._running:
                ok, frame = cap.read()
                
                if ok:
                    timestamp_ms = int((current_frame_idx / fps) * 1000)
                    current_future = self._executor.submit(self.tracker.process, frame, timestamp_ms)
                else:
                    current_future = None
                    
                if prev_frame is not None and future_tracker is not None:
                    hand_result = future_tracker.result()
                    self.quad_tracker.coast_limit = self.coast_limit
                    smoothed = self.quad_tracker.update(hand_result.roles)
                    quad_pts = QuadTracker.quad_points(smoothed)

                    active_quad_pts = quad_pts
                    if local_endfade_mode and local_endfade_trigger_frame is not None:
                        trigger = local_endfade_trigger_frame + local_endfade_offset
                        if prev_frame_idx >= trigger:
                            progress = min(1.0, (prev_frame_idx - trigger) / max(1, local_endfade_duration_frames))
                            progress = progress * progress * (3 - 2 * progress)
                            start_quad = local_endfade_base_quad if local_endfade_base_quad is not None else (quad_pts if quad_pts is not None else np.zeros((4,2)))
                            target_quad = np.array([[0,0], [frame_w,0], [frame_w,frame_h], [0,frame_h]], dtype=np.float32)
                            active_quad_pts = (start_quad * (1.0 - progress) + target_quad * progress).astype(np.int32)

                    active_image_bgra = self.image_bgra
                    if not getattr(self, "has_custom_image", True) and local_warp_mode:
                        active_image_bgra = getattr(self, "grid_bgra", self.image_bgra)

                    if local_warp_mode:
                        out_frame = warp_composite_frame(prev_frame, active_quad_pts, active_image_bgra, feather=local_feather)
                    else:
                        if local_last_placement != local_placement or local_cached_canvas_bgr is None:
                            local_cached_canvas_bgr, local_cached_canvas_alpha = build_canvas(active_image_bgra, local_placement, frame_w, frame_h)
                            local_last_placement = local_placement.copy()
                        out_frame = composite_frame(prev_frame, active_quad_pts, local_cached_canvas_bgr, local_cached_canvas_alpha, feather=local_feather)
                        
                    writer.write(out_frame)
                    
                    if prev_frame_idx % 10 == 0:
                        self.position_changed.emit(prev_frame_idx)
                
                if not ok and prev_frame is None:
                    break
                    
                prev_frame = frame if ok else None
                future_tracker = current_future if ok else None
                prev_frame_idx = current_frame_idx
                if ok:
                    current_frame_idx += 1
                    
        finally:
            writer.close()
            cap.release()
            try:
                os.remove(raw_path)
            except Exception:
                pass


    def stop(self):
        self._running = False
        self.wait()


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
        
        self.init_ui()
        self._is_webcam = False
        self._user_is_seeking = False

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

        left_panel.addWidget(media_group)

        comp_group = QGroupBox("Compositing")
        comp_layout = QFormLayout(comp_group)

        self.cb_warp = QCheckBox("Enable Warp Mode")
        self.cb_warp.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_warp)

        self.cb_debug = QCheckBox("Show Tracking Debug")
        self.cb_debug.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_debug)
        
        self.cb_endfade = QCheckBox("Enable Endfade Mode")
        self.cb_endfade.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_endfade)

        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setRange(0, 31)
        self.slider_feather.setValue(9)
        self.slider_feather.valueChanged.connect(self.on_settings_changed)
        comp_layout.addRow("Feather:", self.slider_feather)

        self.slider_coast = QSlider(Qt.Orientation.Horizontal)
        self.slider_coast.setRange(0, 60)
        self.slider_coast.setValue(0)
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

        left_panel.addWidget(place_group)

        enc_group = QGroupBox("Encoding")
        enc_layout = QFormLayout(enc_group)
        self.cb_codec = QComboBox()
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
        self.timeline.valueChanged.connect(self.on_timeline_moved)
        self.timeline.hide()
        bottom_layout.addWidget(self.timeline)
        
        # Webcam mode controls
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold;")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.hide()
        bottom_layout.addWidget(self.btn_record)
        
        # Unified Export button
        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self.do_export)
        self.btn_export.setEnabled(False) # Enabled when video loaded, or webcam recording finished
        bottom_layout.addWidget(self.btn_export)
        
        right_panel.addWidget(self.bottom_controls)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

        self.setCentralWidget(main_widget)
        
        self.video_path = None
        self.image_path = None
        self.music_path = None
        
        self._prevent_slider_feedback = False

    def select_webcam(self):
        self._is_webcam = True
        self.video_path = "0"
        self.btn_video.setText("Select Video File")
        self.btn_webcam.setText("Webcam: Active")
        self.update_controls_mode()
        self.start_playback()

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self._is_webcam = False
            self.video_path = path
            self.btn_video.setText(os.path.basename(path))
            self.btn_webcam.setText("Use Webcam")
            self.update_controls_mode()
            self.start_playback()

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.image_path = path
            self.btn_image.setText(os.path.basename(path))
            self.start_playback()

    def select_music(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Music", "", "Audio Files (*.mp3 *.wav *.m4a *.flac *.aac)")
        if path:
            self.music_path = path
            self.btn_music.setText(os.path.basename(path))
            self.start_playback()

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
        self.processor.set_media(self.video_path, self.image_path, self.music_path, is_webcam=self._is_webcam)
        self.on_settings_changed()
        if not self.processor.isRunning():
            self.processor.start()

    def on_settings_changed_slider(self):
        if not self._prevent_slider_feedback:
            self.on_settings_changed()

    def on_settings_changed(self):
        self.video_label.warp_mode = self.cb_warp.isChecked()
        codec_str = self.cb_codec.currentText().split(" ")[0]
        
        is_endfade = self.cb_endfade.isChecked()
        if is_endfade and not self._is_webcam and not self.processor.endfade_mode:
            self.lbl_ef_status.setText("Scanning video...")
            self.endfade_group.show()
        elif not is_endfade:
            self.endfade_group.hide()
            
        params = {
            "warp_mode": self.cb_warp.isChecked(),
            "feather": self.slider_feather.value(),
            "coast_limit": self.slider_coast.value(),
            "show_debug": self.cb_debug.isChecked(),
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
        
    def on_interactive_placement(self, placement):
        # Update sliders without triggering loop
        self._prevent_slider_feedback = True
        self.slider_x.setValue(int(placement["x"]))
        self.slider_y.setValue(int(placement["y"]))
        self.slider_scale.setValue(int(placement["scale"] * 100))
        self.slider_rot.setValue(int(placement["rotation_deg"]))
        self._prevent_slider_feedback = False
        
        self.processor.update_params({"placement": placement})

    def toggle_play_pause(self):
        if self.processor._paused:
            self.processor.set_paused(False)
            self.btn_play_pause.setText("Pause")
        else:
            self.processor.set_paused(True)
            self.btn_play_pause.setText("Play")
            
    def toggle_recording(self):
        if self.processor._recording:
            self.processor.stop_recording()
        else:
            self.processor.start_recording()
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: #ffcccc; color: #cc0000; font-weight: bold; border: 1px solid #cc0000;")

    def on_raw_recording_ready(self):
        self.btn_record.setEnabled(False)
        self.btn_record.setText("Rendering...")
        self.btn_export.setEnabled(False)
        self.processor._rendering_webcam = True
        
    def on_recording_finished(self):
        self.btn_record.setEnabled(True)
        self.btn_record.setText("Start Recording")
        self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold;")
        self.btn_export.setEnabled(True)

    def do_export(self):
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "output.mp4", "Video Files (*.mp4)")
        if not out_path: return
        
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
            self.processor.start_export(out_path)
            
    def on_export_finished(self):
        self.btn_export.setEnabled(True)
        self.btn_export.setText("Export")
        QMessageBox.information(self, "Export Complete", "Video rendering finished successfully.")

    def on_timeline_pressed(self):
        self._user_is_seeking = True

    def on_timeline_released(self):
        self._user_is_seeking = False
        self.processor.request_seek(self.timeline.value())

    def on_timeline_moved(self, value):
        if self._user_is_seeking:
            self.processor.request_seek(value)

    @pyqtSlot(int)
    def update_timeline_range(self, total_frames):
        self.timeline.setRange(0, total_frames)

    @pyqtSlot(int)
    def update_timeline_pos(self, frame_idx):
        if not self._user_is_seeking:
            self.timeline.setValue(frame_idx)

    @pyqtSlot(QImage)
    def update_frame(self, qimg):
        # We don't scale here anymore, it's pre-scaled by the thread
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    @pyqtSlot(str)
    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        self.processor.stop()
        if os.path.exists(TEMP_RECORDING):
            try: os.remove(TEMP_RECORDING)
            except: pass
        event.accept()

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
