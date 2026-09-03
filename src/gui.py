import os
import sys
import time
import traceback
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QSlider, QCheckBox, 
    QGroupBox, QFormLayout, QComboBox, QMessageBox
)

from src.audio_mux import FfmpegFrameWriter
from src.compositor import build_canvas, composite_frame, warp_composite_frame, load_image_bgra, make_greenscreen_bgra, full_frame_placement
from src.hand_tracking import HandTracker
from src.smoothing import QuadTracker

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hand_landmarker.task")


class VideoProcessorThread(QThread):
    frame_ready = pyqtSignal(QImage)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.image_path = None
        self.music_path = None
        self.output_path = None
        
        self.codec = "libx264"
        self.preset = "medium"
        
        self.warp_mode = False
        self.placement = {"x": 960, "y": 540, "scale": 1.0, "rotation_deg": 0.0}
        self.feather = 9
        self.coast_limit = 0
        self.show_debug = False

        self._running = False
        self._recording = False
        
        self.cap = None
        self.tracker = None
        self.quad_tracker = None
        self.writer = None
        self.image_bgra = None

    def start_recording(self, output_path):
        self.output_path = output_path
        self._recording = True

    def stop_recording(self):
        self._recording = False

    def update_params(self, params):
        if "warp_mode" in params: self.warp_mode = params["warp_mode"]
        if "feather" in params: self.feather = params["feather"]
        if "coast_limit" in params: self.coast_limit = params["coast_limit"]
        if "show_debug" in params: self.show_debug = params["show_debug"]
        if "placement" in params:
            self.placement.update(params["placement"])
        if "codec" in params: self.codec = params["codec"]

    def set_media(self, video_path, image_path, music_path):
        self.video_path = video_path
        self.image_path = image_path
        self.music_path = music_path
        # When media changes, restart the capture
        if self._running:
            self.stop()
            self.start()

    def run(self):
        if not self.video_path:
            return
            
        self._running = True
        
        # Open Video
        if str(self.video_path).isdigit() or str(self.video_path).lower() == "webcam":
            self.cap = cv2.VideoCapture(int(self.video_path) if str(self.video_path).isdigit() else 0)
            is_webcam = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            is_webcam = False

        if not self.cap.isOpened():
            self.error_occurred.emit(f"Could not open video source: {self.video_path}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 120:
            fps = 30.0
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load Image
        if self.image_path and os.path.isfile(self.image_path):
            self.image_bgra = load_image_bgra(self.image_path)
        else:
            self.image_bgra = make_greenscreen_bgra(frame_w, frame_h)

        self.tracker = HandTracker(MODEL_PATH, num_hands=2)
        self.quad_tracker = QuadTracker(coast_limit=self.coast_limit)
        
        start_time = time.time()
        frame_idx = 0
        
        try:
            while self._running:
                ok, frame = self.cap.read()
                if not ok:
                    if is_webcam:
                        time.sleep(0.01)
                        continue
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self.cap.read()
                        if not ok: break
                
                if is_webcam:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.time() - start_time) * 1000)
                
                # Tracking
                self.quad_tracker.coast_limit = self.coast_limit
                hand_result = self.tracker.process(frame, timestamp_ms)
                smoothed = self.quad_tracker.update(hand_result.roles)
                quad_pts = QuadTracker.quad_points(smoothed)

                # Compositing
                if self.warp_mode:
                    out_frame = warp_composite_frame(frame, quad_pts, self.image_bgra, feather=self.feather)
                else:
                    canvas_bgr, canvas_alpha = build_canvas(self.image_bgra, self.placement, frame_w, frame_h)
                    out_frame = composite_frame(frame, quad_pts, canvas_bgr, canvas_alpha, feather=self.feather)

                if self.show_debug:
                    from src.debug_draw import draw_debug_overlay
                    out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, quad_pts)

                # Recording
                if self._recording:
                    if self.writer is None:
                        try:
                            self.writer = FfmpegFrameWriter(self.output_path, frame_w, frame_h, fps, self.music_path, codec=self.codec, preset=self.preset)
                        except Exception as e:
                            self.error_occurred.emit(f"Failed to start recording: {e}")
                            self._recording = False
                    
                    if self.writer:
                        try:
                            self.writer.write(out_frame)
                        except Exception as e:
                            self.error_occurred.emit(f"Recording error: {e}")
                            self.stop_recording()
                else:
                    if self.writer is not None:
                        self.writer.close()
                        self.writer = None

                # Convert to QImage for GUI
                rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg)

                if not is_webcam:
                    # Sync to fps if reading from file
                    time.sleep(1.0 / fps)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            if self.writer:
                self.writer.close()
                self.writer = None
            if self.cap:
                self.cap.release()
            if self.tracker:
                self.tracker.close()

    def stop(self):
        self._running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overclock Video")
        self.setMinimumSize(1000, 700)
        self.apply_clean_white_theme()

        self.processor = VideoProcessorThread()
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.error_occurred.connect(self.show_error)
        
        self.init_ui()

    def apply_clean_white_theme(self):
        # Apply a simple, clean white theme suitable for a11y
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 102, 204))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow { background-color: white; }
            QLabel { color: black; font-size: 14px; }
            QPushButton { 
                background-color: #f0f0f0; border: 1px solid #ccc; 
                border-radius: 4px; padding: 6px; color: black; font-size: 14px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #d0d0d0; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 6px; margin-top: 10px; padding-top: 10px; color: black; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QSlider::groove:horizontal { border: 1px solid #999; height: 8px; background: #eee; margin: 2px 0; }
            QSlider::handle:horizontal { background: #333333; border: 1px solid #000; width: 18px; margin: -2px 0; border-radius: 3px; }
        """)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left Panel (Settings)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setSpacing(15)

        # Media Selection
        media_group = QGroupBox("Media Inputs")
        media_layout = QFormLayout(media_group)
        
        self.btn_video = QPushButton("Select Video / Webcam")
        self.btn_video.clicked.connect(self.select_video)
        self.btn_video.setAccessibleName("Select Video or Webcam")
        media_layout.addRow("Video:", self.btn_video)

        self.btn_image = QPushButton("Select Image")
        self.btn_image.clicked.connect(self.select_image)
        self.btn_image.setAccessibleName("Select Image to Reveal")
        media_layout.addRow("Image:", self.btn_image)

        self.btn_music = QPushButton("Select Music (Optional)")
        self.btn_music.clicked.connect(self.select_music)
        self.btn_music.setAccessibleName("Select Music File")
        media_layout.addRow("Music:", self.btn_music)

        left_panel.addWidget(media_group)

        # Compositing Settings
        comp_group = QGroupBox("Compositing")
        comp_layout = QFormLayout(comp_group)

        self.cb_warp = QCheckBox("Enable Warp Mode")
        self.cb_warp.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_warp)

        self.cb_debug = QCheckBox("Show Tracking Debug")
        self.cb_debug.stateChanged.connect(self.on_settings_changed)
        comp_layout.addRow(self.cb_debug)

        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setRange(0, 31)
        self.slider_feather.setValue(9)
        self.slider_feather.setSingleStep(2)
        self.slider_feather.valueChanged.connect(self.on_settings_changed)
        self.slider_feather.setAccessibleName("Feathering slider")
        comp_layout.addRow("Feather:", self.slider_feather)

        self.slider_coast = QSlider(Qt.Orientation.Horizontal)
        self.slider_coast.setRange(0, 60)
        self.slider_coast.setValue(0)
        self.slider_coast.valueChanged.connect(self.on_settings_changed)
        self.slider_coast.setAccessibleName("Coast limit slider")
        comp_layout.addRow("Coast Limit:", self.slider_coast)

        left_panel.addWidget(comp_group)

        # Placement Settings
        place_group = QGroupBox("Static Image Placement")
        place_layout = QFormLayout(place_group)

        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(1, 300)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.on_settings_changed)
        place_layout.addRow("Scale %:", self.slider_scale)

        self.slider_rot = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot.setRange(-180, 180)
        self.slider_rot.setValue(0)
        self.slider_rot.valueChanged.connect(self.on_settings_changed)
        place_layout.addRow("Rotation:", self.slider_rot)
        
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setRange(0, 1920)
        self.slider_x.setValue(960)
        self.slider_x.valueChanged.connect(self.on_settings_changed)
        place_layout.addRow("X Offset:", self.slider_x)

        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setRange(0, 1080)
        self.slider_y.setValue(540)
        self.slider_y.valueChanged.connect(self.on_settings_changed)
        place_layout.addRow("Y Offset:", self.slider_y)

        left_panel.addWidget(place_group)

        # Hardware Encoder
        enc_group = QGroupBox("Encoding")
        enc_layout = QFormLayout(enc_group)
        self.cb_codec = QComboBox()
        self.cb_codec.addItems(["libx264 (CPU)", "h264_nvenc (NVIDIA)", "h264_qsv (Intel)", "h264_amf (AMD)"])
        self.cb_codec.currentIndexChanged.connect(self.on_settings_changed)
        enc_layout.addRow("Codec:", self.cb_codec)
        left_panel.addWidget(enc_group)

        # Recording
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setMinimumHeight(50)
        self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold; font-size: 16px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        left_panel.addWidget(self.btn_record)
        
        left_panel.addStretch()

        # Right Panel (Video Preview)
        self.video_label = QLabel("Select a video source to begin.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #fafafa; border: 1px solid #ddd;")

        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.video_label, 3)

        self.setCentralWidget(main_widget)
        
        self.video_path = None
        self.image_path = None
        self.music_path = None

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self.video_path = path
            self.btn_video.setText(os.path.basename(path))
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

    def start_playback(self):
        if not self.video_path:
            return
        self.processor.set_media(self.video_path, self.image_path, self.music_path)
        self.on_settings_changed()
        if not self.processor.isRunning():
            self.processor.start()

    def on_settings_changed(self):
        codec_str = self.cb_codec.currentText().split(" ")[0]
        params = {
            "warp_mode": self.cb_warp.isChecked(),
            "feather": self.slider_feather.value(),
            "coast_limit": self.slider_coast.value(),
            "show_debug": self.cb_debug.isChecked(),
            "codec": codec_str,
            "placement": {
                "x": self.slider_x.value(),
                "y": self.slider_y.value(),
                "scale": self.slider_scale.value() / 100.0,
                "rotation_deg": self.slider_rot.value(),
            }
        }
        self.processor.update_params(params)

    def toggle_recording(self):
        if self.processor._recording:
            self.processor.stop_recording()
            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("background-color: #f0f0f0; color: black; font-weight: bold; font-size: 16px;")
        else:
            out_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "output.mp4", "Video Files (*.mp4)")
            if out_path:
                self.processor.start_recording(out_path)
                self.btn_record.setText("Stop Recording")
                self.btn_record.setStyleSheet("background-color: #ffcccc; color: #cc0000; font-weight: bold; font-size: 16px; border: 1px solid #cc0000;")

    @pyqtSlot(QImage)
    def update_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        # Scale pixmap to fit label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    @pyqtSlot(str)
    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        self.processor.stop()
        event.accept()

def run_gui():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
