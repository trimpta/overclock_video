#!/usr/bin/env python3
"""Overclock Video: track both thumbs + index fingers in a video, and reveal
an image (or a greenscreen placeholder, if none is given) through the
quadrilateral they form, optionally set to a music track.

Run with no arguments to auto-detect inputs in the current directory (or get
prompted for whichever ones are ambiguous), or pass flags directly:
    python overclock.py --video in.mp4 --image pic.png --music song.mp3 --mode debug
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_mux import FfmpegFrameWriter
from src.cli import resolve_args
from src.compositor import (
    build_canvas,
    composite_frame,
    full_frame_placement,
    load_image_bgra,
    make_greenscreen_bgra,
)
from src.config import load_placement, save_placement
from src.debug_draw import draw_debug_overlay
from src.hand_tracking import HandTracker
from src.placement_ui import run_placement_ui
from src.smoothing import QuadTracker
from src import wizard

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


def open_video_capture(video_arg):
    if isinstance(video_arg, int):
        return cv2.VideoCapture(video_arg), True
    v_str = str(video_arg).strip()
    if v_str.isdigit() or v_str.lower() == "webcam":
        idx = int(v_str) if v_str.isdigit() else 0
        return cv2.VideoCapture(idx), True
    return cv2.VideoCapture(video_arg), False


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("--help", "-h"):
        # Keep old CLI behavior if arguments are provided
        args = resolve_args()
        
        if not os.path.isfile(MODEL_PATH):
            print(f"Missing hand tracking model at {MODEL_PATH}")
            sys.exit(1)
            
        print("CLI mode is deprecated. Running without arguments will launch the GUI.")
        # We'll just exit here to enforce GUI usage, as the CLI is being phased out in this update
        print("Please run `python overclock.py` with no arguments to use the new GUI.")
        sys.exit(1)
        
    # Launch GUI
    try:
        from src.gui import run_gui
        run_gui()
    except ImportError as e:
        print(f"Failed to load GUI: {e}")
        print("Make sure PyQt6 is installed: pip install PyQt6")
        sys.exit(1)

if __name__ == "__main__":
    main()
