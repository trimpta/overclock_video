#!/usr/bin/env python3
"""Overclock Video: track both thumbs + index fingers in a video, and reveal
a static image through the quadrilateral they form, set to a music track.

Run with no arguments for an interactive wizard, or pass flags directly:
    python overclock.py --video in.mp4 --image pic.png --music song.mp3 --mode debug
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_mux import FfmpegFrameWriter
from src.cli import resolve_args
from src.compositor import build_canvas, composite_frame, load_image_bgra
from src.config import load_placement, save_placement
from src.debug_draw import draw_debug_overlay
from src.hand_tracking import HandTracker
from src.placement_ui import run_placement_ui
from src.smoothing import QuadTracker
from src import wizard

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


def main():
    args = resolve_args()

    if not os.path.isfile(MODEL_PATH):
        print(f"Missing hand tracking model at {MODEL_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    image_bgra = load_image_bgra(args.image)

    ok, first_frame = cap.read()
    if not ok:
        print("Could not read the first frame of the video.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    placement = None if args.reposition else load_placement(args.placement_config)
    if placement is not None and not args.reposition:
        choice = wizard.ask_reuse_placement(args.placement_config)
        if choice == "redo":
            placement = None

    if placement is None:
        print("\nOpening interactive placement window...")
        placement = run_placement_ui(first_frame, image_bgra, initial_placement=placement)
        if placement is None:
            print("Placement cancelled.")
            sys.exit(1)
        save_placement(args.placement_config, placement)
        print(f"Saved placement to {args.placement_config}")

    canvas_bgr, canvas_alpha = build_canvas(image_bgra, placement, frame_w, frame_h)

    tracker = HandTracker(MODEL_PATH, num_hands=2)
    quad_tracker = QuadTracker(coast_limit=args.coast_limit)
    writer = FfmpegFrameWriter(args.output, frame_w, frame_h, fps, args.music)

    print(f"\nRendering ({args.mode} mode) -> {args.output}")
    frame_idx = 0
    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            hand_result = tracker.process(frame, timestamp_ms)
            smoothed = quad_tracker.update(hand_result.roles)
            quad_pts = QuadTracker.quad_points(smoothed)

            out_frame = composite_frame(frame, quad_pts, canvas_bgr, canvas_alpha, feather=args.feather)

            if args.mode == "debug":
                out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, quad_pts)
                if args.preview:
                    cv2.imshow("Overclock Video - debug preview", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Aborted by user.")
                        break

            writer.write(out_frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                if total_frames:
                    pct = 100 * frame_idx / total_frames
                    print(f"\r  frame {frame_idx}/{total_frames} ({pct:.1f}%) - {fps_actual:.1f} fps", end="", flush=True)
                else:
                    print(f"\r  frame {frame_idx} - {fps_actual:.1f} fps", end="", flush=True)
    finally:
        cap.release()
        tracker.close()
        writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"\nDone. Wrote {frame_idx} frames to {args.output}")


if __name__ == "__main__":
    main()
