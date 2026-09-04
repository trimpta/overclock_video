import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import wizard
from .audio_mux import FfmpegFrameWriter
from .compositor import (
    build_canvas,
    composite_frame,
    warp_composite_frame,
    load_image_bgra,
    make_greenscreen_bgra,
    full_frame_placement,
)
from .config import (
    OUTPUT_DIR,
    ensure_app_dirs,
    load_last_run,
    load_placement,
    resolve_output_file,
    resolve_write_path,
    save_last_run,
    save_placement,
)
from .debug_draw import draw_debug_overlay
from .hand_tracking import HandTracker
from .placement_ui import run_placement_ui
from .smoothing import QuadTracker

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hand_landmarker.task")


@dataclass
class ResolvedArgs:
    video: str
    image: Optional[str]  # None => use a greenscreen placeholder
    music: Optional[str]  # None => render without audio
    output: str
    mode: str
    placement_config: Optional[str]  # None when using the greenscreen (no placement needed)
    reposition: bool
    coast_limit: int
    feather: int
    preview: bool
    warp: bool
    headless: bool
    codec: str
    preset: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overclock",
        description="Track fingertips and show an image through the quad they form.",
    )
    p.add_argument("--video", help="Input video file path, or '0'/'webcam' for live camera")
    p.add_argument("--image", help="Image file path (optional; omit for greenscreen)")
    p.add_argument("--music", help="Audio file path (optional; omit for silent output)")
    p.add_argument("--output", help="Output video file path (default: userdata/output/<video>_overclocked.mp4)")
    p.add_argument("--mode", choices=["render", "debug", "live"], default="render", help="Output mode (default: render)")
    p.add_argument("--warp", action="store_true", help="Enable warp mode (perspective warp image into hand quad)")
    p.add_argument("--headless", action="store_true", help="Run headlessly without interactive windows or prompts")
    p.add_argument("--placement-config", help="Explicit placement JSON path")
    p.add_argument("--reposition", action="store_true", help="Redo placement even if config exists")
    p.add_argument("--coast-limit", type=int, default=15, help="Frames a lost fingertip may be predicted before hiding the image (default: 15)")
    p.add_argument("--feather", type=int, default=9, help="Mask edge softening in pixels (default: 9, 0 to disable)")
    p.add_argument("--codec", default="libx264", help="Video codec (default: libx264)")
    p.add_argument("--preset", default="medium", help="Encoding preset (default: medium)")
    p.add_argument("--preview", action="store_true", help="Show a live window while processing (debug mode only)")
    return p


def is_valid_video(v: str) -> bool:
    if str(v).isdigit() or str(v).lower() == "webcam":
        return True
    return os.path.isfile(v)


def _default_output(video_path: str) -> str:
    ensure_app_dirs()
    if str(video_path).isdigit() or str(video_path).lower() == "webcam":
        return resolve_write_path(os.path.join(OUTPUT_DIR, "webcam_overclocked.mp4"))
    base = os.path.splitext(os.path.basename(video_path))[0]
    return resolve_write_path(os.path.join(OUTPUT_DIR, f"{base}_overclocked.mp4"))


def _default_placement_config(video_path: str, image_path: str) -> str:
    ensure_app_dirs()
    if str(video_path).isdigit() or str(video_path).lower() == "webcam":
        v = "webcam"
    else:
        v = os.path.splitext(os.path.basename(video_path))[0]
    i = os.path.splitext(os.path.basename(image_path))[0]
    return resolve_write_path(os.path.join(OUTPUT_DIR, f"{v}_{i}_placement.json"))


def resolve_args(argv=None) -> ResolvedArgs:
    parser = build_parser()
    args = parser.parse_args(argv)

    user_supplied_all = bool(args.video and (args.image or args.warp))

    video = args.video if args.video and is_valid_video(args.video) else None
    image = args.image if args.image and os.path.isfile(args.image) else None
    music = args.music if args.music and os.path.isfile(args.music) else None
    mode = args.mode or "render"
    output = args.output

    if not user_supplied_all and not args.headless:
        last_run = load_last_run()
        if last_run and wizard.ask_reuse_last_run(last_run):
            video = video or last_run.get("video")
            image = image or last_run.get("image")
            music = music if music is not None else last_run.get("music")
            mode = mode or last_run.get("mode")
            output = output or last_run.get("output")

    if not mode:
        mode = "render" if args.headless else wizard.ask_mode()

    if not video:
        if mode == "live":
            video = "0"
        elif args.headless:
            print("Error: --video must be specified in headless mode.")
            sys.exit(1)
        else:
            candidates = wizard._find_files(wizard.VIDEO_EXTS)
            if len(candidates) == 1:
                print(f"Auto-detected video: {candidates[0]}")
                video = candidates[0]
            else:
                video = wizard.ask_file_path("video", wizard.VIDEO_EXTS, allow_webcam=True)

    if not image and not (args.image is None and user_supplied_all):
        if args.headless:
            image = None
        else:
            candidates = wizard._find_files(wizard.IMAGE_EXTS)
            if len(candidates) == 1:
                print(f"Auto-detected image: {candidates[0]}")
                image = candidates[0]
            elif len(candidates) == 0:
                print("No image found in directory - using a greenscreen placeholder.")
                image = None
            else:
                image = wizard.ask_file_path("image", wizard.IMAGE_EXTS, allow_none=True)

    if not music and not (args.music is None and user_supplied_all):
        if args.headless:
            music = None
        else:
            candidates = wizard._find_files(wizard.AUDIO_EXTS)
            if len(candidates) == 1:
                print(f"Auto-detected music: {candidates[0]}")
                music = candidates[0]
            elif len(candidates) == 0:
                music = None
            else:
                music = wizard.ask_file_path("music", wizard.AUDIO_EXTS, allow_none=True)

    if mode == "live":
        output = None
    elif not output:
        output = _default_output(video)
    else:
        output = resolve_output_file(output, os.path.basename(_default_output(video)))

    placement_config = None
    if image is not None:
        placement_config = args.placement_config or _default_placement_config(video, image)
        os.makedirs(os.path.dirname(placement_config), exist_ok=True)

    saved_mode = last_run.get("mode", "debug") if (mode == "live" and 'last_run' in locals() and last_run) else ("debug" if mode == "live" else mode)
    save_last_run({
        "video": video,
        "image": image,
        "music": music,
        "mode": saved_mode,
        "output": output if mode != "live" else _default_output(video),
        "warp_mode": args.warp,
        "codec": args.codec,
    })

    return ResolvedArgs(
        video=video,
        image=image,
        music=music,
        output=output,
        mode=mode,
        placement_config=placement_config,
        reposition=args.reposition,
        coast_limit=args.coast_limit,
        feather=args.feather,
        preview=args.preview,
        warp=args.warp,
        headless=args.headless,
        codec=args.codec,
        preset=args.preset,
    )


def run_cli(argv=None):
    args = resolve_args(argv)

    if not os.path.isfile(MODEL_PATH):
        print(f"Error: Hand tracking model not found at:\n  {MODEL_PATH}")
        sys.exit(1)

    is_webcam = str(args.video).isdigit() or str(args.video).lower() == "webcam"
    if is_webcam:
        cam_idx = int(args.video) if str(args.video).isdigit() else 0
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(cam_idx)
        else:
            cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        fps = 30.0
    else:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 120:
            fps = 30.0

    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.video}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = None if is_webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames and total_frames <= 0:
        total_frames = None

    ok, first_frame = cap.read()
    if not ok:
        print("Error: Could not read first frame from video source.")
        cap.release()
        sys.exit(1)

    # Deliberately not seeking back to frame 0 here: CAP_PROP_POS_FRAMES
    # seeks are unreliable on some codecs. first_frame will be yielded first.

    if args.image is None:
        image_bgra = make_greenscreen_bgra(frame_w, frame_h)
        placement = full_frame_placement(frame_w, frame_h)
    else:
        image_bgra = load_image_bgra(args.image)
        if args.warp:
            placement = full_frame_placement(frame_w, frame_h)
        else:
            placement = None if args.reposition else load_placement(args.placement_config)
            if placement is not None and not args.reposition and not args.headless:
                choice = wizard.ask_reuse_placement(args.placement_config)
                if choice == "redo":
                    placement = None

            if placement is None:
                if args.headless:
                    placement = full_frame_placement(frame_w, frame_h)
                else:
                    try:
                        print("\nOpening interactive placement window...")
                        placement = run_placement_ui(first_frame, image_bgra, initial_placement=placement)
                    except Exception as e:
                        print(f"Placement UI unavailable ({e}); using default placement.")
                        placement = full_frame_placement(frame_w, frame_h)
                    if placement is None:
                        print("Placement cancelled.")
                        cap.release()
                        sys.exit(1)
                if args.placement_config:
                    save_placement(args.placement_config, placement)

    if not args.warp:
        canvas_bgr, canvas_alpha = build_canvas(image_bgra, placement, frame_w, frame_h)
    else:
        canvas_bgr, canvas_alpha = None, None

    output_path = args.output
    if not output_path:
        output_path = _default_output(args.video)

    ensure_app_dirs()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_output_path = f"{output_path}.tmp.mp4"

    tracker = HandTracker(MODEL_PATH, num_hands=2)
    quad_tracker = QuadTracker(coast_limit=args.coast_limit)
    writer = FfmpegFrameWriter(
        temp_output_path,
        frame_w,
        frame_h,
        fps,
        music_path=args.music,
        codec=args.codec,
        preset=args.preset,
    )

    print(f"\nRendering ({args.mode} mode, warp={args.warp}) -> {output_path}")
    frame_idx = 0
    start_time = time.time()

    def frames():
        yield first_frame
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f

    try:
        for frame in frames():
            timestamp_ms = int((frame_idx / max(fps, 1.0)) * 1000)
            hand_result = tracker.process(frame, timestamp_ms)
            smoothed = quad_tracker.update(hand_result.roles)

            if args.warp:
                ordered_pts = QuadTracker.ordered_quad_points(smoothed)
                quad_pts = np.round(ordered_pts).astype(np.int32) if ordered_pts is not None else None
                out_frame = warp_composite_frame(frame, quad_pts, image_bgra, feather=args.feather)
            else:
                quad_pts = QuadTracker.quad_points(smoothed)
                out_frame = composite_frame(frame, quad_pts, canvas_bgr, canvas_alpha, feather=args.feather)

            if args.mode == "debug":
                out_frame = draw_debug_overlay(out_frame, hand_result.hands_raw, smoothed, quad_pts)
                if args.preview and not args.headless:
                    cv2.imshow("Overclock Video - debug preview", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nAborted by user.")
                        break

            if not writer.write(out_frame) or writer.failed:
                err = writer.error_message or "Video encoding write failed."
                print(f"\nError writing frame {frame_idx}: {err}")
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                if total_frames:
                    pct = 100.0 * frame_idx / total_frames
                    print(f"\r  frame {frame_idx}/{total_frames} ({pct:.1f}%) - {fps_actual:.1f} fps", end="", flush=True)
                else:
                    print(f"\r  frame {frame_idx} - {fps_actual:.1f} fps", end="", flush=True)

    finally:
        cap.release()
        tracker.close()
        writer.close()
        if args.preview and not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if writer.failed:
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
        print(f"\nExport failed: {writer.error_message}")
        sys.exit(1)

    if os.path.exists(temp_output_path):
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        shutil.move(temp_output_path, output_path)

    elapsed = time.time() - start_time
    fps_avg = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\nDone! Wrote {frame_idx} frames in {elapsed:.1f}s ({fps_avg:.1f} fps) to:\n  {output_path}")


if __name__ == "__main__":
    run_cli()
