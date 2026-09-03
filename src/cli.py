import argparse
import os
from dataclasses import dataclass
from typing import Optional

from . import wizard

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


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


def build_parser():
    p = argparse.ArgumentParser(
        description="Track hand/finger positions in a video and reveal an image (or a "
        "greenscreen placeholder) through the quadrilateral formed by both thumbs and "
        "index fingers."
    )
    p.add_argument("--video", help="Path to the input video file or webcam index (0)")
    p.add_argument("--image", help="Path to the image to reveal (default: greenscreen if none found)")
    p.add_argument("--music", help="Path to the music/audio track for the output (default: no audio if none found)")
    p.add_argument("--output", help="Path to write the rendered .mp4 (default: output/<video>_overclocked.mp4)")
    p.add_argument("--mode", choices=["debug", "render", "live"], default=None, help="debug shows tracking overlay, render outputs without overlay, live shows live preview")
    p.add_argument("--placement-config", help="Path to a saved image-placement JSON")
    p.add_argument("--reposition", action="store_true", help="Force the interactive placement UI even if a saved config exists")
    p.add_argument("--coast-limit", type=int, default=45, help="Frames a lost fingertip may be predicted before hiding the image (default: 45)")
    p.add_argument("--feather", type=int, default=9, help="Gaussian blur kernel size (px) for softening the mask edge (default: 9, 0 to disable)")
    p.add_argument("--preview", action="store_true", help="Show a live preview window while processing (debug mode only)")
    return p


def is_valid_video(v: str) -> bool:
    if not v:
        return False
    if str(v).isdigit() or str(v).lower() == "webcam":
        return True
    return os.path.isfile(v)


def _default_output(video_path: str) -> str:
    if str(video_path).isdigit() or str(video_path).lower() == "webcam":
        return os.path.join(OUTPUT_DIR, "webcam_overclocked.mp4")
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(OUTPUT_DIR, f"{base}_overclocked.mp4")


def _default_placement_config(video_path: str, image_path: str) -> str:
    if str(video_path).isdigit() or str(video_path).lower() == "webcam":
        v = "webcam"
    else:
        v = os.path.splitext(os.path.basename(video_path))[0]
    i = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(OUTPUT_DIR, f"{v}_{i}_placement.json")


from .config import load_last_run, save_last_run


def resolve_args(argv=None) -> ResolvedArgs:
    parser = build_parser()
    args = parser.parse_args(argv)

    user_supplied_all = bool(args.video and args.image)

    video = args.video if args.video and is_valid_video(args.video) else None
    image = args.image if args.image and os.path.isfile(args.image) else None
    music = args.music if args.music and os.path.isfile(args.music) else None
    mode = args.mode
    output = args.output

    if not user_supplied_all:
        last_run = load_last_run()
        if last_run and wizard.ask_reuse_last_run(last_run):
            video = video or last_run.get("video")
            image = image or last_run.get("image")
            music = music if music is not None else last_run.get("music")
            mode = mode or last_run.get("mode")
            output = output or last_run.get("output")

    if not mode:
        mode = wizard.ask_mode()

    if not video:
        if mode == "live":
            video = "0"
        else:
            candidates = wizard._find_files(wizard.VIDEO_EXTS)
            if len(candidates) == 1:
                print(f"Auto-detected video: {candidates[0]}")
                video = candidates[0]
            else:
                video = wizard.ask_file_path("video", wizard.VIDEO_EXTS, allow_webcam=True)

    if not image and not (args.image is None and user_supplied_all):
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

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)

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
    )
