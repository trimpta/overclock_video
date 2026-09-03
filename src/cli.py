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
    p.add_argument("--video", help="Path to the input video file (default: auto-detected if the directory has exactly one)")
    p.add_argument("--image", help="Path to the image to reveal (default: greenscreen if none found)")
    p.add_argument("--music", help="Path to the music/audio track for the output (default: no audio if none found)")
    p.add_argument("--output", help="Path to write the rendered .mp4 (default: output/<video>_overclocked.mp4)")
    p.add_argument("--mode", choices=["debug", "render"], default=None, help="debug shows tracking overlay, render doesn't (default: render)")
    p.add_argument("--placement-config", help="Path to a saved image-placement JSON (default: derived in the output dir)")
    p.add_argument("--reposition", action="store_true", help="Force the interactive placement UI even if a saved config exists")
    p.add_argument("--coast-limit", type=int, default=45, help="Frames a lost fingertip may be predicted before hiding the image (default: 45)")
    p.add_argument("--feather", type=int, default=9, help="Gaussian blur kernel size (px) for softening the mask edge (default: 9, 0 to disable)")
    p.add_argument("--preview", action="store_true", help="Show a live preview window while processing (debug mode only)")
    return p


def _default_output(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(OUTPUT_DIR, f"{base}_overclocked.mp4")


def _default_placement_config(video_path: str, image_path: str) -> str:
    v = os.path.splitext(os.path.basename(video_path))[0]
    i = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(OUTPUT_DIR, f"{v}_{i}_placement.json")


def _resolve_video(explicit: Optional[str]) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = wizard._find_files(wizard.VIDEO_EXTS)
    if len(candidates) == 1:
        print(f"Auto-detected video: {candidates[0]}")
        return candidates[0]
    return wizard.ask_file_path("video", wizard.VIDEO_EXTS)


def _resolve_image(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = wizard._find_files(wizard.IMAGE_EXTS)
    if len(candidates) == 1:
        print(f"Auto-detected image: {candidates[0]}")
        return candidates[0]
    if len(candidates) == 0:
        print("No image found in directory - using a greenscreen placeholder.")
        return None
    return wizard.ask_file_path("image", wizard.IMAGE_EXTS, none_label="No image / use greenscreen")


def _resolve_music(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = wizard._find_files(wizard.AUDIO_EXTS)
    if len(candidates) == 1:
        print(f"Auto-detected music: {candidates[0]}")
        return candidates[0]
    if len(candidates) == 0:
        print("No audio found in directory - rendering without music.")
        return None
    return wizard.ask_file_path("music", wizard.AUDIO_EXTS, none_label="No audio / skip music")


def resolve_args(argv=None) -> ResolvedArgs:
    parser = build_parser()
    args = parser.parse_args(argv)

    video = _resolve_video(args.video)
    image = _resolve_image(args.image)
    music = _resolve_music(args.music)
    mode = args.mode or "render"

    output = args.output or _default_output(video)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    placement_config = None
    if image is not None:
        placement_config = args.placement_config or _default_placement_config(video, image)
        os.makedirs(os.path.dirname(placement_config), exist_ok=True)

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
