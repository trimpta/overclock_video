import argparse
import os
from dataclasses import dataclass
from typing import Optional

from . import wizard


@dataclass
class ResolvedArgs:
    video: str
    image: str
    music: str
    output: str
    mode: str
    placement_config: str
    reposition: bool
    coast_limit: int
    feather: int
    preview: bool


def build_parser():
    p = argparse.ArgumentParser(
        description="Track hand/finger positions in a video and reveal a static "
        "image through the quadrilateral formed by both thumbs and index fingers."
    )
    p.add_argument("--video", help="Path to the input video file")
    p.add_argument("--image", help="Path to the image to reveal")
    p.add_argument("--music", help="Path to the music/audio track for the output")
    p.add_argument("--output", help="Path to write the rendered .mp4 (default: <video>_overclocked.mp4)")
    p.add_argument("--mode", choices=["debug", "render"], help="debug shows tracking overlay, render doesn't")
    p.add_argument("--placement-config", help="Path to a saved image-placement JSON (default: derived next to output)")
    p.add_argument("--reposition", action="store_true", help="Force the interactive placement UI even if a saved config exists")
    p.add_argument("--coast-limit", type=int, default=45, help="Frames a lost fingertip may be predicted before hiding the image (default: 45)")
    p.add_argument("--feather", type=int, default=9, help="Gaussian blur kernel size (px) for softening the mask edge (default: 9, 0 to disable)")
    p.add_argument("--preview", action="store_true", help="Show a live preview window while processing (debug mode only)")
    return p


def _default_output(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    return f"{base}_overclocked.mp4"


def _default_placement_config(video_path: str, image_path: str) -> str:
    v = os.path.splitext(os.path.basename(video_path))[0]
    i = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.path.dirname(video_path) or ".", f"{v}_{i}_placement.json")


def resolve_args(argv=None) -> ResolvedArgs:
    parser = build_parser()
    args = parser.parse_args(argv)

    video = args.video if args.video and os.path.isfile(args.video) else None
    if not video:
        video = wizard.ask_file_path("video", wizard.VIDEO_EXTS)

    image = args.image if args.image and os.path.isfile(args.image) else None
    if not image:
        image = wizard.ask_file_path("image", wizard.IMAGE_EXTS)

    music = args.music if args.music and os.path.isfile(args.music) else None
    if not music:
        music = wizard.ask_file_path("music", wizard.AUDIO_EXTS)

    mode = args.mode or wizard.ask_mode()

    output = args.output
    if not output:
        output = wizard.ask_output_path(_default_output(video))

    placement_config = args.placement_config or _default_placement_config(video, image)

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
