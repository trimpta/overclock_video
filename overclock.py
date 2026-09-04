#!/usr/bin/env python3
"""Overclock Video: track both thumbs + index fingers and reveal an image
through the quadrilateral they form, optionally set to a music track.

Launch the GUI with no arguments:
    python overclock.py

Or run headlessly via CLI:
    python overclock.py --cli --video in.mp4 --image pic.png --output out.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP_HELP = """\
Overclock Video

GUI Usage (Default):
  python overclock.py                 Launch the PyQt6 GUI
  python overclock.py --help          Show this help

CLI / Headless Usage:
  python overclock.py --cli [OPTIONS] Run in CLI mode
  python -m src.cli [OPTIONS]         Run CLI module directly

CLI Options:
  --video PATH        Input video file path or '0'/'webcam'
  --image PATH        Overlay image path (optional; greenscreen if omitted)
  --music PATH        Audio soundtrack path (optional)
  --output PATH       Output video path (default: userdata/output/<name>_overclocked.mp4)
  --warp              Enable warp mode (perspective warp image into hand quad)
  --mode {render,debug,live}
                      Output mode (default: render)
  --headless          Run headlessly without interactive prompts or UI windows
  --coast-limit N     Frames to coast lost fingertips (default: 15)
  --feather N         Edge softening in pixels (default: 9)
  --codec CODEC       Video encoder (libx264, h264_nvenc, h264_qsv, h264_amf)
  --preset PRESET     Encoder preset (default: medium)
  --preview           Show preview window during processing
"""


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("--help", "-h"):
            print(APP_HELP, end="")
            sys.exit(0)
        if arg in ("--cli", "-c"):
            from src.cli import run_cli
            run_cli(sys.argv[2:])
            return
        if arg.startswith("-"):
            from src.cli import run_cli
            run_cli(sys.argv[1:])
            return

    try:
        from src.gui import run_gui
        run_gui()
    except ImportError as e:
        print(f"Failed to load GUI: {e}")
        print("Make sure PyQt6 is installed: pip install PyQt6")
        print("Or run headless CLI pipeline: python overclock.py --cli --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
