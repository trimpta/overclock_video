#!/usr/bin/env python3
"""Overclock Video: track both thumbs + index fingers and reveal an image
through the quadrilateral they form, optionally set to a music track.

Launch the GUI with no arguments:
    python overclock.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GUI_HELP = """\
Overclock Video - GUI app

Usage:
  python overclock.py          Launch the GUI
  python overclock.py --help   Show this help

Setup:
  setup.bat                    (Windows) create venv, install deps, launch
  pip install -r requirements.txt && python overclock.py

In the GUI: webcam or video file, optional image/music, warp mode, endfade,
export codecs, image adjust editor, and session restore.

Legacy CLI flags are deprecated; the GUI is the supported path.
"""


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("--help", "-h"):
            print(GUI_HELP, end="")
            sys.exit(0)

        print("CLI mode is deprecated.")
        print("Please run `python overclock.py` with no arguments to use the GUI.")
        print("For usage help: python overclock.py --help")
        sys.exit(1)

    try:
        from src.gui import run_gui
        run_gui()
    except ImportError as e:
        print(f"Failed to load GUI: {e}")
        print("Make sure PyQt6 is installed: pip install PyQt6")
        sys.exit(1)


if __name__ == "__main__":
    main()
