# Overclock Video

A PyQt6 desktop app that tracks both thumbs and index fingertips (via MediaPipe hand landmarks) and reveals an image through the quadrilateral those four fingertips form — like looking through a hand-shaped window at a picture behind the video. Optionally mixes in a music track.

## Setup

**Windows (recommended):** double-click or run `setup.bat`. It creates a `.venv`, installs dependencies, and launches the GUI.

**Manual:**

```bash
pip install -r requirements.txt
python overclock.py
```

The MediaPipe hand-landmark model (`models/hand_landmarker.task`) is included. Audio muxing and export use a bundled ffmpeg binary via `imageio-ffmpeg` — no separate ffmpeg install required.

## Usage

Run with no arguments to open the GUI:

```bash
python overclock.py
```

In the app you can:

- **Webcam / live** — use a live camera feed, record, then export
- **Video file** — load a clip, preview with timeline seek/pause, then export
- **Optional image** — reveal through the hand quad (otherwise a greenscreen/grid placeholder)
- **Optional music** — mux into the exported video
- **Warp mode** — morph the image to the hand quad (vs. static placement as a masked layer)
- **Endfade** — when hands leave near the end of a file, transition the image to fill the frame
- **Export codecs** — `libx264` (CPU), or hardware encoders (`h264_nvenc`, `h264_qsv`, `h264_amf`)
- **Image Adjust Editor** — drag/scale/rotate placement in a dedicated preview window
- **Session restore** — on launch, optionally restore last video/image/music/output settings

Output resolution matches the input video (or webcam) resolution. Exports go to the chosen output directory (default: `output/`).

## CLI note

The supported entry point is the GUI (`python overclock.py` with no args). Legacy CLI modules remain in the repo for reference, but CLI flags are deprecated — passing them prints a short message pointing you to the GUI.
