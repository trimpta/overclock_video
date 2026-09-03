# Overclock Video

Tracks both thumbs and index fingertips in a video (via MediaPipe hand landmarks),
and reveals a static image through the quadrilateral those 4 fingertips form —
like looking through a hand-shaped window at a picture placed behind the video.
The output is set to a music track.

- **debug** mode overlays the full hand skeleton, the 4 tracked fingertips, and
  the tracked quadrilateral outline (with labels/confidence/prediction state).
- **render** mode outputs the final composite with no overlay.

The image does not warp to the quad — it sits at a fixed position/scale/rotation
you set once via an interactive preview window, and the quad acts purely as a
mask revealing whatever part of it currently falls underneath.

Output resolution always matches the input video's resolution.

## Setup

```bash
pip install -r requirements.txt
```

The MediaPipe hand-landmark model (`models/hand_landmarker.task`) is included.
Audio muxing uses a bundled ffmpeg binary via `imageio-ffmpeg` — no separate
ffmpeg install required.

## Usage

Run with flags:

```bash
python overclock.py --video in.mp4 --image pic.png --music song.mp3 --mode debug
```

Or run with no arguments (or partial arguments) for an interactive wizard that
prompts for whatever's missing:

```bash
python overclock.py
```

The first time you run against a given video+image pair, an interactive window
opens so you can drag/scale/rotate the image into place:

- Drag with the left mouse button to move it
- Mouse wheel or `+`/`-` to scale
- `Q`/`E` to rotate
- `R` to reset, `Enter` to confirm, `Esc` to cancel

That placement is saved to a JSON file next to the video so re-renders (or
switching between debug/render mode) don't require repositioning — pass
`--reposition` to redo it.

## Other flags

- `--output PATH` — output file (default: `<video>_overclocked.mp4`)
- `--placement-config PATH` — explicit placement JSON path
- `--coast-limit N` — frames a lost fingertip may be predicted before the
  image hides for that frame (default: 45)
- `--feather N` — mask edge softening in pixels (default: 9, 0 to disable)
- `--preview` — show a live window while processing (debug mode only)
