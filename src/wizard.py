"""Interactive fallback prompts used when required CLI arguments are missing.
Presents numbered option menus rather than raw free-text input wherever
possible.
"""
import glob
import os

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma")


def menu_choice(prompt: str, options: list):
    """options: list of (label, value). Returns the chosen value."""
    print(f"\n{prompt}")
    for i, (label, _) in enumerate(options, start=1):
        print(f"  {i}. {label}")
    while True:
        raw = input(f"Choose 1-{len(options)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print("Invalid choice, try again.")


def _find_files(exts, search_dirs=(".",)):
    found = []
    for d in search_dirs:
        for ext in exts:
            found.extend(glob.glob(os.path.join(d, f"*{ext}")))
            found.extend(glob.glob(os.path.join(d, f"*{ext.upper()}")))
    # de-dupe, keep order
    seen = set()
    unique = []
    for f in found:
        norm = os.path.normpath(f)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def ask_file_path(kind_label: str, exts: tuple, search_dirs=(".",), none_label: str = None):
    """none_label: if set, adds an option returning None (e.g. "No image / use greenscreen")."""
    candidates = _find_files(exts, search_dirs)
    options = [(f"{c}", c) for c in candidates]
    if none_label:
        options.append((none_label, None))
    options.append(("Enter a path manually", "__manual__"))

    choice = menu_choice(f"Select the {kind_label} file:", options)
    if choice != "__manual__":
        return choice

    while True:
        raw = input(f"Path to {kind_label} file: ").strip().strip('"')
        if os.path.isfile(raw):
            return raw
        print(f"File not found: {raw}")


def ask_reuse_placement(config_path: str):
    return menu_choice(
        f"Found an existing image placement at {config_path}.",
        [
            ("Reuse it", "reuse"),
            ("Redo placement (opens the interactive preview)", "redo"),
        ],
    )
