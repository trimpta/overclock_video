"""Interactive fallback prompts used when required CLI arguments are missing.
Presents numbered option menus rather than raw free-text input wherever
possible.
"""
import glob
import os

from .config import MEDIA_DIR, OUTPUT_DIR, USER_DATA_DIR, ensure_app_dirs


def _default_search_dirs():
    ensure_app_dirs()
    return (MEDIA_DIR, USER_DATA_DIR, OUTPUT_DIR)

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
        cleaned_path = raw.strip('"\'')
        if os.path.isfile(cleaned_path):
            return cleaned_path
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print("Invalid choice, try again.")


def _find_files(exts, search_dirs=None):
    if search_dirs is None:
        search_dirs = _default_search_dirs()
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


def ask_file_path(kind_label: str, exts: tuple, search_dirs=None, allow_webcam: bool = False, allow_none: bool = False, none_label: str = None):
    candidates = _find_files(exts, search_dirs)
    options = []
    if allow_webcam:
        options.append(("Webcam (live camera feed)", "0"))
    if allow_none or none_label:
        options.append((none_label or "No music (silent / skip audio)", "__none__"))
    for c in candidates:
        options.append((f"{c}", c))
    options.append(("Enter a path manually", "__manual__"))

    choice = menu_choice(f"Select the {kind_label} file/source:", options)
    if choice == "__none__":
        return None
    if choice != "__manual__":
        return choice

    while True:
        raw = input(f"Path to {kind_label} file (or type 'none' to skip): ").strip().strip('"')
        if (allow_none or none_label) and raw.lower() in ("none", "skip", ""):
            return None
        if str(raw).isdigit() or str(raw).lower() == "webcam":
            return "0"
        if os.path.isfile(raw):
            return raw
        print(f"File not found: {raw}")


def ask_output_path(default_path: str):
    options = [
        (f"Use default: {default_path}", default_path),
        ("Enter a custom path", "__manual__"),
    ]
    choice = menu_choice("Where should the rendered video be saved?", options)
    if choice != "__manual__":
        return choice
    raw = input("Output path: ").strip().strip('"')
    return raw or default_path


def ask_mode():
    return menu_choice(
        "Which mode?",
        [
            ("debug - show hand/finger tracking overlay", "debug"),
            ("render - final output, no overlay", "render"),
            ("live - real-time live preview with looping music (toggle debug overlay with 'R')", "live"),
        ],
    )
def ask_reuse_placement(config_path: str):
    return menu_choice(
        f"Found an existing image placement at {config_path}.",
        [
            ("Reuse it", "reuse"),
            ("Redo placement (opens the interactive preview)", "redo"),
        ],
    )


def ask_reuse_last_run(last_run_data: dict) -> bool:
    prompt_lines = ["Found previous run settings:"]
    if last_run_data.get("video"):
        v_label = "Webcam (live camera feed)" if str(last_run_data['video']).isdigit() or str(last_run_data['video']).lower() == "webcam" else last_run_data['video']
        prompt_lines.append(f"  Video:  {v_label}")
    if last_run_data.get("image"):
        prompt_lines.append(f"  Image:  {last_run_data['image']}")
    if "music" in last_run_data:
        m_label = last_run_data['music'] if last_run_data['music'] else "None (silent)"
        prompt_lines.append(f"  Music:  {m_label}")
    if last_run_data.get("mode"):
        prompt_lines.append(f"  Mode:   {last_run_data['mode']}")
    if last_run_data.get("output") and last_run_data.get("mode") != "live":
        prompt_lines.append(f"  Output: {last_run_data['output']}")
    prompt_lines.append("\nReuse previous inputs?")

    return menu_choice(
        "\n".join(prompt_lines),
        [
            ("Yes - reuse previous settings", True),
            ("No - choose new inputs", False),
        ],
    )
