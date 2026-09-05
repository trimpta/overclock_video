import json
import os

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
USER_DATA_DIR = os.path.join(PROJECT_ROOT, "userdata")
MEDIA_DIR = os.path.join(USER_DATA_DIR, "media")
DEFAULT_LAST_RUN_PATH = os.path.join(USER_DATA_DIR, ".last_run.json")


def ensure_app_dirs():
    for path in (MEDIA_DIR, OUTPUT_DIR, USER_DATA_DIR):
        os.makedirs(path, exist_ok=True)


def resolve_existing_path(path):
    """Turn a stored path into an existing file, preferring media/ over project root."""
    if path is None or path == "":
        return None
    if str(path) in ("0", "webcam") or (isinstance(path, int)):
        return str(path)
    path = os.path.expanduser(str(path).strip().strip('"'))
    if os.path.isfile(path):
        return os.path.normpath(os.path.abspath(path))
    name = os.path.basename(path)
    for folder in (MEDIA_DIR, USER_DATA_DIR, OUTPUT_DIR, PROJECT_ROOT):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


def resolve_output_dir(path: str) -> str:
    """Exports never land in the project root; relative paths are under PROJECT_ROOT."""
    ensure_app_dirs()
    path = (path or "").strip().strip('"')
    if not path:
        return OUTPUT_DIR
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    path = os.path.normpath(path)
    if os.path.normcase(path) == os.path.normcase(PROJECT_ROOT):
        return OUTPUT_DIR
    return path


def resolve_write_path(path: str) -> str:
    """Absolute path for ffmpeg/temp writes; create the parent directory."""
    path = os.path.normpath(os.path.abspath(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def resolve_output_file(path: str, default_name: str = "output.mp4") -> str:
    """Absolute export file path; relative names are under the project, never the repo root."""
    ensure_app_dirs()
    if not default_name.lower().endswith(".mp4"):
        default_name += ".mp4"
    path = (path or "").strip().strip('"')
    if not path:
        return resolve_write_path(os.path.join(OUTPUT_DIR, default_name))
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    path = os.path.normpath(path)
    if os.path.isdir(path):
        return resolve_write_path(os.path.join(path, default_name))
    parent = resolve_output_dir(os.path.dirname(path) or OUTPUT_DIR)
    base = os.path.basename(path)
    if not base.lower().endswith(".mp4"):
        base += ".mp4"
    return resolve_write_path(os.path.join(parent, base))


def load_placement(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_placement(path: str, placement: dict):
    path = resolve_write_path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(placement, f, indent=2)


def load_last_run(path: str = DEFAULT_LAST_RUN_PATH):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video = data.get("video")
        if video and str(video) not in ("0", "webcam") and not str(video).isdigit():
            resolved = resolve_existing_path(video)
            if not resolved:
                return None
            data["video"] = resolved
        if data.get("image"):
            data["image"] = resolve_existing_path(data["image"])
        if data.get("music"):
            data["music"] = resolve_existing_path(data["music"])
        if data.get("out_file"):
            data["out_file"] = resolve_output_file(data["out_file"])
        elif data.get("out_dir"):
            data["out_file"] = resolve_output_file(os.path.join(resolve_output_dir(data["out_dir"]), "output.mp4"))
        return data
    except Exception:
        return None


def save_last_run(data: dict, path: str = DEFAULT_LAST_RUN_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save session settings ({e})")


def session_compositing_settings(data):
    """Extract placement/warp/endfade/feather/coast/codec from last-run JSON."""
    if not isinstance(data, dict):
        return {}
    out = {}
    placement = data.get("placement")
    if isinstance(placement, dict):
        try:
            out["placement"] = {
                "x": float(placement.get("x", 960)),
                "y": float(placement.get("y", 540)),
                "scale": float(placement.get("scale", 1.0)),
                "rotation_deg": float(placement.get("rotation_deg", 0.0)),
            }
        except (TypeError, ValueError):
            pass
    if "warp_mode" in data:
        out["warp_mode"] = bool(data["warp_mode"])
    if "endfade_mode" in data:
        out["endfade_mode"] = bool(data["endfade_mode"])
    if "glitch_fill_enabled" in data:
        out["glitch_fill_enabled"] = bool(data["glitch_fill_enabled"])
    if "glitch_fill_opacity" in data:
        try:
            out["glitch_fill_opacity"] = int(data["glitch_fill_opacity"])
        except (TypeError, ValueError):
            pass
    if "record_mic" in data:
        out["record_mic"] = bool(data["record_mic"])
    if "mic_device" in data and isinstance(data["mic_device"], str):
        out["mic_device"] = data["mic_device"].strip()
    for key, caster in (
        ("endfade_offset", int),
        ("endfade_duration", int),
        ("endfade_image_scale_offset", int),
        ("endfade_image_scale_duration", int),
        ("feather", int),
        ("coast_limit", int),
        ("jagged_border_baseline", float),
        ("jagged_border_thickness", int),
    ):
        if key in data and data[key] is not None:
            try:
                out[key] = caster(data[key])
            except (TypeError, ValueError):
                pass
    if "jagged_border_enabled" in data:
        out["jagged_border_enabled"] = bool(data["jagged_border_enabled"])

    if "endfade_image_scale_enabled" in data:
        out["endfade_image_scale_enabled"] = bool(data["endfade_image_scale_enabled"])
    codec = data.get("codec")
    if isinstance(codec, str) and codec.strip():
        out["codec"] = codec.strip().split(" ")[0]
    return out
