import json
import os

DEFAULT_LAST_RUN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data", ".last_run.json")


def load_placement(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_placement(path: str, placement: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(placement, f, indent=2)


def load_last_run(path: str = DEFAULT_LAST_RUN_PATH):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("video") and not os.path.isfile(data["video"]):
            return None
        if data.get("image") and not os.path.isfile(data["image"]):
            return None
        if data.get("music") and not os.path.isfile(data["music"]):
            return None
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
