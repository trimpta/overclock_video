import json
import os


def load_placement(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_placement(path: str, placement: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(placement, f, indent=2)
