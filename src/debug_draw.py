"""Debug-mode overlay: full hand skeletons, the 4 tracked fingertips, the
tracked quad outline, and labels/confidence/prediction indicators.
"""
import cv2

from .hand_tracking import HAND_CONNECTIONS

ROLE_COLORS = {
    "left_thumb": (255, 80, 80),
    "left_index": (80, 160, 255),
    "right_thumb": (80, 255, 120),
    "right_index": (0, 210, 255),
}
ROLE_LABELS = {
    "left_thumb": "L-Thumb",
    "left_index": "L-Index",
    "right_thumb": "R-Thumb",
    "right_index": "R-Index",
}


def _put_text(frame, text, org, color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_debug_overlay(frame, hands_raw, smoothed, quad_pts):
    for hand in hands_raw:
        pts = [(int(x), int(y)) for x, y in hand.points_px]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 2, (200, 200, 200), -1, cv2.LINE_AA)
        wrist = pts[0]
        _put_text(frame, f"{hand.label} {hand.score:.2f}", (wrist[0] - 20, wrist[1] + 20), (255, 255, 0))

    if quad_pts is not None:
        cv2.polylines(frame, [quad_pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for role, data in smoothed.items():
        if data["lost"] or data["pos"] is None:
            continue
        x, y = int(data["pos"][0]), int(data["pos"][1])
        color = ROLE_COLORS[role]
        cv2.circle(frame, (x, y), 8, color, 2 if data["predicted"] else -1, cv2.LINE_AA)
        tag = ROLE_LABELS[role] + (" (predicted)" if data["predicted"] else "")
        _put_text(frame, tag, (x + 12, y - 10), color)

    return frame
