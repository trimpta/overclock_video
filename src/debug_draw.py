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


def draw_jagged_border(frame, quad_pts, bass_level: float, baseline_jagged: float, thickness: int = 2):
    """Draw a per-frame re-randomized white zigzag border around quad_pts.

    Each of the 4 edges is subdivided into N segments; midpoints are
    displaced perpendicular to the edge by a random amount scaled by both
    ``baseline_jagged`` and the current ``bass_level``.  Because the RNG is
    never seeded the shape changes every frame, creating a constant "vibrating"
    appearance that spikes visually on bass hits.

    Args:
        frame:           BGR frame to draw on (modified in-place and returned).
        quad_pts:        4×2 int32 array of corner points (any winding order).
        bass_level:      Current bass energy [0, 1] from BassAnalyzer.
        baseline_jagged: Baseline jaggedness [0, 1]; 0 = straight when bass=0.
        thickness:       Line thickness in pixels.
    """
    import numpy as np

    if quad_pts is None or len(quad_pts) < 4:
        return frame

    pts = quad_pts.reshape(-1, 2).astype(np.float32)

    # Total jag amplitude in pixels: scales with edge length
    # baseline_jagged=1 → up to ~5% of the shorter frame dimension at bass=1
    frame_h, frame_w = frame.shape[:2]
    max_amp = min(frame_w, frame_h) * 0.04  # 4% of short side at full bass
    amplitude = max_amp * (baseline_jagged + bass_level * (1.0 - baseline_jagged * 0.5))
    amplitude = max(0.0, amplitude)

    # Number of zigzag segments per edge: 3 at silence, up to 18 at full bass
    n_segs = int(3 + (baseline_jagged + bass_level) * 7.5)
    n_segs = max(2, min(n_segs, 20))

    all_points = []

    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]

        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 1.0:
            all_points.append(p0.astype(np.int32))
            continue

        # Unit perpendicular vector (rotate edge 90°)
        perp = np.array([-edge[1], edge[0]], dtype=np.float32) / length

        # Interpolation parameters for the inner subdivision points
        ts = np.linspace(0.0, 1.0, n_segs + 1)

        seg_pts = []
        for j, t in enumerate(ts):
            pt = p0 + edge * t
            if 0 < j < len(ts) - 1:  # displace only interior points
                displacement = np.random.uniform(-amplitude, amplitude)
                pt = pt + perp * displacement
            seg_pts.append(pt.astype(np.int32))

        all_points.extend(seg_pts[:-1])  # exclude last to avoid duplicate at corner

    if len(all_points) < 2:
        return frame

    poly = np.array(all_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    return frame
