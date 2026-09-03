"""Per-fingertip smoothing and short-gap prediction using a constant-velocity
Kalman filter, so brief occlusion / low-confidence frames don't cause the
tracked quadrilateral to flicker or vanish.
"""
import cv2
import numpy as np

from .hand_tracking import ROLE_KEYS

# How many consecutive frames a point may be "coasted" (predicted with no
# fresh measurement) before we give up and treat it as lost for that frame.
DEFAULT_COAST_LIMIT = 0  # Hide mask immediately when tracking is lost


class PointSmoother:
    def __init__(self, coast_limit: int = DEFAULT_COAST_LIMIT):
        self.coast_limit = coast_limit
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self._initialized = False
        self.missed_frames = 0

    def update(self, measurement_xy):
        """measurement_xy: (x, y) or None if not detected this frame.

        Returns (x, y, was_predicted, is_lost).
        """
        if measurement_xy is None:
            if not self._initialized:
                return None, None, False, True

            self.missed_frames += 1
            if self.missed_frames > self.coast_limit:
                return None, None, False, True

            pred = self._kf.predict()
            return float(pred[0, 0]), float(pred[1, 0]), True, False

        x, y = measurement_xy
        if not self._initialized:
            self._kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self._initialized = True
            self.missed_frames = 0
            return float(x), float(y), False, False

        self._kf.predict()
        corrected = self._kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.missed_frames = 0
        return float(corrected[0, 0]), float(corrected[1, 0]), False, False


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points to TL, TR, BR, BL using spatial heuristics.

    TL = min(x+y), BR = max(x+y), TR = max(x-y), BL = min(x-y).
    Ties / collisions are resolved so each corner is a distinct input point.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts.shape[0]}")

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]

    tl = int(np.argmin(s))
    br = int(np.argmax(s))
    tr = int(np.argmax(d))
    bl = int(np.argmin(d))

    if len({tl, tr, br, bl}) == 4:
        return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)

    # Collision (ties or near-degenerate quad): lock TL/BR from sum, assign
    # the two remaining points by x-y.
    if tl == br:
        order = np.argsort(s, kind="mergesort")
        tl, br = int(order[0]), int(order[-1])
        if tl == br:
            tl, br = 0, min(2, pts.shape[0] - 1)

    rem = [i for i in range(4) if i != tl and i != br]
    if d[rem[0]] >= d[rem[1]]:
        tr, bl = rem[0], rem[1]
    else:
        tr, bl = rem[1], rem[0]

    return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)


class QuadTracker:
    """Tracks the 4 fingertip roles and produces a smoothed quad each frame."""

    def __init__(self, coast_limit: int = DEFAULT_COAST_LIMIT):
        self._coast_limit = coast_limit
        self._smoothers = {role: PointSmoother(coast_limit) for role in ROLE_KEYS}

    @property
    def coast_limit(self):
        return self._coast_limit

    @coast_limit.setter
    def coast_limit(self, value):
        self._coast_limit = value
        for smoother in self._smoothers.values():
            smoother.coast_limit = value

    def reset(self):
        self._smoothers = {role: PointSmoother(self._coast_limit) for role in ROLE_KEYS}

    def update(self, roles: dict):
        """roles: dict role -> (x, y, score) or None.

        Returns dict role -> {"pos": (x, y), "predicted": bool, "lost": bool}.
        """
        out = {}
        for role in ROLE_KEYS:
            measurement = roles.get(role)
            xy = (measurement[0], measurement[1]) if measurement else None
            x, y, predicted, lost = self._smoothers[role].update(xy)
            out[role] = {
                "pos": None if lost else (x, y),
                "predicted": predicted,
                "lost": lost,
            }
        return out

    @staticmethod
    def _raw_quad_positions(smoothed: dict):
        """Return the 4 fingertip positions, or None if any role is lost."""
        if any(smoothed[role]["lost"] or smoothed[role]["pos"] is None for role in ROLE_KEYS):
            return None
        return [smoothed[role]["pos"] for role in ROLE_KEYS]

    @staticmethod
    def quad_points(smoothed: dict):
        """Returns the 4 polygon points in winding order around their centroid for
        fillPoly (stencil), or None if any of the 4 fingertips is currently lost.

        Angle-sort is intentional for a simple non-self-intersecting polygon.
        For perspective warp use ordered_quad_points / order_points_tl_tr_br_bl.
        """
        pts = QuadTracker._raw_quad_positions(smoothed)
        if pts is None:
            return None
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        # Sort points by angle relative to centroid to ensure a simple non-self-intersecting polygon
        pts_sorted = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        return np.array(pts_sorted, dtype=np.int32)

    @staticmethod
    def ordered_quad_points(smoothed: dict):
        """Returns the 4 points as TL, TR, BR, BL for perspective warp, or None
        if any fingertip is currently lost.
        """
        pts = QuadTracker._raw_quad_positions(smoothed)
        if pts is None:
            return None
        return order_points_tl_tr_br_bl(np.array(pts, dtype=np.float32))
