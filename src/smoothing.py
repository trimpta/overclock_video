"""Per-fingertip smoothing and short-gap prediction using a constant-velocity
Kalman filter, so brief occlusion / low-confidence frames don't cause the
tracked quadrilateral to flicker or vanish.
"""
import cv2
import numpy as np

from .hand_tracking import ROLE_KEYS

# How many consecutive frames a point may be "coasted" (predicted with no
# fresh measurement) before we give up and treat it as lost for that frame.
DEFAULT_COAST_LIMIT = 45  # ~1.5s at 30fps


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


class QuadTracker:
    """Tracks the 4 fingertip roles and produces a smoothed quad each frame."""

    def __init__(self, coast_limit: int = DEFAULT_COAST_LIMIT):
        self._smoothers = {role: PointSmoother(coast_limit) for role in ROLE_KEYS}

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
    def quad_points(smoothed: dict):
        """Returns the 4 polygon points in winding order (TL, TR, BR, BL) for
        fillPoly, or None if any of the 4 fingertips is currently lost.
        """
        if any(smoothed[role]["lost"] for role in ROLE_KEYS):
            return None
        tl = smoothed["left_index"]["pos"]
        tr = smoothed["right_index"]["pos"]
        br = smoothed["right_thumb"]["pos"]
        bl = smoothed["left_thumb"]["pos"]
        return np.array([tl, tr, br, bl], dtype=np.int32)
