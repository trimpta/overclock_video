"""Hand / fingertip tracking using MediaPipe's HandLandmarker task API."""
from dataclasses import dataclass, field

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8

# Standard 21-point MediaPipe hand skeleton connections, used for debug drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base
]

ROLE_KEYS = ("left_thumb", "left_index", "right_thumb", "right_index")


@dataclass
class RawHand:
    label: str  # "Left" or "Right" as reported by MediaPipe
    score: float
    points_px: list  # 21 (x, y) pixel tuples


@dataclass
class HandFrameResult:
    # role -> (x_px, y_px, score) or None if that fingertip wasn't seen this frame
    roles: dict = field(default_factory=dict)
    hands_raw: list = field(default_factory=list)  # list[RawHand], for debug overlay


class HandTracker:
    def __init__(
        self,
        model_path: str,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def reset(self):
        self._last_timestamp_ms = -1

    def process(self, frame_bgr, timestamp_ms: int) -> HandFrameResult:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # MediaPipe VIDEO mode requires strictly monotonically increasing timestamps
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"MediaPipe tracking error: {e}")
            return HandFrameResult()

        out = HandFrameResult()
        seen_labels = set()

        # Combine landmark and handedness results and sort by confidence score descending
        detections = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            top = handedness[0]
            detections.append((top.score, top.category_name, hand_landmarks))
        detections.sort(key=lambda d: d[0], reverse=True)

        for score, label, hand_landmarks in detections:
            points_px = [(lm.x * w, lm.y * h) for lm in hand_landmarks]
            out.hands_raw.append(RawHand(label=label, score=score, points_px=points_px))

            # If MediaPipe reports two hands with the same label, keep the higher-confidence one (first in sorted)
            if label in ("Left", "Right"):
                if label in seen_labels:
                    continue
                seen_labels.add(label)

                thumb_x, thumb_y = points_px[THUMB_TIP_IDX]
                index_x, index_y = points_px[INDEX_TIP_IDX]
                prefix = label.lower()
                out.roles[f"{prefix}_thumb"] = (thumb_x, thumb_y, score)
                out.roles[f"{prefix}_index"] = (index_x, index_y, score)

        for key in ROLE_KEYS:
            out.roles.setdefault(key, None)

        return out

    def close(self):
        self._landmarker.close()
