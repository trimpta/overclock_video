"""Interactive preview window for setting where the image sits on the
static canvas: drag to move, scroll/+/- to scale, q/e to rotate.
"""
import cv2
import numpy as np

from .compositor import build_canvas

INSTRUCTIONS = [
    "Drag with left mouse button to move the image",
    "Mouse wheel or +/- to scale, Q/E to rotate",
    "R to reset, ENTER to confirm, ESC to cancel",
]


class _State:
    def __init__(self, x, y, scale, rotation_deg):
        self.x = x
        self.y = y
        self.scale = scale
        self.rotation_deg = rotation_deg
        self.dragging = False
        self.drag_offset = (0, 0)


def _default_placement(image_bgra, frame_w, frame_h):
    img_w = image_bgra.shape[1]
    target_w = frame_w * 0.4
    scale = target_w / img_w if img_w > 0 else 1.0
    return _State(x=frame_w / 2, y=frame_h / 2, scale=scale, rotation_deg=0.0)


def run_placement_ui(first_frame_bgr, image_bgra, initial_placement=None, window_name="Set image placement"):
    h, w = first_frame_bgr.shape[:2]

    if initial_placement:
        state = _State(
            x=initial_placement.get("x", w / 2),
            y=initial_placement.get("y", h / 2),
            scale=initial_placement.get("scale", 1.0),
            rotation_deg=initial_placement.get("rotation_deg", 0.0),
        )
    else:
        state = _default_placement(image_bgra, w, h)

    def on_mouse(event, mx, my, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.drag_offset = (mx - state.x, my - state.y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if state.dragging:
                state.x = mx - state.drag_offset[0]
                state.y = my - state.drag_offset[1]
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
            factor = 1.05 if delta > 0 else 1 / 1.05
            state.scale = max(0.02, min(8.0, state.scale * factor))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    result = None
    while True:
        placement = {"x": state.x, "y": state.y, "scale": state.scale, "rotation_deg": state.rotation_deg}
        canvas_bgr, canvas_alpha = build_canvas(image_bgra, placement, w, h)
        alpha = (canvas_alpha.astype(np.float32) / 255.0)[:, :, None]
        preview = (first_frame_bgr.astype(np.float32) * (1 - alpha) + canvas_bgr.astype(np.float32) * alpha).astype(np.uint8)

        ys, xs = np.nonzero(canvas_alpha)
        if len(xs) > 0:
            cv2.rectangle(preview, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 255), 1)

        y0 = 25
        for line in INSTRUCTIONS:
            cv2.putText(preview, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 24

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            result = placement
            break
        elif key == 27:  # Esc
            result = None
            break
        elif key == ord('q'):
            state.rotation_deg -= 5
        elif key == ord('e'):
            state.rotation_deg += 5
        elif key in (ord('+'), ord('=')):
            state.scale = min(8.0, state.scale * 1.05)
        elif key == ord('-'):
            state.scale = max(0.02, state.scale / 1.05)
        elif key == ord('r'):
            state = _default_placement(image_bgra, w, h)
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            result = None
            break

    cv2.destroyWindow(window_name)
    return result
