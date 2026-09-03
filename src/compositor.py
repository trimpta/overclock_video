"""Static-stencil compositing: the image sits at a fixed position/scale/
rotation on a canvas the size of the video. Each frame, the quadrilateral
formed by the tracked fingertips acts as a mask/window revealing whatever
part of that static image currently falls under it. The image itself never
warps or moves on its own.
"""
import cv2
import numpy as np


def load_image_bgra(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def build_canvas(image_bgra, placement: dict, frame_w: int, frame_h: int):
    """Bakes the static image (scaled + rotated once, per placement) onto a
    frame-sized canvas. Returns (canvas_bgr uint8 HxWx3, canvas_alpha uint8 HxW).
    Areas the image doesn't cover have alpha=0, so the original video shows
    through even if the mask covers that area.
    """
    scale = max(placement.get("scale", 1.0), 0.01)
    rotation_deg = placement.get("rotation_deg", 0.0)
    cx = placement.get("x", frame_w / 2)
    cy = placement.get("y", frame_h / 2)

    scaled = cv2.resize(image_bgra, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = scaled.shape[:2]

    if abs(rotation_deg) > 1e-3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += new_w / 2 - w / 2
        M[1, 2] += new_h / 2 - h / 2
        rotated = cv2.warpAffine(
            scaled, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
    else:
        rotated = scaled

    rh, rw = rotated.shape[:2]
    tl_x = int(round(cx - rw / 2))
    tl_y = int(round(cy - rh / 2))

    canvas_bgr = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((frame_h, frame_w), dtype=np.uint8)

    dest_x0, dest_y0 = max(tl_x, 0), max(tl_y, 0)
    dest_x1, dest_y1 = min(tl_x + rw, frame_w), min(tl_y + rh, frame_h)
    if dest_x1 > dest_x0 and dest_y1 > dest_y0:
        src_x0, src_y0 = dest_x0 - tl_x, dest_y0 - tl_y
        src_x1, src_y1 = src_x0 + (dest_x1 - dest_x0), src_y0 + (dest_y1 - dest_y0)
        canvas_bgr[dest_y0:dest_y1, dest_x0:dest_x1] = rotated[src_y0:src_y1, src_x0:src_x1, :3]
        canvas_alpha[dest_y0:dest_y1, dest_x0:dest_x1] = rotated[src_y0:src_y1, src_x0:src_x1, 3]

    return canvas_bgr, canvas_alpha


def composite_frame(frame_bgr, quad_pts, canvas_bgr, canvas_alpha, feather: int = 9):
    """quad_pts: 4x2 int32 array in polygon winding order, or None if the
    quad isn't currently valid (image stays fully hidden this frame).
    """
    if quad_pts is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad_pts, 255)

    if feather > 0:
        k = feather | 1  # ensure odd kernel size
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = (mask.astype(np.float32) / 255.0) * (canvas_alpha.astype(np.float32) / 255.0)
    alpha = alpha[:, :, None]

    out = frame_bgr.astype(np.float32) * (1 - alpha) + canvas_bgr.astype(np.float32) * alpha
    return out.astype(np.uint8)
