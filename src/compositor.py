"""Static-stencil compositing: the image sits at a fixed position/scale/
rotation on a canvas the size of the video. Each frame, the quadrilateral
formed by the tracked fingertips acts as a mask/window revealing whatever
part of that static image currently falls under it. The image itself never
warps or moves on its own.
"""
import cv2
import numpy as np

from .smoothing import order_points_tl_tr_br_bl


def load_image_bgra(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


# Standard chroma-key green (BGR order).
GREENSCREEN_BGR = (0, 177, 64)


def make_greenscreen_bgra(frame_w: int, frame_h: int):
    """A fully opaque, frame-sized solid-green placeholder used when no
    image is supplied. Since it's a uniform fill, it covers the whole quad
    regardless of placement, so it needs no interactive positioning.
    """
    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    canvas[:, :, :3] = GREENSCREEN_BGR
    canvas[:, :, 3] = 255
    return canvas


def make_grid_bgra(frame_w: int, frame_h: int, cell_size: int = 100):
    """A wireframe grid used as a fallback for Warp Mode."""
    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    canvas[:, :, :3] = (240, 240, 240)  # Light gray background
    canvas[:, :, 3] = 255

    line_color = (150, 150, 150) # Darker gray lines
    for x in range(0, frame_w, cell_size):
        cv2.line(canvas, (x, 0), (x, frame_h), line_color, 2)
    for y in range(0, frame_h, cell_size):
        cv2.line(canvas, (0, y), (frame_w, y), line_color, 2)
        
    return canvas


def full_frame_placement(frame_w: int, frame_h: int):
    return {"x": frame_w / 2, "y": frame_h / 2, "scale": 1.0, "rotation_deg": 0.0}


def build_canvas(image_bgra, placement: dict, frame_w: int, frame_h: int):
    """Bakes the static image (scaled + rotated once, per placement) onto a
    frame-sized canvas. Returns (canvas_bgr uint8 HxWx3, canvas_alpha uint8 HxW).
    Areas the image doesn't cover have alpha=0, so the original video shows
    through even if the mask covers that area.
    """
    scale = max(placement.get("scale", 1.0), 0.01)
    img_h, img_w = image_bgra.shape[:2]
    
    if int(img_w * scale) == 0 or int(img_h * scale) == 0:
        return np.zeros((frame_h, frame_w, 3), dtype=np.uint8), np.zeros((frame_h, frame_w), dtype=np.uint8)
        
    rotation_deg = placement.get("rotation_deg", 0.0)
    cx = placement.get("x", frame_w / 2)
    cy = placement.get("y", frame_h / 2)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(image_bgra, (0, 0), fx=scale, fy=scale, interpolation=interp)
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
    if quad_pts is None or len(quad_pts) < 3:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad_pts], 255)

    if canvas_bgr.shape[:2] != (h, w) or canvas_alpha.shape[:2] != (h, w):
        # Cache/frame size mismatch — skip composite rather than crash OpenCV
        return frame_bgr

    if feather > 0:
        k = feather | 1  # ensure odd kernel size
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha_8u = cv2.multiply(mask, canvas_alpha, scale=1.0/255.0)
    alpha_3ch = cv2.merge([alpha_8u, alpha_8u, alpha_8u])

    frame_16 = frame_bgr.astype(np.uint16)
    canvas_16 = canvas_bgr.astype(np.uint16)

    out_16 = frame_16 * (255 - alpha_3ch) + canvas_16 * alpha_3ch
    out = (out_16 // 255).astype(np.uint8)
    return out


def warp_composite_frame(frame_bgr, quad_pts, image_bgra, feather: int = 9):
    """Warps the raw image_bgra to fit exactly into quad_pts.

    Destination corners are forced to TL, TR, BR, BL via
    ``order_points_tl_tr_br_bl`` so callers may pass unsorted quads
    (e.g. angle-sorted ``QuadTracker.quad_points``).
    """
    if quad_pts is None:
        return frame_bgr

    try:
        dst_pts = order_points_tl_tr_br_bl(quad_pts)
        if not np.all(np.isfinite(dst_pts)):
            return frame_bgr
        ordered_i32 = np.round(dst_pts).astype(np.int32)
        area = abs(cv2.contourArea(ordered_i32))
        if area < 1.0:
            return frame_bgr
    except Exception:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    img_h, img_w = image_bgra.shape[:2]
    if img_w < 2 or img_h < 2:
        return frame_bgr

    src_pts = np.array([
        [0, 0],
        [img_w, 0],
        [img_w, img_h],
        [0, img_h]
    ], dtype=np.float32)

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            image_bgra, M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
    except (cv2.error, ValueError, ZeroDivisionError):
        return frame_bgr

    warped_bgr = warped[:, :, :3]
    warped_alpha = warped[:, :, 3]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [ordered_i32], 255)

    if feather > 0:
        k = feather | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    alpha_8u = cv2.multiply(mask, warped_alpha, scale=1.0/255.0)
    alpha_3ch = cv2.merge([alpha_8u, alpha_8u, alpha_8u])

    frame_16 = frame_bgr.astype(np.uint16)
    warped_16 = warped_bgr.astype(np.uint16)
    
    out_16 = frame_16 * (255 - alpha_3ch) + warped_16 * alpha_3ch
    out = (out_16 // 255).astype(np.uint8)
    return out
