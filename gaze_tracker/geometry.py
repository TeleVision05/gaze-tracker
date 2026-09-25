"""Pure-numpy eye geometry and pupil centroid maths.

Nothing in this module touches OpenCV, dlib, tkinter or a camera, so it can be
imported and tested anywhere numpy is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]

# dlib 68-point face model, eye landmark indices.
# Left eye:  36 outer, 37/38 top, 39 inner, 40/41 bottom
# Right eye: 42 inner, 43/44 top, 45 outer, 46/47 bottom
LEFT_EYE = {
    "outer": 36,
    "inner": 39,
    "top": (37, 38),
    "bottom": (40, 41),
    "all": list(range(36, 42)),
}
RIGHT_EYE = {
    "outer": 45,
    "inner": 42,
    "top": (43, 44),
    "bottom": (46, 47),
    "all": list(range(42, 48)),
}

# Fraction of the darkest pixels treated as pupil candidates.
PUPIL_DARK_PERCENTILE = 18
# Below this many dark pixels, fall back to the single darkest pixel.
MIN_MASK_PIXELS = 5
# Eye width / height is clipped to this range before stretching vertical offsets.
STRETCH_MIN = 1.5
STRETCH_MAX = 4.0
# Pupil ratios are clipped to this range (slightly outside the eye box).
RATIO_MIN = -0.3
RATIO_MAX = 1.3


def midpoint(p1: Point, p2: Point) -> Point:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


@dataclass(frozen=True)
class EyeBox:
    """Corner and lid positions of one eye in image coordinates.

    ``left_corner`` / ``right_corner`` are ordered by x, regardless of which
    landmark is the anatomical inner or outer corner.
    """

    left_corner: Point
    right_corner: Point
    top: Point
    bottom: Point

    @property
    def width(self) -> float:
        return self.right_corner[0] - self.left_corner[0]

    @property
    def height(self) -> float:
        return self.bottom[1] - self.top[1]

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height (open eyes are roughly 2 to 4)."""
        return self.width / self.height

    @property
    def centre(self) -> Point:
        return (
            (self.left_corner[0] + self.right_corner[0]) / 2.0,
            (self.top[1] + self.bottom[1]) / 2.0,
        )


def eye_box(pts: dict[int, Point], eye: dict) -> EyeBox:
    """Build an :class:`EyeBox` from a ``{landmark_index: (x, y)}`` mapping."""
    corner_a = pts[eye["outer"]]
    corner_b = pts[eye["inner"]]
    top = midpoint(pts[eye["top"][0]], pts[eye["top"][1]])
    bottom = midpoint(pts[eye["bottom"][0]], pts[eye["bottom"][1]])
    if corner_a[0] <= corner_b[0]:
        left_c, right_c = corner_a, corner_b
    else:
        left_c, right_c = corner_b, corner_a
    return EyeBox(left_c, right_c, top, bottom)


def crop_bounds(
    pts: dict[int, Point], frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Padded (x0, y0, x1, y1) crop around the eye landmarks, clamped to the frame.

    Vertical padding is a full eye height because looking up or down moves the
    iris toward the lids.
    """
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    pad_x = max(2, (x1 - x0) // 4)
    pad_y = max(4, (y1 - y0))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(frame_w, x1 + pad_x),
        min(frame_h, y1 + pad_y),
    )


def pupil_ratios(px: float, py: float, box: EyeBox) -> tuple[float, float] | None:
    """Pupil position as (horizontal, vertical) ratios, 0.5 meaning centred.

    Horizontal is the fraction across the eye width. Vertical is the lid
    fraction stretched by the eye's aspect ratio so up/down has a dynamic
    range comparable to left/right. Returns ``None`` for a degenerate eye.
    """
    eye_w = box.width
    eye_h = box.height
    if eye_w < 1 or abs(eye_h) < 1:
        return None

    cx, _ = box.centre
    hx = 0.5 + (px - cx) / eye_w
    vy_lid = (py - box.top[1]) / eye_h
    stretch = float(np.clip(eye_w / eye_h, STRETCH_MIN, STRETCH_MAX))
    vy = 0.5 + (vy_lid - 0.5) * stretch
    return (
        float(np.clip(hx, RATIO_MIN, RATIO_MAX)),
        float(np.clip(vy, RATIO_MIN, RATIO_MAX)),
    )


def roi_margins(h: int, w: int) -> tuple[int, int]:
    """(x, y) border to ignore in an eye crop so lids and lashes are excluded."""
    return max(1, w // 10), max(1, h // 8)


def dark_mask(
    roi: np.ndarray, percentile: float = PUPIL_DARK_PERCENTILE
) -> tuple[np.ndarray, float]:
    """Boolean mask of the darkest ``percentile`` of pixels and the cutoff value."""
    flat = roi.reshape(-1).astype(np.float32)
    cutoff = float(np.percentile(flat, percentile))
    return roi <= cutoff, cutoff


def pupil_centroid(
    roi: np.ndarray, mask: np.ndarray, cutoff: float
) -> tuple[float, float]:
    """Darkness-weighted centroid (x, y) of the masked pixels.

    Pixels are weighted by how far below ``cutoff`` they are. With fewer than
    :data:`MIN_MASK_PIXELS` masked pixels, the single darkest pixel is used.
    """
    ys, xs = np.where(mask)
    if len(xs) < MIN_MASK_PIXELS:
        y, x = np.unravel_index(int(np.argmin(roi)), roi.shape)
        return float(x), float(y)
    weights = (cutoff + 1.0) - roi[ys, xs].astype(np.float32)
    weights = np.clip(weights, 1e-3, None)
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))
