"""OpenCV-backed pupil detection on a real eye crop.

Only this module and ``app`` import OpenCV; the maths they rely on lives in
``geometry`` so it can be tested without a camera.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import (
    Point,
    crop_bounds,
    dark_mask,
    eye_box,
    pupil_centroid,
    pupil_ratios,
    roi_margins,
)

PUPIL_UPSCALE = 4


def detect_pupil(eye_gray: np.ndarray) -> tuple[float, float] | None:
    """Darkest-region centroid; works in original or upscaled crop coords."""
    if eye_gray.size == 0 or eye_gray.shape[0] < 6 or eye_gray.shape[1] < 8:
        return None

    h, w = eye_gray.shape
    eye = cv2.equalizeHist(eye_gray)
    eye = cv2.GaussianBlur(eye, (5, 5), 0)

    # Ignore lid/lash borders (lighter vertically so up/down pupil is not clipped)
    mx, my = roi_margins(h, w)
    roi = eye[my : h - my, mx : w - mx]
    if roi.size == 0:
        return None

    mask, cutoff = dark_mask(roi)
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cx, cy = pupil_centroid(roi, mask_u8 > 0, cutoff)
    return (mx + cx, my + cy)


def landmark_points(shape, eye: dict) -> dict[int, Point]:
    """Extract ``{index: (x, y)}`` for one eye from a dlib full_object_detection."""
    return {i: (shape.part(i).x, shape.part(i).y) for i in eye["all"]}


def eye_ratios(shape, gray: np.ndarray, eye: dict) -> tuple[float, float] | None:
    """Pupil position of one eye as image-aligned (horizontal, vertical) ratios.

    Vertical span of the open eye is tiny compared with its width, so vertical
    offsets are measured in eye-width units then lid-normalised so up/down has
    usable dynamic range for the screen map.
    """
    pts = landmark_points(shape, eye)
    box = eye_box(pts, eye)

    x0, y0, x1, y1 = crop_bounds(pts, gray.shape[1], gray.shape[0])
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    # Upscale for sub-pixel-ish vertical sensitivity
    up = cv2.resize(
        crop,
        (crop.shape[1] * PUPIL_UPSCALE, crop.shape[0] * PUPIL_UPSCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    pupil_up = detect_pupil(up)
    if pupil_up is None:
        return None

    px = x0 + pupil_up[0] / PUPIL_UPSCALE
    py = y0 + pupil_up[1] / PUPIL_UPSCALE
    return pupil_ratios(px, py, box)
