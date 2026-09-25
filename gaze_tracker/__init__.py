"""Calibrated webcam gaze tracker.

The package is split so the maths can be tested without a camera:

* ``geometry``    - eye landmark geometry and pupil centroid (numpy only)
* ``calibration`` - quadratic gaze -> screen calibration fit (numpy only)
* ``vision``      - OpenCV pupil detection on a real eye crop
* ``app``         - the tkinter/dlib/OpenCV application (see ``main.py``)

Importing ``gaze_tracker`` pulls in only the numpy modules; OpenCV, dlib and
tkinter are imported lazily by ``vision`` and ``app`` when they are used.
"""

from .calibration import (
    calibration_error,
    calibration_targets,
    fit_gaze_model,
    map_gaze,
    poly_features,
)
from .geometry import (
    LEFT_EYE,
    RIGHT_EYE,
    EyeBox,
    crop_bounds,
    dark_mask,
    eye_box,
    midpoint,
    pupil_centroid,
    pupil_ratios,
    roi_margins,
)

__all__ = [
    "LEFT_EYE",
    "RIGHT_EYE",
    "EyeBox",
    "calibration_error",
    "calibration_targets",
    "crop_bounds",
    "dark_mask",
    "eye_box",
    "fit_gaze_model",
    "map_gaze",
    "midpoint",
    "poly_features",
    "pupil_centroid",
    "pupil_ratios",
    "roi_margins",
]
