"""Offline tests for the numpy-only eye geometry. No camera, OpenCV or dlib."""

import sys
import unittest

import numpy as np

from gaze_tracker import geometry
from gaze_tracker.geometry import (
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


def _synthetic_eye(eye: dict, x0: float = 100.0, y0: float = 50.0, w: float = 40.0, h: float = 20.0):
    """Landmark dict for one eye: corners w apart, lids h apart, centred."""
    cx = x0 + w / 2
    cy = y0 + h / 2
    left, right = (x0, cy), (x0 + w, cy)
    if eye["outer"] < eye["inner"]:  # left eye: outer corner is on the left
        outer, inner = left, right
    else:  # right eye: outer corner is on the right
        outer, inner = right, left
    pts = {
        eye["outer"]: outer,
        eye["inner"]: inner,
        eye["top"][0]: (cx - w / 4, y0),
        eye["top"][1]: (cx + w / 4, y0),
        eye["bottom"][0]: (cx + w / 4, y0 + h),
        eye["bottom"][1]: (cx - w / 4, y0 + h),
    }
    return pts


class ImportHygieneTests(unittest.TestCase):
    def test_geometry_does_not_import_cv2_or_dlib(self):
        self.assertNotIn("cv2", sys.modules)
        self.assertNotIn("dlib", sys.modules)
        self.assertIs(geometry.np, np)


class LandmarkIndexTests(unittest.TestCase):
    def test_eye_indices_are_disjoint_and_in_dlib_range(self):
        left = set(LEFT_EYE["all"])
        right = set(RIGHT_EYE["all"])
        self.assertEqual(left & right, set())
        self.assertEqual(left | right, set(range(36, 48)))
        for eye in (LEFT_EYE, RIGHT_EYE):
            used = {eye["outer"], eye["inner"], *eye["top"], *eye["bottom"]}
            self.assertEqual(used, set(eye["all"]))


class MidpointTests(unittest.TestCase):
    def test_midpoint(self):
        self.assertEqual(midpoint((0, 0), (4, 2)), (2.0, 1.0))


class EyeBoxTests(unittest.TestCase):
    def test_corners_ordered_by_x_for_both_eyes(self):
        for eye in (LEFT_EYE, RIGHT_EYE):
            box = eye_box(_synthetic_eye(eye), eye)
            self.assertLess(box.left_corner[0], box.right_corner[0])
            self.assertEqual(box.left_corner, (100.0, 60.0))
            self.assertEqual(box.right_corner, (140.0, 60.0))

    def test_width_height_and_aspect_ratio(self):
        box = eye_box(_synthetic_eye(LEFT_EYE, w=40, h=20), LEFT_EYE)
        self.assertAlmostEqual(box.width, 40.0)
        self.assertAlmostEqual(box.height, 20.0)
        self.assertAlmostEqual(box.aspect_ratio, 2.0)
        self.assertEqual(box.centre, (120.0, 60.0))

    def test_lid_midpoints(self):
        box = eye_box(_synthetic_eye(LEFT_EYE), LEFT_EYE)
        self.assertEqual(box.top, (120.0, 50.0))
        self.assertEqual(box.bottom, (120.0, 70.0))


class PupilRatioTests(unittest.TestCase):
    def setUp(self):
        self.box = eye_box(_synthetic_eye(LEFT_EYE, w=40, h=20), LEFT_EYE)

    def test_centred_pupil_is_half_half(self):
        self.assertEqual(pupil_ratios(120.0, 60.0, self.box), (0.5, 0.5))

    def test_horizontal_is_fraction_of_eye_width(self):
        hx, _ = pupil_ratios(130.0, 60.0, self.box)  # +10px of a 40px eye
        self.assertAlmostEqual(hx, 0.75)
        hx, _ = pupil_ratios(100.0, 60.0, self.box)  # at the left corner
        self.assertAlmostEqual(hx, 0.0)

    def test_vertical_is_stretched_by_aspect_ratio(self):
        # aspect 2.0 -> stretch 2.0; a quarter of the way down the lid gap
        _, vy = pupil_ratios(120.0, 55.0, self.box)
        self.assertAlmostEqual(vy, 0.0)

    def test_ratios_are_clipped(self):
        hx, vy = pupil_ratios(0.0, 0.0, self.box)
        self.assertEqual((hx, vy), (-0.3, -0.3))
        hx, vy = pupil_ratios(1000.0, 1000.0, self.box)
        self.assertEqual((hx, vy), (1.3, 1.3))

    def test_stretch_is_clamped_to_range(self):
        wide = EyeBox((0, 10), (100, 10), (50, 0), (50, 20))  # aspect 5 -> 4
        _, vy = pupil_ratios(50.0, 0.0, wide)
        self.assertAlmostEqual(vy, max(0.5 - 0.5 * 4.0, -0.3))
        square = EyeBox((0, 10), (20, 10), (10, 0), (10, 20))  # aspect 1 -> 1.5
        _, vy = pupil_ratios(10.0, 5.0, square)  # quarter down
        self.assertAlmostEqual(vy, 0.5 + (0.25 - 0.5) * 1.5)

    def test_degenerate_eye_returns_none(self):
        flat = EyeBox((0, 10), (40, 10), (20, 10), (20, 10))
        self.assertIsNone(pupil_ratios(20.0, 10.0, flat))
        narrow = EyeBox((0, 10), (0.5, 10), (0, 0), (0, 20))
        self.assertIsNone(pupil_ratios(0.0, 10.0, narrow))


class CropBoundsTests(unittest.TestCase):
    def test_padding_and_clamping(self):
        pts = _synthetic_eye(LEFT_EYE, x0=100, y0=50, w=40, h=20)
        # pad_x = 40 // 4 = 10, pad_y = max(4, 20) = 20
        self.assertEqual(crop_bounds(pts, 1280, 720), (90, 30, 150, 90))
        # clamps to the frame
        self.assertEqual(crop_bounds(pts, 120, 60), (90, 30, 120, 60))
        near_origin = _synthetic_eye(LEFT_EYE, x0=2, y0=3, w=40, h=20)
        self.assertEqual(crop_bounds(near_origin, 1280, 720)[:2], (0, 0))


class PupilCentroidTests(unittest.TestCase):
    def test_roi_margins(self):
        self.assertEqual(roi_margins(80, 100), (10, 10))
        self.assertEqual(roi_margins(6, 8), (1, 1))

    def test_dark_mask_selects_synthetic_blob(self):
        roi = np.full((20, 20), 200, dtype=np.uint8)
        roi[5:15, 5:15] = 0  # 25% of pixels, above the 18th percentile
        mask, cutoff = dark_mask(roi)
        self.assertEqual(cutoff, 0.0)
        self.assertEqual(int(mask.sum()), 100)
        self.assertTrue(mask[5:15, 5:15].all())
        self.assertFalse(mask[0, 0])

    def test_centroid_of_uniform_blob_is_its_centre(self):
        roi = np.full((20, 20), 200, dtype=np.uint8)
        roi[5:15, 5:15] = 0
        mask, cutoff = dark_mask(roi)
        cx, cy = pupil_centroid(roi, mask, cutoff)
        self.assertAlmostEqual(cx, 9.5)
        self.assertAlmostEqual(cy, 9.5)

    def test_centroid_is_weighted_toward_darker_pixels(self):
        roi = np.full((10, 30), 100, dtype=np.uint8)
        roi[4:6, 5:7] = 50   # lighter blob on the left
        roi[4:6, 23:25] = 0  # darker blob on the right
        mask = roi < 100
        cx, _ = pupil_centroid(roi, mask, 50.0)
        self.assertGreater(cx, 14.5)  # pulled right of the unweighted mean

    def test_fallback_to_darkest_pixel_when_mask_is_sparse(self):
        roi = np.full((12, 16), 180, dtype=np.uint8)
        roi[3, 7] = 20
        mask = np.zeros_like(roi, dtype=bool)
        self.assertEqual(pupil_centroid(roi, mask, 20.0), (7.0, 3.0))


if __name__ == "__main__":
    unittest.main()
