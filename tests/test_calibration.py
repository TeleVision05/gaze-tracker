"""Offline tests for the gaze -> screen calibration maths. No camera or dlib."""

import unittest

import numpy as np

from gaze_tracker.calibration import (
    calibration_error,
    calibration_targets,
    fit_gaze_model,
    map_gaze,
    poly_features,
)


def _grid(n: int = 5):
    gs = np.linspace(0.2, 0.8, n)
    return np.array([(gx, gy) for gy in gs for gx in gs], dtype=np.float64)


class PolyFeatureTests(unittest.TestCase):
    def test_quadratic_design_vector(self):
        np.testing.assert_array_equal(poly_features(2.0, 3.0), [1, 2, 3, 6, 4, 9])
        self.assertEqual(poly_features(0.0, 0.0).dtype, np.float64)


class CalibrationTargetTests(unittest.TestCase):
    def test_three_by_three_grid_row_major(self):
        targets = calibration_targets(1000, 800, margin=80)
        self.assertEqual(len(targets), 9)
        self.assertEqual(targets[0], (80, 80))
        self.assertEqual(targets[4], (500, 400))
        self.assertEqual(targets[8], (920, 720))
        self.assertEqual(targets[1], (500, 80))  # row-major: x varies first

    def test_default_margin(self):
        self.assertEqual(calibration_targets(1000, 800)[0], (80, 80))


class FitAndMapTests(unittest.TestCase):
    def test_recovers_affine_mapping(self):
        gaze = _grid()
        screen = np.column_stack([1000.0 * gaze[:, 0] + 50.0, 800.0 * gaze[:, 1] + 20.0])
        w, mean, std = fit_gaze_model(gaze, screen)
        self.assertEqual(w.shape, (2, 6))
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        sx, sy = map_gaze(w, mean, std, 0.5, 0.5)
        self.assertAlmostEqual(sx, 550.0, places=3)
        self.assertAlmostEqual(sy, 420.0, places=3)
        self.assertLess(calibration_error(w, mean, std, gaze, screen), 1e-3)

    def test_recovers_quadratic_mapping(self):
        gaze = _grid()
        gx, gy = gaze[:, 0], gaze[:, 1]
        screen = np.column_stack(
            [300 * gx * gx + 400 * gy + 10, 200 * gx * gy + 500 * gy * gy + 5]
        )
        w, mean, std = fit_gaze_model(gaze, screen)
        self.assertLess(calibration_error(w, mean, std, gaze, screen), 1e-3)
        sx, sy = map_gaze(w, mean, std, 0.35, 0.65)
        self.assertAlmostEqual(sx, 300 * 0.35**2 + 400 * 0.65 + 10, places=3)
        self.assertAlmostEqual(sy, 200 * 0.35 * 0.65 + 500 * 0.65**2 + 5, places=3)

    def test_constant_axis_does_not_produce_nan(self):
        gaze = np.array([(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)] * 3, dtype=np.float64)
        screen = np.column_stack([1000 * gaze[:, 0], np.full(len(gaze), 400.0)])
        w, mean, std = fit_gaze_model(gaze, screen)
        self.assertEqual(std[1], 1.0)  # guarded against division by ~0
        sx, sy = map_gaze(w, mean, std, 0.5, 0.5)
        self.assertTrue(np.isfinite([sx, sy]).all())
        self.assertAlmostEqual(sx, 500.0, places=3)
        self.assertAlmostEqual(sy, 400.0, places=3)

    def test_map_returns_python_floats(self):
        gaze = _grid(3)
        w, mean, std = fit_gaze_model(gaze, gaze * 100)
        sx, sy = map_gaze(w, mean, std, 0.4, 0.6)
        self.assertIsInstance(sx, float)
        self.assertIsInstance(sy, float)

    def test_accepts_lists_of_tuples(self):
        gaze = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2), (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]
        screen = [(80, 80), (500, 80), (920, 80), (80, 720), (500, 720), (920, 720)]
        w, mean, std = fit_gaze_model(gaze, screen)
        self.assertLess(calibration_error(w, mean, std, gaze, screen), 1e-3)


if __name__ == "__main__":
    unittest.main()
