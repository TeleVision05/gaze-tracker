"""Quadratic gaze-to-screen calibration (numpy only)."""

from __future__ import annotations

import numpy as np

MARGIN = 80


def poly_features(gx: float, gy: float) -> np.ndarray:
    """Quadratic design vector used by standard gaze calibration."""
    return np.array([1.0, gx, gy, gx * gy, gx * gx, gy * gy], dtype=np.float64)


def fit_gaze_model(
    gaze_xy: np.ndarray, screen_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``screen = W @ poly_features(normalised gx, gy)``.

    Features are standardised so the smaller vertical signal is not
    ridge-shrunk away. Returns ``(weights 2x6, mean 2, std 2)``.
    """
    gaze_xy = np.asarray(gaze_xy, dtype=np.float64)
    screen_xy = np.asarray(screen_xy, dtype=np.float64)
    mean = gaze_xy.mean(axis=0)
    std = gaze_xy.std(axis=0)
    std = np.where(std < 1e-4, 1.0, std)
    normed = (gaze_xy - mean) / std

    design = np.vstack([poly_features(g[0], g[1]) for g in normed])
    reg = 1e-6 * np.eye(design.shape[1])
    ata = design.T @ design + reg
    bx = np.linalg.solve(ata, design.T @ screen_xy[:, 0])
    by = np.linalg.solve(ata, design.T @ screen_xy[:, 1])
    return np.vstack([bx, by]), mean, std


def map_gaze(
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    gx: float,
    gy: float,
) -> tuple[float, float]:
    """Map a gaze ratio pair to screen pixels with a fitted model."""
    nx = (gx - mean[0]) / std[0]
    ny = (gy - mean[1]) / std[1]
    v = weights @ poly_features(nx, ny)
    return float(v[0]), float(v[1])


def calibration_error(
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    gaze_xy: np.ndarray,
    screen_xy: np.ndarray,
) -> float:
    """Mean Euclidean pixel error of the model over the calibration points."""
    errs = []
    for g, s in zip(gaze_xy, screen_xy):
        px, py = map_gaze(weights, mean, std, g[0], g[1])
        errs.append(((px - s[0]) ** 2 + (py - s[1]) ** 2) ** 0.5)
    return float(np.mean(errs))


def calibration_targets(
    width: int, height: int, margin: int = MARGIN
) -> list[tuple[int, int]]:
    """3x3 grid of screen targets in row-major order (top-left first)."""
    xs = [margin, width // 2, width - margin]
    ys = [margin, height // 2, height - margin]
    return [(x, y) for y in ys for x in xs]
