"""Fullscreen tkinter application: calibrate, then a circle follows your gaze.

This is the only module that opens a camera or a window. dlib is imported
lazily inside :class:`GazeTracker` so the rest of the package (and the tests)
work without it.
"""

from __future__ import annotations

import argparse
import os
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

from .calibration import calibration_error, calibration_targets, fit_gaze_model, map_gaze
from .geometry import LEFT_EYE, RIGHT_EYE
from .vision import eye_ratios

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "shape_predictor_68_face_landmarks.dat"
MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

CALIB_SETTLE_S = 0.45  # wait for saccade before sampling
CALIB_HOLD_S = 1.5
MIN_SAMPLES = 18
SMOOTH_GAZE = 0.35  # EMA on gaze features (before mapping)
SMOOTH_SCREEN = 0.40  # EMA on screen position
DOT_R = 18
# Vertical pupil travel is physically smaller than horizontal; boost after mapping.
Y_GAIN = 2.0
X_GAIN = 1.0
BG = "#1a1a1a"


def resolve_model_path(cli_value: str | None = None) -> Path:
    """Model path from the CLI flag, else ``GAZE_MODEL_PATH``, else the repo root."""
    if cli_value:
        return Path(cli_value).expanduser()
    env_value = os.environ.get("GAZE_MODEL_PATH")
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_MODEL_PATH


def resolve_camera_index(cli_value: int | None = None) -> int:
    """Camera index from the CLI flag, else ``GAZE_CAMERA_INDEX``, else 0."""
    if cli_value is not None:
        return cli_value
    return int(os.environ.get("GAZE_CAMERA_INDEX", "0"))


class GazeTracker:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, camera_index: int = 0):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing dlib model at {model_path}. Download {MODEL_URL}, "
                "decompress it with bunzip2 and place the .dat file at that path "
                "(or point GAZE_MODEL_PATH / --model at it)."
            )

        import dlib  # heavy native import; keep it out of the test path

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(model_path))

        self._build_window()
        self._bind_keys()
        self._open_camera(camera_index)

        self.phase = "calibrate"
        self.weights: np.ndarray | None = None
        self.gaze_mean: np.ndarray | None = None
        self.gaze_std: np.ndarray | None = None
        self.y_gain = Y_GAIN
        self.last_x = self.sw / 2
        self.last_y = self.sh / 2
        self.smooth_gx = 0.5
        self.smooth_gy = 0.5
        self.have_smooth_gaze = False
        self.debug = False

        self.targets = calibration_targets(self.sw, self.sh)
        self.calib_i = 0
        self.calib_samples: list[tuple[float, float]] = []
        self.calib_gaze: list[tuple[float, float]] = []
        self.calib_screen: list[tuple[float, float]] = []
        self.point_start: float | None = None
        self.collecting = False

        self.start_calibration()
        self.root.after(16, self._tick)

    # ------------------------------------------------------------------ setup

    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gaze Tracker")
        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)

        self.canvas = tk.Canvas(
            self.root, width=self.sw, height=self.sh, bg=BG, highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.hint = self.canvas.create_text(
            self.sw // 2,
            40,
            text="",
            fill="#cccccc",
            font=("Helvetica", 20),
            justify="center",
        )
        self.target = self.canvas.create_oval(0, 0, 0, 0, fill="#44cc88", outline="")
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill="#ff3344", outline="")
        self._hide(self.dot)

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("c", lambda e: self.start_calibration())
        self.root.bind("C", lambda e: self.start_calibration())
        self.root.bind("<space>", lambda e: self._skip_point())
        self.root.bind("d", lambda e: self._toggle_debug())
        self.root.bind("D", lambda e: self._toggle_debug())
        # Live vertical sensitivity: + / - keys
        self.root.bind("=", lambda e: self._bump_y_gain(0.15))
        self.root.bind("+", lambda e: self._bump_y_gain(0.15))
        self.root.bind("-", lambda e: self._bump_y_gain(-0.15))
        self.root.bind("_", lambda e: self._bump_y_gain(-0.15))

    def _open_camera(self, camera_index: int) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        # Prefer higher res for pupil detail; camera may ignore these
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --------------------------------------------------------------- drawing

    def _hide(self, item) -> None:
        self.canvas.coords(item, -100, -100, -100, -100)

    def _show_target(self, x: int, y: int, r: int = 24) -> None:
        self.canvas.coords(self.target, x - r, y - r, x + r, y + r)
        self.canvas.itemconfigure(self.target, fill="#44cc88")

    def _show_dot(self, x: float, y: float) -> None:
        r = DOT_R
        self.canvas.coords(self.dot, x - r, y - r, x + r, y + r)

    def _set_hint(self, text: str) -> None:
        self.canvas.itemconfigure(self.hint, text=text)

    # ----------------------------------------------------------- calibration

    def start_calibration(self) -> None:
        self.phase = "calibrate"
        self.weights = None
        self.gaze_mean = None
        self.gaze_std = None
        self.have_smooth_gaze = False
        self.calib_i = 0
        self.calib_samples = []
        self.calib_gaze = []
        self.calib_screen = []
        self.collecting = False
        self.point_start = None
        self._hide(self.dot)
        self._begin_point()

    def _bump_y_gain(self, delta: float) -> None:
        self.y_gain = float(np.clip(self.y_gain + delta, 0.8, 4.0))
        if self.phase == "track":
            self._set_hint(
                f"Tracking  ·  Y gain {self.y_gain:.2f}  ·  +/- adjust  ·  C recalibrate  ·  Esc quit"
            )

    def _begin_point(self) -> None:
        if self.calib_i >= len(self.targets):
            self._finish_calibration()
            return
        tx, ty = self.targets[self.calib_i]
        self._show_target(tx, ty)
        self.calib_samples = []
        self.collecting = True
        self.point_start = None
        self._set_hint(
            f"Look at the green dot  ({self.calib_i + 1}/{len(self.targets)})\n"
            "Keep head still · Space skip · Esc quit"
        )

    def _skip_point(self) -> None:
        if self.phase != "calibrate":
            return
        self.calib_i += 1
        self._begin_point()

    def _finish_calibration(self) -> None:
        if len(self.calib_gaze) < 6:
            self._set_hint("Calibration failed — need more points.\nPress C to retry.")
            self._hide(self.target)
            return

        gaze = np.array(self.calib_gaze, dtype=np.float64)
        screen = np.array(self.calib_screen, dtype=np.float64)
        self.weights, self.gaze_mean, self.gaze_std = fit_gaze_model(gaze, screen)
        mean_err = calibration_error(
            self.weights, self.gaze_mean, self.gaze_std, gaze, screen
        )

        self.phase = "track"
        self._hide(self.target)
        self._set_hint(
            f"Tracking (err ~{mean_err:.0f}px, Y gain {self.y_gain:.1f})  ·  "
            "+/- vertical  ·  C recalibrate  ·  Esc quit"
        )
        self.last_x = float(self.calib_screen[-1][0])
        self.last_y = float(self.calib_screen[-1][1])
        self._show_dot(self.last_x, self.last_y)

    def _calib_step(self, gaze: tuple[float, float] | None) -> None:
        now = time.monotonic()
        if gaze is None:
            self._set_hint(
                "Face / eyes not found — look at the green dot\n"
                f"({self.calib_i + 1}/{len(self.targets)})"
            )
            self.point_start = None
            self.calib_samples = []
            return

        if self.point_start is None:
            self.point_start = now
        elapsed = now - self.point_start

        # Don't sample until eyes have time to land on the target
        if elapsed < CALIB_SETTLE_S:
            progress = elapsed / CALIB_SETTLE_S
            self._set_hint(
                f"Move eyes to green · settling…  ({self.calib_i + 1}/{len(self.targets)})\n"
                f"{progress * 100:.0f}%"
            )
            return

        self.calib_samples.append(gaze)
        collect_elapsed = elapsed - CALIB_SETTLE_S
        progress = min(1.0, collect_elapsed / CALIB_HOLD_S)
        r = 24 + int(10 * progress)
        tx, ty = self.targets[self.calib_i]
        self.canvas.coords(self.target, tx - r, ty - r, tx + r, ty + r)
        self._set_hint(
            f"Hold gaze on green  ({self.calib_i + 1}/{len(self.targets)})\n"
            f"{progress * 100:.0f}%"
        )

        if collect_elapsed >= CALIB_HOLD_S and len(self.calib_samples) >= MIN_SAMPLES:
            # Median of latter half — rejects saccade leftovers
            half = self.calib_samples[len(self.calib_samples) // 2 :]
            avg = (
                float(np.median([g[0] for g in half])),
                float(np.median([g[1] for g in half])),
            )
            self.calib_gaze.append(avg)
            self.calib_screen.append(self.targets[self.calib_i])
            self.calib_i += 1
            self.collecting = False
            self.root.after(250, self._begin_point)

    # -------------------------------------------------------------- tracking

    def _toggle_debug(self) -> None:
        self.debug = not self.debug
        if not self.debug:
            cv2.destroyWindow("Gaze Debug")

    def _draw_debug(self, frame, shape, lg, rg) -> None:
        debug = frame.copy()
        for eye in (LEFT_EYE, RIGHT_EYE):
            for i in eye["all"]:
                p = shape.part(i)
                cv2.circle(debug, (p.x, p.y), 1, (0, 255, 0), -1)
        label = []
        if lg is not None:
            label.append(f"L {lg[0]:.2f},{lg[1]:.2f}")
        if rg is not None:
            label.append(f"R {rg[0]:.2f},{rg[1]:.2f}")
        cv2.putText(
            debug,
            " ".join(label) or "no pupils",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )
        cv2.imshow("Gaze Debug", debug)
        cv2.waitKey(1)

    def _estimate_gaze(self, frame) -> tuple[float, float] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces:
            return None

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)

        lg = eye_ratios(shape, gray, LEFT_EYE)
        rg = eye_ratios(shape, gray, RIGHT_EYE)

        if self.debug:
            self._draw_debug(frame, shape, lg, rg)

        if lg is None and rg is None:
            return None
        if lg is None:
            return rg
        if rg is None:
            return lg

        # Drop frames where eyes strongly disagree (blink / bad detect)
        if abs(lg[0] - rg[0]) > 0.4 or abs(lg[1] - rg[1]) > 0.7:
            return None

        return (lg[0] + rg[0]) / 2.0, (lg[1] + rg[1]) / 2.0

    def _smooth(self, gaze: tuple[float, float] | None) -> tuple[float, float] | None:
        if gaze is None:
            return None
        if not self.have_smooth_gaze:
            self.smooth_gx, self.smooth_gy = gaze
            self.have_smooth_gaze = True
        else:
            a = SMOOTH_GAZE
            self.smooth_gx = a * gaze[0] + (1 - a) * self.smooth_gx
            self.smooth_gy = a * gaze[1] + (1 - a) * self.smooth_gy
        return (self.smooth_gx, self.smooth_gy)

    def _tick(self) -> None:
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            smoothed = self._smooth(self._estimate_gaze(frame))

            if self.phase == "calibrate" and self.collecting:
                self._calib_step(smoothed)
            elif (
                self.phase == "track"
                and smoothed is not None
                and self.weights is not None
                and self.gaze_mean is not None
                and self.gaze_std is not None
            ):
                self._track_step(smoothed)

        self.root.after(16, self._tick)

    def _track_step(self, gaze: tuple[float, float]) -> None:
        gx, gy = gaze
        sx, sy = map_gaze(self.weights, self.gaze_mean, self.gaze_std, gx, gy)
        # Expand travel around screen center (vertical needs more help)
        cx, cy = self.sw / 2.0, self.sh / 2.0
        sx = cx + (sx - cx) * X_GAIN
        sy = cy + (sy - cy) * self.y_gain
        sx = float(np.clip(sx, 0, self.sw))
        sy = float(np.clip(sy, 0, self.sh))
        a = SMOOTH_SCREEN
        self.last_x = a * sx + (1 - a) * self.last_x
        self.last_y = a * sy + (1 - a) * self.last_y
        self._show_dot(self.last_x, self.last_y)

    # ------------------------------------------------------------- lifecycle

    def quit(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaze-tracker",
        description="Calibrate, then a fullscreen circle follows where you look.",
    )
    parser.add_argument(
        "--model",
        help="Path to shape_predictor_68_face_landmarks.dat "
        "(default: $GAZE_MODEL_PATH or the repo root).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="OpenCV camera index (default: $GAZE_CAMERA_INDEX or 0).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tracker = GazeTracker(
        model_path=resolve_model_path(args.model),
        camera_index=resolve_camera_index(args.camera),
    )
    tracker.run()


if __name__ == "__main__":
    main()
