# Roadmap

## Current state

What works, and how it was verified (2026-09-25, import into git):

- **Pure maths is tested offline.** `gaze_tracker/geometry.py` and
  `gaze_tracker/calibration.py` import only numpy. `python -m pytest tests -v`
  runs 26 tests in ~0.2 s and they also pass under a system Python with no
  OpenCV or dlib installed (`python -m unittest discover -s tests`).
- **Every module byte-compiles** (`python -m py_compile` over all `.py`), and
  `gaze_tracker.app` imports cleanly in a venv with OpenCV 4.11, dlib 20.0 and
  numpy 2.3 (the two venvs the code was developed in).
- **The app itself** (camera loop, 9-point calibration, tracking dot, debug
  overlay, live vertical gain) was carried over from the original single-file
  script with its behaviour unchanged. It was not re-run on a webcam as part
  of this import; the last known-working run was the original script in
  July 2026.

## Known gaps

Concrete, from reading the code:

- **Head movement breaks tracking.** Gaze ratios are measured relative to eye
  corners only (`pupil_ratios`); there is no head-pose compensation, so
  translating or rotating the head after calibration shifts the dot.
- **Vertical is weak by construction.** Pupil travel between the lids is a
  few pixels, so `pupil_ratios` stretches it by the eye's aspect ratio and
  `app.py` multiplies again by `Y_GAIN = 2.0`, with `+`/`-` to tweak live.
  The stretch is clipped to `[1.5, 4.0]` and the gain to `[0.8, 4.0]`, both
  hand-picked.
- **Blink handling is crude.** A frame is only dropped when the two eyes
  disagree by more than 0.4 horizontally or 0.7 vertically
  (`_estimate_gaze`); a symmetric blink passes through as a bad sample.
- **Calibration is not persisted.** Weights live on the `GazeTracker`
  instance and are lost on exit; every launch recalibrates.
- **Calibration failure is only reported, not diagnosed.** Fewer than 6
  points shows "need more points"; there is no per-point residual shown.
- **Pupil detection is a darkest-percentile heuristic** (`dark_mask` at the
  18th percentile after `equalizeHist`). Dark eyebrows or heavy eyeliner in
  the padded crop can capture the centroid.
- **The camera loop runs on tkinter's `after(16)`** with a blocking
  `cap.read()`, so a slow camera stalls the UI.
- **No `vision.py` tests.** `detect_pupil` and `eye_ratios` need OpenCV and
  a dlib `full_object_detection`; only their numpy internals are tested.
- **Single-face assumption.** The largest detected face wins; nothing warns
  when several faces are present.

## Milestones

### M1: Reproducible local run

- [ ] Run `python main.py` end to end on a webcam and record the reported
      calibration error in this file.
- [ ] Add a `--list-cameras` helper so `GAZE_CAMERA_INDEX` can be chosen
      without trial and error.
- [ ] Print a clear error (not a traceback) when the camera cannot be opened.

Done when: a fresh clone plus the README steps produces a tracking dot, and
the observed calibration error is written here with the date.

### M2: OpenCV-level tests

- [ ] Fixture: a synthetic grayscale eye crop (light sclera, dark disc) that
      `vision.detect_pupil` must locate within 1 px.
- [ ] Fixture: a fake `full_object_detection` (object with `.part(i).x/.y`)
      so `vision.eye_ratios` can be tested without dlib.
- [ ] Mark these tests to skip when OpenCV is not importable.

Done when: `pytest` covers `vision.py` and still passes without a camera.

### M3: Persisted calibration

- [ ] Save `(weights, mean, std, screen size)` to a JSON file after a
      successful calibration.
- [ ] `--load-calibration PATH` skips the calibration phase when the screen
      size matches.
- [ ] Show per-point residuals on the calibration-complete screen.

Done when: relaunching with a saved calibration starts tracking immediately
and the residual display matches `calibration_error`.

### M4: Better blink and outlier rejection

- [ ] Compute a per-eye lid aspect ratio from `EyeBox` and drop frames when
      both eyes are below a threshold (blink).
- [ ] Reject calibration samples more than 2 sigma from the point median
      before fitting.

Done when: blinking during tracking no longer jerks the dot, verified by the
debug overlay showing dropped frames.

### M5: Head-pose compensation

- [ ] Estimate head translation/rotation from the full 68 landmarks
      (`solvePnP` against a generic 3D face model).
- [ ] Feed pose terms into the calibration features or normalise eye crops
      by pose before `pupil_ratios`.

Done when: moving the head ~5 cm after calibration shifts the dot by less
than the current per-point calibration error.

## Won't do / out of scope

- Training or shipping a learned gaze model (this project is the geometric
  baseline).
- Multi-user or multi-face tracking.
- Recording video or gaze logs to disk.
- Packaging as a pip-installable library or a GUI installer.
- Committing the dlib model; it stays a documented download.

## Open questions

- Is a 3x3 calibration grid enough for the quadratic model (6 terms per
  axis, 9 points), or does a 4x4 grid measurably lower the error?
- Should the vertical stretch in `pupil_ratios` be learned per user during
  calibration instead of clipped to `[1.5, 4.0]`?
- Would a MediaPipe Face Mesh iris landmark replace the darkest-region
  centroid with less noise, at the cost of a heavier dependency?
- Should the two eyes be fitted separately and fused, rather than averaged
  before the fit?
