# gaze-tracker

A calibrated webcam gaze tracker. You look at nine green dots on a fullscreen
window, it fits a quadratic map from pupil position to screen position, and
then a red circle follows where you look. No special hardware: a laptop
webcam, OpenCV, dlib's 68-point face landmark model, and tkinter.

It exists as a personal experiment in how far pure geometry (darkest-region
pupil centroid, eye-corner and lid landmarks, least-squares calibration) can
get you without a trained gaze model.

**Status: experimental.** It runs and tracks, but accuracy is a few degrees
at best, it needs your head held still, and vertical tracking is noticeably
weaker than horizontal (there is a live `+`/`-` vertical gain for that reason).

## Layout

```
main.py                     CLI entry point (python main.py)
gaze_tracker/
  geometry.py               eye-box geometry, pupil ratios, pupil centroid  (numpy only)
  calibration.py            quadratic gaze -> screen fit and 3x3 targets    (numpy only)
  vision.py                 OpenCV pupil detection on a real eye crop
  app.py                    tkinter fullscreen app, camera loop, dlib landmarks
tests/                      offline unit tests for geometry and calibration
```

`import gaze_tracker` pulls in only numpy. OpenCV, dlib and tkinter are only
imported by `vision` and `app`, and dlib is imported lazily inside the
`GazeTracker` constructor.

## Install

Python 3.10+ (developed on 3.13).

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # numpy, opencv-python, dlib
pip install -r requirements-dev.txt    # adds pytest
```

`dlib` builds from source on most platforms and needs CMake and a C++
compiler (`brew install cmake` on macOS). tkinter ships with python.org and
Homebrew Python.

### Landmark model (not in the repo)

The app needs dlib's 68-point face landmark model, ~100 MB, which is
gitignored (`*.dat`, `*.bz2`). Download and decompress it into the repo root:

```sh
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

You should end up with `shape_predictor_68_face_landmarks.dat` next to
`main.py`. To keep it elsewhere, set `GAZE_MODEL_PATH` or pass `--model`.

## Configure

No secrets are required. Two optional settings, read from the environment
(see `.env.example`) and overridable on the command line:

| Variable            | CLI flag   | Default                                       |
|---------------------|------------|-----------------------------------------------|
| `GAZE_MODEL_PATH`   | `--model`  | `<repo>/shape_predictor_68_face_landmarks.dat` |
| `GAZE_CAMERA_INDEX` | `--camera` | `0`                                           |

Tuning constants (settle/hold times, smoothing, vertical gain) are module-level
constants at the top of `gaze_tracker/app.py` and `gaze_tracker/geometry.py`.

## Run

```sh
python main.py
# or
python main.py --model ~/models/shape_predictor_68_face_landmarks.dat --camera 1
```

A fullscreen window opens and calibration starts immediately.

| Key         | Action                                              |
|-------------|-----------------------------------------------------|
| look at dot | hold your gaze on each green dot until it fills     |
| `Space`     | skip the current calibration point                  |
| `C`         | recalibrate                                         |
| `D`         | toggle the OpenCV debug window (landmarks + ratios) |
| `+` / `-`   | raise / lower vertical gain while tracking          |
| `Esc`       | quit                                                |

Keep your head still during and after calibration; the model maps pupil
position relative to eye corners, so moving your head moves the dot.

## Test

The tests cover the pure maths only and never open a camera, a window or
the dlib model. They need numpy and nothing else:

```sh
python -m pytest tests -v
# or, with no pytest installed
python -m unittest discover -s tests -v
```

## Data & secrets

- No credentials, API keys or network calls anywhere in the code.
- The only external data is the dlib landmark model, downloaded by you,
  gitignored, and never committed. The `.bz2` archive is ignored too.
- Nothing is recorded or written to disk: frames stay in memory and
  calibration lives only for the running session.
- `.env` is ignored; `.env.example` lists the two optional variable names.
