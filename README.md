# Sign Language Translator

> **Status: pipeline ready, no trained model yet.** The code to record data, train a classifier for static ASL fingerspelling letters and show the predicted letter live is written and tested on synthetic data. No real data has been recorded and no model has been trained, so there is no accuracy number yet. Until a model exists, the app only tracks a hand and draws its 21 landmarks.

## Results

**TODO (after recording real data):**

- Accuracy: _not measured yet._ To be filled in from `results.json`, stating whether it is a random-frame split or a held-out session.
- Confusion matrix: _not generated yet_ (`confusion_matrix.png`).
- Demo GIF: _not recorded yet._

## What works today

- Opens the webcam, mirrors the image, detects one hand per frame with [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and draws the 21 landmarks.
- `collect_data.py` records labelled landmarks to a CSV.
- `train.py` trains and evaluates a small classifier on that CSV.
- `app.py` shows the predicted letter and confidence when `model.joblib` exists.

## Install

Use Python 3.9 to 3.12: the pinned MediaPipe release has no build for newer versions. Tested with Python 3.11 on macOS (Apple Silicon).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`mediapipe` is pinned to 0.10.21 on purpose. MediaPipe 1.x removed the `mp.solutions.hands` API this project uses, so a plain `pip install mediapipe` no longer works with it.

If no window appears, run `python3 camera_test.py` to check that OpenCV can read your camera at all. On macOS, allow camera access for your terminal in System Settings → Privacy & Security → Camera.

## Train it yourself

Three commands:

```bash
python3 collect_data.py   # 1. record samples of each letter
python3 train.py          # 2. train, evaluate, save model.joblib
python3 app.py            # 3. run live with the predicted letter on screen
```

**1. Collect.** Make a letter's hand shape and press (or hold) that letter's key; each key event saves one frame. The window shows a counter per letter against the target (200 by default). Move your hand a little while recording (nearer, further, slightly turned), otherwise the samples are near-copies of each other. Q quits. Data is appended to `data/landmarks.csv`, so you can stop and continue later. A wrong key press saves a wrongly labelled row: delete the last lines of the CSV to undo it.

The default letters are `A B C D F I L V W Y`: ten static letters with clearly different hand shapes. Change them with `--letters`, for example `python3 collect_data.py --letters ABCDEFGHIK`. **J and Z are left out because they are signed with a movement** (J traces a hook, Z a zigzag), and this pipeline classifies single frames. Letters that look alike as landmarks (M, N, S, T and E are all fists; U, R, K and V differ by small finger positions) can be added, but expect them to be confused more often.

**2. Train.** `train.py` compares a small neural network (MLP) and a random forest by 5-fold cross-validation on the training split, keeps the better one, and reports test accuracy, per-letter accuracy and a confusion matrix. It writes `model.joblib`, `results.json` and `confusion_matrix.png`.

**3. Run.** `app.py` averages the prediction over the last 10 frames (`--smooth`) and shows "unsure" when confidence is below 60% (`--threshold`). `--camera 1` selects another camera in both `app.py` and `collect_data.py`.

### Getting an honest accuracy number

By default `train.py` tests on a random 20% of the frames. Frames recorded a fraction of a second apart are almost identical, so this number is too optimistic; the script prints a warning saying so. For a number that means more, record a second session on another day or in different lighting and test on it as a whole:

```bash
python3 collect_data.py                   # run again later: a new session
python3 train.py --holdout-session last   # train on earlier sessions, test on the newest
```

Every run of `collect_data.py` stamps its rows with a session id (its start time), which is what makes this possible.

## Features

Each sample is the 21 hand landmarks (x, y, z) from MediaPipe. The CSV stores them raw, so the processing below can be changed without re-recording. `features.py` turns them into 60 numbers:

1. **Aspect ratio:** x is a fraction of the frame width and y of the frame height, so x and z are rescaled to the same unit as y.
2. **Position:** every landmark is taken relative to the wrist. The wrist becomes (0, 0, 0) and is dropped, leaving 20 × 3 values.
3. **Size:** values are divided by the wrist to middle-finger-knuckle distance, so distance from the camera does not matter. If that distance is zero (a broken detection) the division is skipped.
4. **Handedness:** left hands are mirrored, so one model serves both hands.

Rotation is not normalised: a letter signed at a very different angle from the recordings looks different to the model.

## Limitations

- **Static letters only.** One frame at a time; no J, Z, words or sentences.
- **One signer.** A model trained on one person's hand is only tested on that person. Accuracy for other people is unknown until other people record data.
- **Lighting and camera.** Recognition depends on MediaPipe finding the hand, which gets worse in dim light, with motion blur, or when fingers hide each other.
- **One hand**, facing the camera roughly as it was during recording.
- **A small letter set.** Ten letters by default, chosen to be easy to tell apart. The full alphabet will score lower.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

The tests need no webcam. They cover the feature maths, the CSV round trip, the prediction smoothing, and `train.py` end to end on synthetic landmarks generated in a temporary folder. Those synthetic classes are random points made to be easy to separate; they check that the code runs, not how well real signs are recognised.

## Files

| File | Purpose |
|---|---|
| `app.py` | Webcam loop: hand tracking, plus letter prediction if a model exists |
| `collect_data.py` | Record labelled landmarks to `data/landmarks.csv` |
| `train.py` | Train and evaluate the classifier |
| `features.py` | Landmarks to feature vector (no camera code) |
| `dataset.py` | Read and write the landmark CSV |
| `predict.py` | Load the model, smooth predictions over frames |
| `hand_tracking.py` | Camera and MediaPipe helpers (the only module that imports them) |
| `camera_test.py` | Minimal check that the camera opens |
| `tests/` | pytest suite |

**What is committed:** `data/` is git-ignored: it is large, personal and can be re-recorded. `model.joblib`, `results.json` and `confusion_matrix.png` are not ignored, so they can be committed once they come from real recordings, which lets others run the app without recording anything. A `.joblib` file only loads reliably with the scikit-learn version that wrote it (recorded in `results.json`), and like any pickle it should only be loaded from a source you trust.

## Later

- Motion signs (J, Z and whole words) by feeding sequences of landmarks to a sequence model.
- Data from more than one signer.
