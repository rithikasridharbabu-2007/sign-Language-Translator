# Sign Language Translator

> **Status: early prototype.** The app tracks a hand live from a webcam and draws its 21 landmarks. It does not recognise or translate signs yet; that is the next step below.

## What works today

- Opens the webcam and mirrors the image.
- Detects one hand per frame with [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and draws the 21 landmarks and their connections.
- Press **Q** to quit.

## Run it

Requires a webcam and a Python version that MediaPipe supports (it usually lags the newest Python release; check its PyPI page).

```bash
pip install -r requirements.txt
python3 app.py              # default camera
python3 app.py --camera 1   # a different camera
```

If no window appears, run `python3 camera_test.py` to check that OpenCV can read your camera at all. On macOS, allow camera access for your terminal in System Settings → Privacy & Security → Camera.

## Planned next steps

1. **Collect data:** record landmark coordinates for a small set of static ASL fingerspelling letters, saved as CSV.
2. **Classify static letters:** normalise landmarks relative to the wrist and hand size, then train a small classifier on them.
3. **Show the result:** overlay the predicted letter and confidence on the video.
4. **Motion signs:** feed sequences of landmarks to a sequence model for signs that involve movement (such as J and Z).

## Files

| File | Purpose |
|---|---|
| `app.py` | Webcam loop with hand-landmark tracking |
| `camera_test.py` | Minimal check that the camera opens |
