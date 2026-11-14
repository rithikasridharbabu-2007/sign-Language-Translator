"""Turn 21 raw hand landmarks into a feature vector for the letter classifier.

Pure NumPy: no camera, OpenCV or MediaPipe imports, so this module can be
tested anywhere.

Choices
-------
* Translation: every landmark is expressed relative to the wrist (landmark 0),
  so where the hand is in the frame does not matter. The wrist itself becomes
  (0, 0, 0) and is dropped, leaving 20 landmarks x 3 = 60 features.
* Aspect ratio: MediaPipe gives x as a fraction of the image WIDTH and y as a
  fraction of the image HEIGHT (z uses the same scale as x). On a 1280x720
  frame those units differ, so x and z are multiplied by width/height first to
  put all three axes in the same unit.
* Scale: coordinates are divided by the wrist -> middle-finger-MCP (landmark 9)
  distance, a bone length that does not change with the hand shape, so the
  distance to the camera does not matter. If that distance is ~0 (a degenerate
  detection) the division is skipped instead of producing inf/NaN.
* Handedness: left hands are mirrored (x negated) so that one model serves
  both hands.

Not handled: rotation. A letter signed with the hand tilted far from how it
was recorded will look different to the model.
"""

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
NUM_FEATURES = (NUM_LANDMARKS - 1) * 3
MIN_HAND_SIZE = 1e-6

# Bump this when the feature definition changes; it is stored with the model
# so app.py can refuse a model trained on different features.
FEATURE_VERSION = 1


def _as_landmark_array(landmarks):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape == (NUM_LANDMARKS * 3,):
        points = points.reshape(NUM_LANDMARKS, 3)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"expected {NUM_LANDMARKS} landmarks with (x, y, z) each, got shape {points.shape}"
        )
    return points


def is_left(handedness):
    return str(handedness).strip().lower().startswith("l")


def landmarks_to_features(landmarks, handedness="Right", aspect_ratio=1.0):
    """Return a (60,) float vector for one hand.

    landmarks    -- 21 (x, y, z) points as MediaPipe reports them (or 63 flat values).
    handedness   -- "Left" or "Right"; left hands are mirrored.
    aspect_ratio -- frame width / frame height of the image the landmarks came from.
    """
    points = _as_landmark_array(landmarks).copy()

    points[:, 0] *= aspect_ratio
    points[:, 2] *= aspect_ratio

    points -= points[WRIST]

    hand_size = np.linalg.norm(points[MIDDLE_MCP])
    if hand_size > MIN_HAND_SIZE:
        points /= hand_size

    if is_left(handedness):
        points[:, 0] *= -1.0

    return points[1:].reshape(-1)


def batch_to_features(landmarks, handedness, aspect_ratios=None):
    """Vectorised-by-loop version: (n, 21, 3) or (n, 63) -> (n, 60)."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    n = len(landmarks)
    if aspect_ratios is None:
        aspect_ratios = np.ones(n)
    out = np.empty((n, NUM_FEATURES))
    for i in range(n):
        out[i] = landmarks_to_features(landmarks[i], handedness[i], aspect_ratios[i])
    return out
