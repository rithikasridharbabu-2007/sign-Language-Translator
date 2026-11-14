"""Load a trained model and turn per-frame probabilities into a steady letter.

No camera, OpenCV or MediaPipe imports here, so this can be tested anywhere.
"""

import os
from collections import deque

import numpy as np

from features import FEATURE_VERSION, landmarks_to_features

DEFAULT_MODEL = "model.joblib"
UNSURE = None


def load_model(path=DEFAULT_MODEL):
    """Return the bundle saved by train.py, or None if there is no model file."""
    if not os.path.exists(path):
        return None
    import joblib

    bundle = joblib.load(path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"{path} was trained with feature version {bundle.get('feature_version')}, "
            f"but features.py is now version {FEATURE_VERSION}. Run train.py again."
        )
    return bundle


class LetterPredictor:
    """Average class probabilities over the last `window` frames.

    update() returns (letter, confidence). letter is None ("unsure") when the
    averaged top probability is below `threshold`, or when no hand is visible.
    """

    def __init__(self, model, labels, window=10, threshold=0.6):
        self.model = model
        self.labels = list(labels)
        self.threshold = threshold
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def update(self, landmarks, handedness, aspect_ratio=1.0):
        if landmarks is None:
            self.reset()
            return UNSURE, 0.0
        features = landmarks_to_features(landmarks, handedness, aspect_ratio)
        self.history.append(self.model.predict_proba(features.reshape(1, -1))[0])
        return self.current()

    def current(self):
        if not self.history:
            return UNSURE, 0.0
        mean = np.mean(self.history, axis=0)
        best = int(np.argmax(mean))
        confidence = float(mean[best])
        if confidence < self.threshold:
            return UNSURE, confidence
        return self.labels[best], confidence
