import numpy as np

from predict import LetterPredictor


class ScriptedModel:
    """Returns a fixed sequence of probability rows, one per call."""

    def __init__(self, rows):
        self.rows = list(rows)

    def predict_proba(self, X):
        return np.array([self.rows.pop(0)])


HAND = np.random.default_rng(0).uniform(0.2, 0.8, size=(21, 3))


def test_smoothing_averages_recent_frames():
    model = ScriptedModel([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    predictor = LetterPredictor(model, ["A", "B"], window=3, threshold=0.5)
    assert predictor.update(HAND, "Right")[0] == "A"
    assert predictor.update(HAND, "Right")[0] == "A"
    # One odd frame does not flip the answer: mean is still (0.63, 0.37).
    letter, confidence = predictor.update(HAND, "Right")
    assert letter == "A" and abs(confidence - 1.9 / 3) < 1e-9


def test_window_forgets_old_frames():
    model = ScriptedModel([[1, 0], [0, 1], [0, 1]])
    predictor = LetterPredictor(model, ["A", "B"], window=2, threshold=0.5)
    for _ in range(3):
        letter, _ = predictor.update(HAND, "Right")
    assert letter == "B"


def test_unsure_below_threshold():
    model = ScriptedModel([[0.55, 0.45]])
    predictor = LetterPredictor(model, ["A", "B"], window=5, threshold=0.6)
    letter, confidence = predictor.update(HAND, "Right")
    assert letter is None and abs(confidence - 0.55) < 1e-9


def test_no_hand_resets_history():
    model = ScriptedModel([[1, 0], [0, 1]])
    predictor = LetterPredictor(model, ["A", "B"], window=5, threshold=0.6)
    predictor.update(HAND, "Right")
    assert predictor.update(None, None) == (None, 0.0)
    assert predictor.update(HAND, "Right")[0] == "B"  # old A frame is gone
