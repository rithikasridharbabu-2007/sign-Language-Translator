import os
import sys

import numpy as np
import pytest

# Make the top-level modules importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fake_class_centres(n_classes, seed=0):
    """One made-up 21-landmark 'hand shape' per class, far apart from each other.

    These are random points, NOT real hand poses. They only exist to exercise
    the code.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(n_classes, 21, 3))


@pytest.fixture
def synthetic_csv(tmp_path):
    """Write a synthetic two-session CSV in a temp dir and return its path."""
    from dataset import append_sample

    def make(letters="ABCD", per_class=40, sessions=("s1", "s2"), noise=0.005, name="landmarks.csv"):
        path = tmp_path / name
        rng = np.random.default_rng(1)
        centres = fake_class_centres(len(letters))
        for session in sessions:
            for letter, centre in zip(letters, centres):
                for i in range(per_class):
                    points = centre + rng.normal(0, noise, size=(21, 3))
                    points += rng.uniform(-0.1, 0.1, size=3)  # hand somewhere else in frame
                    hand = "Right" if i % 2 else "Left"
                    if hand == "Left":
                        points[:, 0] = 1.0 - points[:, 0]
                    append_sample(str(path), session, letter, hand, 1280, 720, points)
        return str(path)

    return make
