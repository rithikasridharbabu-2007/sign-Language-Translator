"""Runs only where OpenCV and a MediaPipe with the legacy hands API are installed.

No camera is used: MediaPipe is given a blank image, which checks that the
tracker starts and that the "no hand" path works. Detection of a real hand is
not tested here.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
mp = pytest.importorskip("mediapipe")
if not hasattr(mp, "solutions"):
    pytest.skip("this mediapipe version has no mp.solutions", allow_module_level=True)


def test_blank_frame_has_no_hand():
    from hand_tracking import make_hands, read_hand

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with make_hands() as hands:
        assert read_hand(hands, frame) == (None, None, None)
