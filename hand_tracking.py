"""Camera and MediaPipe helpers shared by app.py and collect_data.py.

This is the only module that imports OpenCV and MediaPipe. The scripts import
it inside main(), so features.py, dataset.py, predict.py and train.py (and the
tests) work without a webcam or those packages.
"""

import sys

import cv2
import mediapipe as mp

if not hasattr(mp, "solutions"):
    raise SystemExit(
        f"mediapipe {mp.__version__} no longer includes the 'mp.solutions' hand tracker "
        "this project uses. Install the pinned version: pip install -r requirements.txt"
    )

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def open_camera(index):
    # macOS: the AVFoundation backend is more reliable; elsewhere use the default.
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def make_hands():
    return mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def read_hand(hands, frame):
    """Run MediaPipe on a BGR frame.

    Returns (hand_landmarks, points, handedness): the MediaPipe object for
    drawing, the 21 raw (x, y, z) tuples, and "Left"/"Right". All three are
    None when no hand is found.

    The handedness label is whatever MediaPipe reports for the frame it is
    given. Both scripts mirror the frame first, so the label is consistent
    between recording and prediction, which is what the model needs.
    """
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None, None, None
    hand_landmarks = results.multi_hand_landmarks[0]
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    handedness = "Right"
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label
    return hand_landmarks, points, handedness


def draw_hand(frame, hand_landmarks):
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def draw_text(frame, text, origin, scale=0.8, color=(255, 255, 255), thickness=2):
    """Text with a dark outline so it stays readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, origin, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)
