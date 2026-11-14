import argparse
import sys

import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def open_camera(index):
    # macOS: the AVFoundation backend is more reliable; elsewhere use the default.
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def main():
    parser = argparse.ArgumentParser(description="Live hand-landmark tracking from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    args = parser.parse_args()

    cap = open_camera(args.camera)

    # Optional: set resolution (safe to keep)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        print("Try another camera: python3 app.py --camera 1")
        return

    print("✅ Webcam started! Show your hand. Press Q to quit.")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Failed to read frame from camera.")
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR -> RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            results = hands.process(frame_rgb)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # Show window
            cv2.imshow("Sign Language App", frame)

            # Quit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()