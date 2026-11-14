"""Live hand tracking, plus the predicted ASL letter if a trained model exists.

    python3 app.py              # default camera
    python3 app.py --camera 1   # a different camera

Without model.joblib this only draws the hand landmarks.
"""

import argparse
import sys

from predict import DEFAULT_MODEL, LetterPredictor, load_model

HOW_TO_TRAIN = (
    "No trained model found ({path}), so only hand landmarks are shown.\n"
    "To create one:\n"
    "  python3 collect_data.py   # record samples of each letter\n"
    "  python3 train.py          # train and save model.joblib\n"
    "  python3 app.py            # run again"
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Live hand tracking with ASL static-letter prediction."
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"trained model file (default: {DEFAULT_MODEL})")
    parser.add_argument("--smooth", type=int, default=10,
                        help="average predictions over this many frames (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="show 'unsure' below this confidence (default: 0.6)")
    args = parser.parse_args(argv)

    bundle = load_model(args.model)
    predictor = None
    if bundle is None:
        print(HOW_TO_TRAIN.format(path=args.model))
    else:
        predictor = LetterPredictor(bundle["model"], bundle["labels"],
                                    window=max(1, args.smooth), threshold=args.threshold)
        print(f"Loaded {args.model}: letters {' '.join(bundle['labels'])}")

    # Imported here so --help works without a camera stack.
    import cv2
    from hand_tracking import draw_hand, draw_text, make_hands, open_camera, read_hand

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print("Camera could not be opened.")
        print("Try another camera: python3 app.py --camera 1")
        return 1

    print("Webcam started. Show your hand. Press Q to quit.")

    with make_hands() as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror, like looking in a mirror
            height, width = frame.shape[:2]
            hand_landmarks, points, handedness = read_hand(hands, frame)
            if hand_landmarks is not None:
                draw_hand(frame, hand_landmarks)

            if predictor is not None:
                letter, confidence = predictor.update(points, handedness, width / height)
                if points is None:
                    draw_text(frame, "no hand", (20, 50), 1.0, (200, 200, 200))
                elif letter is None:
                    draw_text(frame, f"unsure ({confidence:.0%})", (20, 50), 1.0, (0, 200, 255))
                else:
                    draw_text(frame, letter, (20, 90), 3.0, (0, 255, 0), 5)
                    draw_text(frame, f"{confidence:.0%}", (120, 85), 1.0, (0, 255, 0))

            cv2.imshow("Sign Language App", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
