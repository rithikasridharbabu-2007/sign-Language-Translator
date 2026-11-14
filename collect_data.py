"""Record labelled hand landmarks from the webcam into data/landmarks.csv.

    python3 collect_data.py                    # the 10 default letters
    python3 collect_data.py --letters ABCDEFGHIK

Hold a letter shape up to the camera and press (or hold) that letter's key:
each key event saves the current frame's 21 raw landmarks. Move your hand a
little between samples (closer, further, slightly turned) so the samples are
not all copies of each other. Press Q to quit (Esc always quits too).
"""

import argparse
import os
import sys
from datetime import datetime

from dataset import DEFAULT_CSV, append_sample, count_labels

# Ten static letters whose hand shapes differ clearly from each other.
# J and Z are left out because they are drawn in the air (motion), and
# look-alike groups such as M/N/S/T/E (all fists) and U/R/K/V are thinned out.
DEFAULT_LETTERS = "ABCDFILVWY"
MOTION_LETTERS = "JZ"
ESC = 27


def parse_letters(text):
    letters = []
    for ch in text.upper():
        if not ("A" <= ch <= "Z"):
            raise argparse.ArgumentTypeError(f"'{ch}' is not a letter A-Z")
        if ch not in letters:
            letters.append(ch)
    if not letters:
        raise argparse.ArgumentTypeError("give at least one letter")
    return letters


def new_session_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def counter_lines(letters, counts, target, per_line=5):
    cells = [f"{l}:{counts.get(l, 0)}/{target}" for l in letters]
    return ["  ".join(cells[i:i + per_line]) for i in range(0, len(cells), per_line)]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record labelled hand landmarks for static ASL letters."
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--letters", type=parse_letters, default=parse_letters(DEFAULT_LETTERS),
                        help=f"letters to record (default: {DEFAULT_LETTERS})")
    parser.add_argument("--target", type=int, default=200,
                        help="samples to aim for per letter (default: 200)")
    parser.add_argument("--output", default=DEFAULT_CSV,
                        help=f"CSV file to append to (default: {DEFAULT_CSV})")
    args = parser.parse_args(argv)

    letters = args.letters
    moving = [l for l in letters if l in MOTION_LETTERS]
    if moving:
        print(f"Note: {', '.join(moving)} involve movement in ASL. This tool saves single "
              "frames, so a still pose of them will not be the real sign.")
    q_is_letter = "Q" in letters
    quit_hint = "Esc to quit (Q records the letter Q)" if q_is_letter else "Q to quit"

    # Imported here so --help and the tests work without a camera stack.
    import cv2
    from hand_tracking import draw_hand, draw_text, make_hands, open_camera, read_hand

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print("Camera could not be opened.")
        print("Try another camera: python3 collect_data.py --camera 1")
        return 1

    session_id = new_session_id()
    counts = count_labels(args.output)
    saved_now = 0
    message = ""
    message_frames = 0

    print(f"Session {session_id}. Appending to {args.output}.")
    print(f"Letters: {' '.join(letters)}. Press a letter key to save a sample; {quit_hint}.")

    with make_hands() as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            hand_landmarks, points, handedness = read_hand(hands, frame)
            if hand_landmarks is not None:
                draw_hand(frame, hand_landmarks)

            y = 30
            for line in counter_lines(letters, counts, args.target):
                draw_text(frame, line, (10, y), 0.7)
                y += 28
            done = sum(1 for l in letters if counts.get(l, 0) >= args.target)
            status = "hand: " + (handedness if points else "none")
            draw_text(frame, f"{done}/{len(letters)} letters at target | {status} | {quit_hint}",
                      (10, y), 0.6, (200, 255, 200))
            if message_frames > 0:
                draw_text(frame, message, (10, height - 20), 0.9, (0, 255, 255))
                message_frames -= 1

            cv2.imshow("Collect ASL letter data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC or (not q_is_letter and key in (ord("q"), ord("Q"))):
                break
            if key == 255:
                continue
            ch = chr(key).upper()
            if ch in letters:
                if points is None:
                    message = f"No hand visible: {ch} not saved"
                else:
                    append_sample(args.output, session_id, ch, handedness, width, height, points)
                    counts[ch] += 1
                    saved_now += 1
                    message = f"Saved {ch} ({counts[ch]}/{args.target})"
                message_frames = 20

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved {saved_now} samples in session {session_id}.")
    if os.path.exists(args.output):
        print("Totals in the file: "
              + "  ".join(f"{l}:{counts.get(l, 0)}" for l in letters))
        print("Next: python3 train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
