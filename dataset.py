"""Read and write the raw-landmark CSV (data/landmarks.csv).

One row per recorded frame. The landmarks are stored RAW, exactly as MediaPipe
reported them, so the normalisation in features.py can change later without
re-recording anything.

Columns: session_id, label, handedness, frame_width, frame_height,
         x0, y0, z0, ..., x20, y20, z20
"""

import csv
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np

from features import NUM_LANDMARKS

DEFAULT_CSV = os.path.join("data", "landmarks.csv")

META_COLUMNS = ["session_id", "label", "handedness", "frame_width", "frame_height"]
LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
COLUMNS = META_COLUMNS + LANDMARK_COLUMNS


class DatasetError(Exception):
    """The CSV is missing, empty or malformed."""


@dataclass
class Dataset:
    landmarks: np.ndarray      # (n, 21, 3) raw landmarks
    labels: np.ndarray         # (n,) letters
    handedness: np.ndarray     # (n,) "Left" / "Right"
    sessions: np.ndarray       # (n,) session ids
    aspect_ratios: np.ndarray  # (n,) frame width / height

    def __len__(self):
        return len(self.labels)


def append_sample(path, session_id, label, handedness, frame_width, frame_height, landmarks):
    """Append one sample, creating the file (and its folder) with a header if needed."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected shape ({NUM_LANDMARKS}, 3), got {points.shape}")

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS)
        # repr() keeps full float precision, so reading back gives identical values.
        writer.writerow(
            [session_id, label, handedness, int(frame_width), int(frame_height)]
            + [repr(float(v)) for v in points.reshape(-1)]
        )


def load_dataset(path):
    if not os.path.exists(path):
        raise DatasetError(f"no data file at {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(
                f"{path} is empty or missing columns (first missing: {missing[0]})"
            )
        rows = list(reader)

    if not rows:
        raise DatasetError(f"{path} has a header but no samples")

    try:
        landmarks = np.array(
            [[float(r[c]) for c in LANDMARK_COLUMNS] for r in rows]
        ).reshape(len(rows), NUM_LANDMARKS, 3)
        widths = np.array([float(r["frame_width"]) for r in rows])
        heights = np.array([float(r["frame_height"]) for r in rows])
    except (TypeError, ValueError) as e:
        raise DatasetError(f"{path} has a row that is not numeric: {e}") from e

    aspect = np.where((widths > 0) & (heights > 0), widths / np.maximum(heights, 1), 1.0)

    return Dataset(
        landmarks=landmarks,
        labels=np.array([r["label"] for r in rows]),
        handedness=np.array([r["handedness"] for r in rows]),
        sessions=np.array([r["session_id"] for r in rows]),
        aspect_ratios=aspect,
    )


def count_labels(path):
    """Samples per label already in the CSV (empty Counter if there is no file)."""
    if not os.path.exists(path):
        return Counter()
    with open(path, newline="") as f:
        return Counter(r["label"] for r in csv.DictReader(f) if r.get("label"))
