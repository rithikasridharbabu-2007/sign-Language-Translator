import numpy as np
import pytest

from dataset import COLUMNS, DatasetError, append_sample, count_labels, load_dataset


def test_csv_round_trip(tmp_path):
    path = str(tmp_path / "sub" / "landmarks.csv")  # folder is created on demand
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(21, 3))
    b = rng.uniform(-1, 1, size=(21, 3))

    append_sample(path, "20260101-120000", "A", "Right", 1280, 720, a)
    append_sample(path, "20260102-120000", "B", "Left", 640, 480, b)

    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0].split(",") == COLUMNS
    assert len(lines) == 3  # one header, two samples

    data = load_dataset(path)
    assert len(data) == 2
    np.testing.assert_array_equal(data.landmarks[0], a)  # exact, not approximate
    np.testing.assert_array_equal(data.landmarks[1], b)
    assert data.labels.tolist() == ["A", "B"]
    assert data.handedness.tolist() == ["Right", "Left"]
    assert data.sessions.tolist() == ["20260101-120000", "20260102-120000"]
    np.testing.assert_allclose(data.aspect_ratios, [1280 / 720, 640 / 480])
    assert count_labels(path) == {"A": 1, "B": 1}


def test_append_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        append_sample(str(tmp_path / "x.csv"), "s", "A", "Right", 1, 1, np.zeros((5, 3)))


def test_missing_and_empty_files(tmp_path):
    missing = str(tmp_path / "nope.csv")
    assert count_labels(missing) == {}
    with pytest.raises(DatasetError, match="no data file"):
        load_dataset(missing)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DatasetError):
        load_dataset(str(empty))

    header_only = tmp_path / "header.csv"
    header_only.write_text(",".join(COLUMNS) + "\n")
    with pytest.raises(DatasetError, match="no samples"):
        load_dataset(str(header_only))
