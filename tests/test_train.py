"""End-to-end tests of train.py on SYNTHETIC landmarks.

The data is random, well-separated fake classes written to a pytest temp dir.
The accuracies here say nothing about real sign recognition.
"""

import json
import os

import joblib
import numpy as np
import pytest

import train as train_module
from features import FEATURE_VERSION
from predict import LetterPredictor, load_model
from tests.conftest import fake_class_centres


def test_end_to_end_random_split(synthetic_csv, tmp_path, capsys):
    out = tmp_path / "out"
    code = train_module.main(["--data", synthetic_csv(), "--out-dir", str(out)])
    assert code == 0

    for name in ("model.joblib", "results.json", "confusion_matrix.png"):
        assert (out / name).stat().st_size > 0

    results = json.loads((out / "results.json").read_text())
    assert results["letters"] == ["A", "B", "C", "D"]
    assert results["split"]["type"] == "random_frames"
    assert results["train_samples"] + results["test_samples"] == 4 * 40 * 2
    assert set(results["cross_validation"]["scores"]) == {"mlp", "random_forest"}
    assert results["chosen_model"] in ("mlp", "random_forest")
    assert results["test_accuracy"] > 0.95  # fake classes are trivially separable
    assert set(results["per_letter"]) == set("ABCD")
    matrix = np.array(results["confusion_matrix"]["rows_true_cols_predicted"])
    assert matrix.shape == (4, 4) and matrix.sum() == results["test_samples"]
    assert any("OVER-ESTIMATES" in w for w in results["warnings"])

    printed = capsys.readouterr().out
    assert "WARNING" in printed and "OVER-ESTIMATES" in printed
    assert "confusion matrix" in printed and "per-letter accuracy" in printed

    bundle = joblib.load(out / "model.joblib")
    assert bundle["feature_version"] == FEATURE_VERSION
    assert list(bundle["labels"]) == ["A", "B", "C", "D"]


def test_same_seed_same_result(synthetic_csv, tmp_path):
    path = synthetic_csv()
    a = train_module.train(path, str(tmp_path / "a"))
    b = train_module.train(path, str(tmp_path / "b"))
    assert a["confusion_matrix"] == b["confusion_matrix"]
    assert a["cross_validation"] == b["cross_validation"]


def test_holdout_session(synthetic_csv, tmp_path):
    results = train_module.train(synthetic_csv(), str(tmp_path / "out"), holdout_sessions=["last"])
    assert results["split"] == {"type": "holdout_session", "holdout_sessions": ["s2"]}
    assert results["train_samples"] == results["test_samples"] == 4 * 40
    assert not any("OVER-ESTIMATES" in w for w in results["warnings"])


def test_holdout_errors(synthetic_csv, tmp_path):
    out = str(tmp_path / "out")
    with pytest.raises(train_module.TrainError, match="not in the data"):
        train_module.train(synthetic_csv(), out, holdout_sessions=["nope"])
    one_session = synthetic_csv(sessions=("only",), name="one.csv")
    with pytest.raises(train_module.TrainError, match="nothing to train on"):
        train_module.train(one_session, out, holdout_sessions=["last"])
    assert not os.path.exists(out)


def test_no_data_fails_clearly(tmp_path, capsys):
    out = tmp_path / "out"
    code = train_module.main(["--data", str(tmp_path / "missing.csv"), "--out-dir", str(out)])
    assert code == 1
    err = capsys.readouterr().err
    assert "no data file" in err and "collect_data.py" in err
    assert not out.exists()


def test_too_few_samples_fails_clearly(synthetic_csv, tmp_path, capsys):
    out = tmp_path / "out"
    path = synthetic_csv(per_class=3, sessions=("s1",))
    code = train_module.main(["--data", path, "--out-dir", str(out)])
    assert code == 1
    err = capsys.readouterr().err
    assert "too few samples" in err and "A: 3" in err
    assert not out.exists()


def test_single_letter_fails_clearly(synthetic_csv, tmp_path):
    with pytest.raises(train_module.TrainError, match="at least 2 different letters"):
        train_module.train(synthetic_csv(letters="A"), str(tmp_path / "out"))


def test_trained_model_drives_predictor(synthetic_csv, tmp_path):
    out = tmp_path / "out"
    train_module.train(synthetic_csv(), str(out))
    bundle = load_model(str(out / "model.joblib"))
    predictor = LetterPredictor(bundle["model"], bundle["labels"], window=5, threshold=0.6)

    centre_b = fake_class_centres(4)[1]
    for _ in range(5):
        letter, confidence = predictor.update(centre_b, "Right", 1280 / 720)
    assert letter == "B" and confidence >= 0.6

    assert predictor.update(None, None) == (None, 0.0)  # hand left the frame
    assert len(predictor.history) == 0


def test_load_model_missing_and_stale(tmp_path):
    assert load_model(str(tmp_path / "model.joblib")) is None
    stale = tmp_path / "stale.joblib"
    joblib.dump({"model": None, "labels": [], "feature_version": -1}, stale)
    with pytest.raises(RuntimeError, match="train.py"):
        load_model(str(stale))

