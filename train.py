"""Train the static-letter classifier from data/landmarks.csv.

    python3 train.py                          # random 80/20 split of all frames
    python3 train.py --holdout-session last   # test on a whole recording session

Writes model.joblib, results.json and confusion_matrix.png.
No camera, OpenCV or MediaPipe needed.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from dataset import DEFAULT_CSV, DatasetError, load_dataset
from features import FEATURE_VERSION, batch_to_features

RANDOM_SPLIT_WARNING = (
    "The test set is a RANDOM split of frames. Frames recorded a fraction of a "
    "second apart are almost identical, so near-copies of every test frame are "
    "in the training set. This accuracy OVER-ESTIMATES how well the model works "
    "on a new day, new lighting or a new person. For an honest number, record a "
    "second session (run collect_data.py again) and train with "
    "--holdout-session last."
)
ONE_SIGNER_NOTE = (
    "All data comes from whoever recorded it. Accuracy for other signers is not "
    "measured by this script."
)


class TrainError(Exception):
    """A problem the user can fix (no data, too few samples, bad option)."""


def build_candidates(seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed),
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
    }


def resolve_holdout(requested, sessions):
    """Turn the --holdout-session values into a set of session ids that exist."""
    available = sorted(set(sessions))
    chosen = set()
    for name in requested:
        if name == "last":
            chosen.add(available[-1])  # ids are timestamps, so sorted = chronological
        elif name in available:
            chosen.add(name)
        else:
            raise TrainError(
                f"session '{name}' is not in the data. Sessions available: {', '.join(available)}"
            )
    if len(chosen) >= len(available):
        raise TrainError(
            "holding out every session leaves nothing to train on. The data has "
            f"{len(available)} session(s): {', '.join(available)}. Record another "
            "session with collect_data.py, or train without --holdout-session."
        )
    return chosen


def check_class_counts(labels, min_per_class, what):
    counts = Counter(labels)
    if len(counts) < 2:
        raise TrainError(
            f"need at least 2 different letters to train, found {len(counts)} in {what}."
        )
    short = {k: v for k, v in sorted(counts.items()) if v < min_per_class}
    if short:
        detail = ", ".join(f"{k}: {v}" for k, v in short.items())
        raise TrainError(
            f"too few samples in {what} (need at least {min_per_class} per letter): {detail}. "
            "Record more with collect_data.py."
        )
    return counts


def save_confusion_png(matrix, labels, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * n + 2), max(4.5, 0.6 * n + 1.5)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n), labels=labels)
    ax.set_yticks(range(n), labels=labels)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title(title, fontsize=10)
    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, label="Test frames")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_confusion(matrix, labels):
    width = max(4, len(str(matrix.max())) + 1)
    lines = ["true\\pred" + "".join(f"{l:>{width}}" for l in labels)]
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>9}" + "".join(f"{v:>{width}}" for v in row))
    return "\n".join(lines)


def loud(text):
    bar = "!" * 72
    print(f"\n{bar}\nWARNING: {text}\n{bar}\n")


def train(data_path=DEFAULT_CSV, out_dir=".", test_size=0.2, seed=42,
          holdout_sessions=None, min_per_class=20):
    """Run the whole pipeline and return the results dict that is also saved."""
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    try:
        data = load_dataset(data_path)
    except DatasetError as e:
        raise TrainError(
            f"{e}. Record some first: python3 collect_data.py"
        ) from e

    check_class_counts(data.labels, min_per_class, data_path)
    X = batch_to_features(data.landmarks, data.handedness, data.aspect_ratios)
    y = data.labels
    all_sessions = sorted(set(data.sessions))
    warnings = []

    if holdout_sessions:
        held = resolve_holdout(holdout_sessions, data.sessions)
        test_mask = np.isin(data.sessions, sorted(held))
        X_train, y_train, X_test, y_test = X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]
        check_class_counts(y_train, min_per_class, "the training sessions")
        unseen = sorted(set(y_test) - set(y_train))
        if unseen:
            raise TrainError(
                f"held-out session has letters that are not in the training sessions: {', '.join(unseen)}"
            )
        untested = sorted(set(y_train) - set(y_test))
        if untested:
            warnings.append(
                "The held-out session has no samples for: " + ", ".join(untested)
                + ". Those letters are not tested."
            )
        split = {"type": "holdout_session", "holdout_sessions": sorted(held)}
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        warnings.append(RANDOM_SPLIT_WARNING)
        split = {"type": "random_frames", "test_size": test_size}
    warnings.append(ONE_SIGNER_NOTE)

    labels = sorted(set(y_train))

    # Compare the two models by cross-validation on the training split only,
    # so the test set plays no part in choosing between them.
    folds = min(5, min(Counter(y_train).values()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    candidates = build_candidates(seed)
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv)
        cv_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"cross-validation ({folds} folds) {name:>14}: "
              f"{scores.mean():.4f} +/- {scores.std():.4f}")
    chosen = max(cv_scores, key=lambda name: cv_scores[name]["mean"])
    print(f"chosen model: {chosen}")

    model = candidates[chosen].fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predicted))
    matrix = confusion_matrix(y_test, predicted, labels=labels)
    per_letter = {}
    for label, row, correct in zip(labels, matrix, matrix.diagonal()):
        per_letter[label] = {
            "accuracy": float(correct / row.sum()) if row.sum() else None,
            "test_samples": int(row.sum()),
        }

    split_name = ("held-out session " + ", ".join(split["holdout_sessions"])
                  if holdout_sessions else "random frame split (optimistic)")
    print(f"\ntest accuracy ({split_name}): {accuracy:.4f} on {len(y_test)} frames")
    print("\nper-letter accuracy:")
    for label, info in per_letter.items():
        shown = "  n/a" if info["accuracy"] is None else f"{info['accuracy']:.4f}"
        print(f"  {label}: {shown}  ({info['test_samples']} test frames)")
    print("\nconfusion matrix (rows = true letter, columns = predicted):")
    print(format_confusion(matrix, labels))

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    results_path = os.path.join(out_dir, "results.json")
    png_path = os.path.join(out_dir, "confusion_matrix.png")

    joblib.dump(
        {"model": model, "labels": list(model.classes_), "model_name": chosen,
         "feature_version": FEATURE_VERSION, "sklearn_version": sklearn.__version__},
        model_path,
    )
    results = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_path": data_path,
        "letters": labels,
        "samples_per_letter": dict(sorted(Counter(y.tolist()).items())),
        "sessions": all_sessions,
        "split": split,
        "seed": seed,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "cross_validation": {"folds": folds, "scores": cv_scores},
        "chosen_model": chosen,
        "test_accuracy": accuracy,
        "per_letter": per_letter,
        "confusion_matrix": {"labels": labels, "rows_true_cols_predicted": matrix.tolist()},
        "feature_version": FEATURE_VERSION,
        "sklearn_version": sklearn.__version__,
        "warnings": warnings,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    save_confusion_png(matrix, labels, png_path,
                       f"{chosen}, {split_name}\naccuracy {accuracy:.3f} on {len(y_test)} frames")
    print(f"\nsaved {model_path}, {results_path}, {png_path}")

    for text in warnings:
        loud(text)
    if not holdout_sessions and len(all_sessions) > 1:
        print(f"This data has {len(all_sessions)} sessions; for a more honest number run:\n"
              "  python3 train.py --holdout-session last\n")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the ASL static-letter classifier from recorded landmarks."
    )
    parser.add_argument("--data", default=DEFAULT_CSV, help=f"landmark CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--out-dir", default=".",
                        help="where to write model.joblib, results.json and confusion_matrix.png (default: .)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction of frames for the random test split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--holdout-session", nargs="+", metavar="ID",
                        help="test on these whole recording sessions instead of a random split; "
                             "'last' means the most recent session")
    parser.add_argument("--min-per-class", type=int, default=20,
                        help="refuse to train if any letter has fewer samples (default: 20)")
    args = parser.parse_args(argv)

    try:
        train(args.data, args.out_dir, args.test_size, args.seed,
              args.holdout_session, args.min_per_class)
    except TrainError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
