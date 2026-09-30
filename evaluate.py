"""Evaluate a trained model on its test split, per chunk and per 1-minute clip.

    python evaluate.py                                   # latest run in models/
    python evaluate.py --model-dir models/bee_cnn_20260930-120000
"""
import argparse
from pathlib import Path

import keras
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import CLASS_NAMES, DATA_DIR, NOT_PRESENT, latest_model_dir, make_dataset


def _report(title, y_true, probs):
    y_pred = probs.argmax(axis=1)
    present_true, present_pred = y_true != NOT_PRESENT, y_pred != NOT_PRESENT
    labels = list(range(len(CLASS_NAMES)))
    return "\n".join([
        f"===== {title}: {len(y_true)} samples =====",
        f"Hive state (4 classes) accuracy: {accuracy_score(y_true, y_pred):.3f}",
        classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                              digits=3, zero_division=0),
        "Confusion matrix (rows = true, columns = predicted, class order as above):",
        str(confusion_matrix(y_true, y_pred, labels=labels)),
        "",
        f"Queen presence (2 classes) accuracy: {accuracy_score(present_true, present_pred):.3f}",
        classification_report(present_true, present_pred, labels=[False, True],
                              target_names=["Queen not present", "Queen present"],
                              digits=3, zero_division=0),
        "Confusion matrix (rows = true, columns = predicted): [not present, present]",
        str(confusion_matrix(present_true, present_pred, labels=[False, True])),
        "",
    ])


def evaluate(model, test_df, data_dir=DATA_DIR, batch_size=64):
    """Return a text report for the rows of test_df."""
    probs = model.predict(make_dataset(test_df, data_dir, batch_size), verbose=1)
    # One original Kaggle clip = one 1-minute segment = 12 chunks: average their probabilities.
    cols = [f"p{i}" for i in range(probs.shape[1])]
    clips = test_df.assign(**dict(zip(cols, probs.T))).groupby(["recording_id", "segment"])
    return "\n".join([
        _report("Chunk level (one ~5 s spectrogram)", test_df.queen_status.to_numpy(), probs),
        _report("Clip level (1-minute segment, mean of its chunks)",
                clips.queen_status.first().to_numpy(), clips[cols].mean().to_numpy()),
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, help="run folder (default: latest in models/)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    run = args.model_dir or latest_model_dir()
    model = keras.models.load_model(run / "model.keras")
    split = pd.read_csv(run / "split.csv")
    report = evaluate(model, split[split.split == "test"], args.data_dir)
    (run / "test_report.txt").write_text(report)
    print(report)
    print(f"Report saved to {run / 'test_report.txt'}")


if __name__ == "__main__":
    main()
