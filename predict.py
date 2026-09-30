"""Predict queen presence and hive state for mel-spectrogram PNGs.

    python predict.py data/melspec/2022-06-05--17-41-01_2__segment0__chunk00.png
    python predict.py some_folder/                 # every *.png in the folder
    python predict.py a.png b.png --model-dir models/bee_cnn_20260930-120000

With several files it also prints a combined verdict (mean probability),
e.g. the 12 chunks of one 1-minute recording.
"""
import argparse
from pathlib import Path

import keras
import tensorflow as tf

from common import CLASS_NAMES, NOT_PRESENT, latest_model_dir, load_png


def describe(probs):
    status = int(probs.argmax())
    present = "NO " if status == NOT_PRESENT else "YES"
    return f"queen present: {present} | hive state: {CLASS_NAMES[status]} ({probs[status]:.0%})"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="PNG files or folders of PNGs")
    p.add_argument("--model-dir", type=Path, help="run folder (default: latest in models/)")
    args = p.parse_args()

    files = []
    for path in args.paths:
        files += sorted(path.glob("*.png")) if path.is_dir() else [path]
    if not files:
        raise SystemExit("No PNG files found.")

    run = args.model_dir or latest_model_dir()
    model = keras.models.load_model(run / "model.keras")
    print(f"Model: {run}\n")

    ds = tf.data.Dataset.from_tensor_slices([str(f) for f in files]).map(load_png).batch(64)
    probs = model.predict(ds, verbose=0)
    for f, pr in zip(files, probs):
        print(f"{f.name}: {describe(pr)}")

    final = probs.mean(axis=0)
    if len(files) > 1:
        print(f"\nAll {len(files)} files combined: {describe(final)}")
    print("Probabilities: " + " | ".join(f"{name} {q:.1%}" for name, q in zip(CLASS_NAMES, final)))


if __name__ == "__main__":
    main()
