"""Train the queen-bee CNN on the mel-spectrogram PNGs in data/.

    python train.py                  # full training (defaults follow the AI-Belha card)
    python train.py --quick          # ~1 minute smoke test on a small subset
    python train.py --split hive     # harder test: evaluate on hives never seen in training
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from common import CLASS_NAMES, DATA_DIR, MODELS_DIR, build_model, load_split, make_dataset
from evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--model-name", default="bee_cnn")
    p.add_argument("--split", choices=["recording", "hive"], default="recording",
                   help="recording: random 70/15/15 split of recordings (default); "
                        "hive: test on hives 3 and 5, train on hives 1 and 4")
    p.add_argument("--epochs", type=int, default=100)
    # Batch 128 instead of the card's 32: on the Apple GPU batch 32 is slower than the CPU.
    # A 4x bigger batch at the same lr also means smoother, fewer updates (an effectively lower lr).
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="smoke test: 3000 random chunks, 2 epochs")
    p.add_argument("--cpu", action="store_true", help="ignore the GPU and train on the CPU")
    args = p.parse_args()
    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.get_visible_devices("GPU")
    print(f"Training on: {'GPU (' + gpus[0].name + ')' if gpus else 'CPU'}\n")
    keras.utils.set_random_seed(args.seed)

    df = load_split(args.data_dir, args.split, seed=args.seed)
    if args.quick:
        df = df.sample(n=3000, random_state=args.seed)
        args.epochs = 2
    train_df, val_df, test_df = (df[df.split == s] for s in ("train", "val", "test"))
    print("Chunks per class and split:")
    print(pd.crosstab(df.queen_status.map(dict(enumerate(CLASS_NAMES))), df.split, margins=True), "\n")

    out = MODELS_DIR / f"{args.model_name}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    df[["png", "recording_id", "segment", "hive", "queen_status", "split"]].to_csv(out / "split.csv", index=False)
    (out / "labels.json").write_text(json.dumps(CLASS_NAMES, indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    # "Balanced" class weights so the rarer states count as much as the common one.
    counts = np.bincount(train_df.queen_status, minlength=len(CLASS_NAMES))
    class_weight = {i: len(train_df) / (len(CLASS_NAMES) * max(int(c), 1)) for i, c in enumerate(counts)}
    print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()}, "\n")

    model = build_model()
    model.summary()
    # Macro F1 weights all 4 states equally, so an epoch that stops detecting a rare
    # state (e.g. "Queen not present") scores low even if its accuracy looks fine.
    # Checkpointing, early stopping and LR reduction all follow val_macro_f1.
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.F1Score(average="macro", name="macro_f1")])
    model.fit(
        make_dataset(train_df, args.data_dir, args.batch_size, training=True),
        validation_data=make_dataset(val_df, args.data_dir, args.batch_size),
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.ModelCheckpoint(out / "model.keras", monitor="val_macro_f1", mode="max",
                                            save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=10),
            keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5,
                                              patience=5, min_lr=1e-6),
            keras.callbacks.CSVLogger(out / "history.csv"),
            keras.callbacks.TensorBoard(out / "logs"),
        ],
    )

    model = keras.models.load_model(out / "model.keras")  # best epoch (highest val_macro_f1)
    report = evaluate(model, test_df, args.data_dir)
    (out / "test_report.txt").write_text(report)
    print(report)
    print(f"Run saved to {out}")


if __name__ == "__main__":
    main()
