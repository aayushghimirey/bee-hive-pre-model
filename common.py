"""Shared code: class labels, train/val/test split, tf.data pipeline and the CNN."""
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, regularizers

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

IMG_HEIGHT, IMG_WIDTH = 128, 216  # 128 mel bins x 216 time frames (one ~5 s chunk)

# queen_status codes in data/splits.csv (same 4 classes as the AI-Belha model card)
CLASS_NAMES = [
    "Queen present (original queen)",    # 0
    "Queen not present",                 # 1
    "Queen present and rejected",        # 2
    "Queen present and newly accepted",  # 3
]
NOT_PRESENT = 1  # the only status with queen_presence == 0
UNSEEN_HIVES = [3, 5]  # test hives for --split hive (cohort A in splits.csv)


def load_split(data_dir=DATA_DIR, mode="recording", val_frac=0.15, test_frac=0.15, seed=42):
    """Read splits.csv and add a 'split' column (train / val / test).

    All chunks of one recording go to the same split, so near-identical 5 s
    chunks of the same recording never end up in both train and test.
      mode="recording": 70/15/15 of the recordings of every (hive, status) pair.
      mode="hive":      test = hives 3 and 5 (never seen in training),
                        train/val = hives 1 and 4.
    """
    df = pd.read_csv(Path(data_dir) / "splits.csv")
    recs = df.drop_duplicates("recording_id")
    split = {}
    if mode == "hive":
        split.update({rid: "test" for rid in recs.loc[recs.hive.isin(UNSEEN_HIVES), "recording_id"]})
        recs = recs[~recs.hive.isin(UNSEEN_HIVES)]
        test_frac = 0.0
    rng = np.random.default_rng(seed)
    for _, group in recs.groupby(["hive", "queen_status"]):
        ids = rng.permutation(group.recording_id.to_numpy())
        n_val, n_test = round(len(ids) * val_frac), round(len(ids) * test_frac)
        for i, rid in enumerate(ids):
            split[rid] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    df["split"] = df.recording_id.map(split)
    return df


def load_png(path):
    """PNG file -> float32 tensor (128, 216, 1) with values 0..255."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=1)
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, 1))
    return tf.cast(img, tf.float32)


def _mask_band(x, axis, max_width):
    """Replace a random band along `axis` (0 = frequency, 1 = time) with the image mean."""
    size = tf.shape(x)[axis]
    width = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32)
    start = tf.random.uniform([], 0, size - width + 1, dtype=tf.int32)
    idx = tf.range(size)
    keep = (idx < start) | (idx >= start + width)
    keep = tf.reshape(keep, [-1, 1, 1] if axis == 0 else [1, -1, 1])
    return tf.where(keep, x, tf.reduce_mean(x))


def spec_augment(x):
    """SpecAugment-style masking of one frequency band and one time band (training only)."""
    return _mask_band(_mask_band(x, axis=0, max_width=16), axis=1, max_width=24)


def make_dataset(df, data_dir=DATA_DIR, batch_size=32, training=False):
    """tf.data pipeline yielding (spectrogram batch, one-hot queen_status batch)."""
    paths = [str(Path(data_dir) / "melspec" / p) for p in df.png]
    ds = tf.data.Dataset.from_tensor_slices((paths, df.queen_status.to_numpy()))
    if training:
        ds = ds.shuffle(len(paths))
    ds = ds.map(lambda p, y: (load_png(p), tf.one_hot(y, len(CLASS_NAMES))),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (spec_augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def conv_block(x, filters, separable=True, strides=1):
    conv = layers.SeparableConv2D if separable else layers.Conv2D
    x = conv(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.MaxPooling2D(2)(x)


def build_model(num_classes=len(CLASS_NAMES), dropout=0.3, l2=0.01):
    """Small YAMNet/MobileNet-style CNN.

    Backbone: one Conv2D block, then depthwise-separable conv blocks and global
    average pooling. Head as in the AI-Belha card: Dense (ReLU, L2) -> BatchNorm
    -> Dropout -> softmax over the 4 queen states.
    """
    inputs = keras.Input((IMG_HEIGHT, IMG_WIDTH, 1), name="melspec")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = conv_block(x, 32, separable=False, strides=2)  # stride 2 like YAMNet: ~5x faster training
    for filters in (64, 128, 256, 256):
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="queen_status")(x)
    return keras.Model(inputs, outputs, name="bee_cnn")


def latest_model_dir():
    runs = sorted(MODELS_DIR.glob("*/model.keras"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No trained model found in models/. Run train.py first.")
    return runs[-1].parent
