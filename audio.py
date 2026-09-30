"""Turn a recording into model input: the same mel-spectrogram format as data/melspec.

The original preprocessing code is not in data/, so this recipe was reconstructed
from the PNGs themselves:
  * librosa defaults: 22,050 Hz mono, n_fft 2048, hop 512, 128 mel bins,
    which gives exactly 216 frames per 5 s chunk
  * every 5 s chunk gets its own spectrogram
  * dB relative to the loudest point of the whole recording, 80 dB range -> pixels 0..255
    (in data/ every recording's brightest pixel is exactly 255, and no pixel is ever 0)
  * low frequencies at the bottom of the image, bright = loud
"""
import io

import librosa
import numpy as np

SR = 22050
CHUNK_SECONDS = 5
CHUNK = SR * CHUNK_SECONDS
DB_RANGE = 80.0


def load_audio(data, max_seconds=None):
    """Audio file bytes (WAV, FLAC, OGG, MP3) -> mono float32 samples at 22,050 Hz."""
    y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True, duration=max_seconds)
    return y


def audio_to_melspec(y):
    """Mono 22,050 Hz audio -> uint8 array (n_chunks, 128, 216) like the PNGs in data/melspec.

    Audio after the last full 5 s chunk is dropped.
    """
    n = len(y) // CHUNK
    power = np.stack([librosa.feature.melspectrogram(y=y[i * CHUNK:(i + 1) * CHUNK], sr=SR)
                      for i in range(n)])
    db = librosa.power_to_db(power, ref=power.max(), top_db=None)
    img = np.clip(255 * (db + DB_RANGE) / DB_RANGE, 0, 255)
    return img[:, ::-1, :].astype(np.uint8)  # flip so low frequencies are at the bottom
