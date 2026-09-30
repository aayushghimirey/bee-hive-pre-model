"""Web app: upload a beehive recording and get queen presence + hive state.

    python app.py                     # then open http://127.0.0.1:8000
    MODEL_DIR=models/bee_cnn_20260930-172124 python app.py   # pick a specific run

By default it loads the newest run in models/ that has a model.keras.
"""
import base64
import os
import threading
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from audio import CHUNK, CHUNK_SECONDS, audio_to_melspec, load_audio
from common import CLASS_NAMES, NOT_PRESENT, PROJECT_DIR, latest_model_dir

MAX_SECONDS = 600    # analyse at most the first 10 minutes
PREVIEW_CHUNKS = 12  # spectrogram preview shows the first minute

MODEL_DIR = Path(os.environ["MODEL_DIR"]) if "MODEL_DIR" in os.environ else latest_model_dir()
model = keras.models.load_model(MODEL_DIR / "model.keras")
model_lock = threading.Lock()
print(f"Loaded model from {MODEL_DIR}")

app = FastAPI(title="Queen Bee Detector")


def describe(probs):
    status = int(probs.argmax())
    return {"hive_state": CLASS_NAMES[status], "confidence": round(float(probs[status]), 4),
            "queen_present": status != NOT_PRESENT}


@app.get("/")
def index():
    return FileResponse(PROJECT_DIR / "static" / "index.html")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        y = load_audio(file.file.read(), max_seconds=MAX_SECONDS)
    except Exception:
        raise HTTPException(400, "Could not read this file. Please upload a WAV recording.")
    if len(y) < CHUNK:
        raise HTTPException(400, f"The recording is too short: at least {CHUNK_SECONDS} seconds are needed.")
    if not np.abs(y).max() > 0:
        raise HTTPException(400, "The recording is silent.")

    images = audio_to_melspec(y)
    with model_lock:
        probs = model.predict(images[..., None].astype("float32"), verbose=0)
    # The verdict averages all 5 s chunks, like the clip-level score in test_report.txt.
    mean = probs.mean(axis=0)
    preview = tf.io.encode_png(np.concatenate(list(images[:PREVIEW_CHUNKS]), axis=1)[..., None]).numpy()
    return {
        "filename": file.filename,
        "model": MODEL_DIR.name,
        "seconds_analysed": len(images) * CHUNK_SECONDS,
        **describe(mean),
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, mean)},
        "chunks": [{"start": i * CHUNK_SECONDS, **describe(p)} for i, p in enumerate(probs)],
        "spectrogram": "data:image/png;base64," + base64.b64encode(preview).decode(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
