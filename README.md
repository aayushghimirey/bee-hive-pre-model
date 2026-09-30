# Queen Bee Detector (TensorFlow CNN)

A CNN that listens to beehive sound (as mel-spectrogram images) and predicts:

1. **Is the queen present?** yes / no
2. **Hive state**: one of the 4 queen states used by the
   [AI-Belha Classifier](https://huggingface.co/NOSInovacao/AI-Belha-Classifier)

| `queen_status` | Hive state                        | Queen present? |
|:--------------:|-----------------------------------|:--------------:|
| 0              | Queen present (original queen)    | yes            |
| 1              | Queen not present                 | **no**         |
| 2              | Queen present and rejected        | yes            |
| 3              | Queen present and newly accepted  | yes            |

One model gives both answers. It predicts the hive state, and "queen present"
follows from it, because state 1 is the only "no". In `data/splits.csv` every
row with `queen_presence = 0` has `queen_status = 1`, so a separate presence
model would learn the same thing.

## Results

Run `bee_cnn_20260930-172124`, on the test split (`--split recording`):

| | Chunk (5 s) | Clip (1 min) | AI-Belha card |
|---|---|---|---|
| Hive state accuracy | 0.898 | 0.928 | 0.73 |
| Macro F1 (all 4 states weighted equally) | 0.876 | 0.913 | 0.66 |
| Queen presence accuracy | 0.953 | 0.966 | – |
| "Queen not present" recall | 0.728 | 0.776 | 0.61 |

- **Always answering "queen present" already scores 0.865** on presence,
  so the "Queen not present" recall is the number to watch.
- **Validation scores swung a lot during training** (`val_macro_f1` between
  0.21 and 0.86). The saved model is the best epoch (epoch 3), and a retrain
  may land somewhat differently.
- **Other hives are a different story.** See
  [other hives and microphones](#important-other-hives-and-microphones).

## How this follows the AI-Belha model card

| AI-Belha card | This project | Why |
|---|---|---|
| Input: raw `.wav` → YAMNet | Input: the PNG mel-spectrograms in `data/melspec` (128 mel bins × 216 frames ≈ 5 s) | `data/` has spectrogram images, not audio, and YAMNet only accepts raw waveforms |
| YAMNet backbone (MobileNet style: 1 Conv2D + 13 depthwise-separable convs + global average pooling) | Small CNN trained from scratch, same design: 1 Conv2D (stride 2) + 4 depthwise-separable conv blocks + global average pooling | Same idea, just smaller (~184k parameters), so it trains on a laptop |
| Head: Dense 1024 (ReLU, L2 0.01) → BatchNorm → Dropout 0.3 → softmax(4) | Dense 256 (ReLU, L2 0.01) → BatchNorm → Dropout 0.3 → softmax(4) | The backbone outputs 256 features instead of 1024 |
| One audio patch = one training sample | One ~5 s spectrogram chunk = one training sample | |
| One-hot labels, categorical cross-entropy | Same | |
| Adam, lr 0.001, batch 32, ≤ 100 epochs, EarlyStopping (patience 10), ReduceLROnPlateau (patience 5), TensorBoard + CSV logs | Same, except **batch 128** | On the Apple GPU, batch 32 is slower than the CPU. With the same learning rate, a 4× bigger batch also gives smoother, fewer updates, which is effectively a lower learning rate |
| Saves `.h5` + labels `.npy` + history `.csv` | Saves `model.keras` (current Keras format) + `labels.json` + `history.csv` | |
| Random 20 % validation split | Train/val/test split **by recording** (see [Splits](#splits)) | Chunks of one recording are near duplicates, so a random split leaks test data into training |

Three small additions aimed at the card's weak recall on some classes:
- **Balanced class weights**: state 3 is half of the data, so rarer states get a higher weight.
- **SpecAugment**: during training, one random frequency band and one random time band of each spectrogram are masked out.
- **Best epoch chosen by validation macro F1** (the average F1 over the 4 states,
  shown as `val_macro_f1` during training) instead of validation loss. In an
  early test, validation accuracy still looked fine (0.71) while the model had
  almost stopped predicting "Queen not present" (recall 0.01). Macro F1 exposes
  that, because every state counts equally. Early stopping and learning-rate
  reduction follow the same score.

## Project layout

```
common.py      labels, data split, tf.data pipeline, CNN definition
train.py       trains the model, then evaluates it on the test split
evaluate.py    re-evaluates a saved model
predict.py     predicts on PNG files or folders
audio.py       audio file -> mel-spectrogram chunks in the data/melspec format
app.py         FastAPI web app: upload a recording, get a prediction
static/index.html   the web page served by app.py
data/
  melspec/     85,188 PNGs, 128 × 216 grayscale (the model input)
  splits.csv   labels for each PNG (the only label file the code reads)
  manifest.csv, hive_spectra.json, cache/   not used
models/        created by train.py, one folder per training run
```

About the data: 4 hives (1, 3, 4, 5) and 1,222 recordings from 5 June to
15 July 2022. Each recording is split into 1-minute segments, and each segment
into 12 chunks of ~5 s, so each PNG is one chunk.

| Hive state | Chunks |
|---|---:|
| Queen present (original queen) | 12,456 |
| Queen not present | 11,352 |
| Queen present and rejected | 18,636 |
| Queen present and newly accepted | 42,744 |

## Setup

```bash
cd queen-bee
python3.12 -m venv .venv          # TensorFlow does not support Python 3.14 yet
source .venv/bin/activate
pip install -r requirements.txt
```

The `.venv` is already created and installed on this Mac, so `source .venv/bin/activate` is enough.

### GPU (Apple Silicon)

Training uses the Mac's GPU automatically through Apple's `tensorflow-metal`
plugin, which `requirements.txt` installs on Apple Silicon Macs. When training
starts, the first line says which device is in use:

```
Training on: GPU (/physical_device:GPU:0)
```

- **TensorFlow is pinned to 2.18.1** because `tensorflow-metal` 1.2.0, its
  latest release, fails to load with newer TensorFlow.
- **Measured on an M5:** GPU at batch 128 ≈ 1.7 min per epoch, CPU ≈ 2.4 min
  per epoch. The model is small, so the GPU gain is modest (about 1.4×).
- **The GPU computes correctly:** predictions match the CPU to float32
  rounding, and gradients agree within about 1%.
- **If the GPU ever gives errors or NaN losses**, add `--cpu` to train on
  the CPU instead.

## 1. Smoke test (~1 minute)

```bash
python train.py --quick
```

This trains for 2 epochs on 3,000 random chunks. It only checks that everything
runs, so poor accuracy here is expected. You can delete the
`models/bee_cnn_<timestamp>` folder it creates afterwards.

## 2. Train

```bash
python train.py
```

- Speed: about **2 min per epoch** on an Apple M5 GPU (training plus
  validation). Early stopping usually ends training well before 100 epochs,
  so expect roughly 1–2 hours.
- At the end it prints the test report and saves everything to
  `models/bee_cnn_<timestamp>/`.
- Watch training live in another terminal with `tensorboard --logdir models`.

| Option | Default | Meaning |
|---|---|---|
| `--split` | `recording` | `recording` or `hive` (see [Splits](#splits)) |
| `--epochs` | 100 | Maximum epochs (early stopping usually ends sooner) |
| `--batch-size` | 128 | |
| `--lr` | 0.001 | Adam learning rate |
| `--model-name` | `bee_cnn` | Prefix of the run folder |
| `--seed` | 42 | Seed for the split and training |
| `--quick` | off | Smoke test |
| `--cpu` | off | Ignore the GPU and train on the CPU |

Output folder `models/bee_cnn_<timestamp>/`:

| File | Content |
|---|---|
| `model.keras` | The model from the epoch with the highest validation macro F1 |
| `test_report.txt` | Test metrics: per chunk and per 1-minute clip, for the hive state and for queen presence |
| `history.csv` | Loss / accuracy / macro F1 / learning rate per epoch |
| `labels.json` | Class names in output order |
| `split.csv` | Which chunks were train / val / test |
| `config.json` | Options used for this run |
| `logs/` | TensorBoard logs |

## 3. Evaluate and predict

```bash
python evaluate.py                          # latest run in models/
python evaluate.py --model-dir models/bee_cnn_20260930-120000

python predict.py data/melspec/2022-06-05--17-41-01_2__segment0__chunk00.png
python predict.py path/to/folder/           # every PNG in the folder + a combined verdict
```

Example `predict.py` output:

```
chunk00.png: queen present: YES | hive state: Queen present and newly accepted (81%)
Probabilities: Queen present (original queen) 5.1% | Queen not present 3.0% | ...
```

When you pass several files (for example the 12 chunks of one minute),
`predict.py` also averages their probabilities into one combined verdict,
which is more reliable than a single 5 s chunk.

`predict.py` expects PNGs made the same way as `data/melspec` (128 × 216
grayscale). To predict from audio, use the web app below.

## 4. Web app: upload a recording

```bash
python app.py        # then open http://127.0.0.1:8000
```

Drop a WAV file (MP3, FLAC and OGG also work) on the page and click
**Predict**. The page shows:
- **Verdict:** queen present or not, plus the hive state with its confidence.
  The verdict averages all 5 s chunks, like the clip-level score in
  `test_report.txt`.
- **Probabilities:** a bar for each of the 4 hive states.
- **Timeline:** one coloured cell per 5 s chunk, so you can see where the
  prediction changes.
- **What the model sees:** the mel spectrogram of the first minute.

Limits: recordings need at least 5 seconds, and only the first 10 minutes are
analysed. A 5-minute WAV takes about 3 seconds.

- **Which model:** by default the app loads the newest run in `models/` that
  has a `model.keras`. A later `train.py --quick` run would become "newest",
  so name the run explicitly if needed:
  `MODEL_DIR=models/bee_cnn_20260930-172124 python app.py`
- **API:** use it directly from the command line:
  `curl -F "file=@recording.wav" http://127.0.0.1:8000/predict`

### How audio becomes model input (`audio.py`)

The original preprocessing code is not in `data/`, so the recipe was
reconstructed from the PNGs and checked against real hive recordings:

| Step | Setting | Evidence |
|---|---|---|
| Load | mono, resampled to 22,050 Hz | librosa default. A 5 s chunk then gives exactly 216 frames |
| Mel spectrogram | `n_fft` 2048, hop 512, 128 mels (librosa defaults), computed per 5 s chunk | Chunk borders in `data/melspec` jump more than neighbouring columns do |
| dB → pixels | dB relative to the loudest point of the **whole recording**, 80 dB range → 0–255 | Every recording's brightest pixel is exactly 255 (25 of 25), while single chunks often aren't and no pixel is ever 0 |
| Orientation | low frequencies at the bottom, bright = loud | Row profile of real hive audio matches `data/melspec` (correlation 0.98 vs 0.90 flipped) |

### Important: other hives and microphones

The model only knows the 4 hives it was trained on. I tested it on the
labelled recordings in `~/vault/Machine Learning/dataset/`, which come from a
different beehive dataset:
- It correctly said "queen present" for all 14 queen-present files.
- It **also** said "queen present" for all 5 queen-absent files, so it
  detected none of them.
- Changing the loudness reference and dB range (9 variants) did not help,
  so the cause is the different hives and microphones, not the preprocessing.

Treat predictions on your own hives as unverified. To make the model work
there, retrain or fine-tune it with labelled recordings from those hives.

## Splits

Chunks of the same recording are almost identical, and recordings are about
one hour apart. With a random chunk split, the test set would contain pieces
of recordings the model trained on, and the accuracy would look better than
it really is. So every chunk of a recording always goes to the same split.

- **`--split recording`** (default): for each (hive, state) pair, 70 % of the
  recordings go to train, 15 % to validation and 15 % to test. All hives and
  all states appear in every split. This is the closest comparison to the
  AI-Belha card, but neighbouring recordings of the same hour can still land
  on both sides, so treat the result as optimistic.
- **`--split hive`**: train and validate on hives 1 and 4, test on hives
  3 and 5, which the model never sees during training. This is the honest
  "will it work on a new hive?" test, and accuracy is expected to be clearly
  lower.

  ```bash
  python train.py --split hive --model-name bee_cnn_hive
  ```

A caveat about this dataset: every hive went through the states in the same
order (original → removed → rejected → accepted), and hives 1 and 4 changed
state on the same days. A model can therefore partly learn "which week it
was" (weather, colony growth) instead of the queen state. The `hive` split
helps catch that, because hives 3 and 5 were recorded in different weeks
(5–14 June) than hives 1 and 4 (14 June – 15 July).

## Evaluation levels

`test_report.txt` reports every metric twice:
- **Chunk level**: one prediction per ~5 s spectrogram.
- **Clip level**: the 12 chunk probabilities of each 1-minute segment are
  averaged. This matches the 1-minute clips of the original Kaggle dataset
  and is closer to how you would use the model on a real hive.

## Credits

- Dataset: Anna Yang (2022), *Smart Bee Colony Monitor: Clips of Beehive Sounds*, Kaggle, DOI 10.34740/KAGGLE/DSV/4451415
- Approach: [AI-Belha Classifier](https://huggingface.co/NOSInovacao/AI-Belha-Classifier) (YAMNet transfer learning, MIT License)
