# Setup Guide

This guide explains how to run the project locally from a clean checkout.

## 1. Clone the Repository

```bash
git clone https://github.com/hamzaf287/eeem068-human-sentiment-analysis.git
cd eeem068-human-sentiment-analysis
```

## 2. Create a Virtual Environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Add the Dataset

Metadata files should be placed in:

```text
data/raw/ende/
```

Required files:

```text
english_train.txt
english_dev.txt
english_test.txt
german_train.txt
german_dev.txt
german_test.txt
sentiment_train.txt
sentiment_dev.txt
sentiment_test.txt
image_index_train.txt
image_index_dev.txt
image_index_test.txt
```

Images should be placed in:

```text
data/raw/train_images/
data/raw/dev_images/
data/raw/test_images/
```

Image filenames should use the sample ID format:

```text
0.jpg
1.jpg
2.jpg
...
```

## 5. Verify the Data Pipeline

Run from the repository root:

macOS/Linux:

```bash
PYTHONPATH=. python3 -m src.data.test_dataset
```

Windows:

```powershell
$env:PYTHONPATH="."
python -m src.data.test_dataset
```

Expected behavior:

- Reports train/dev/test dataset sizes
- Prints class distributions
- Loads one image batch successfully

Expected dataset sizes are approximately:

```text
train: 20240
dev:   5063
test:  5067
```

## 6. Train and Evaluate Models

Run all commands from the repository root.

### Full-Image Model

macOS/Linux:

```bash
PYTHONPATH=. python3 -m src.training.train_full_image
```

Windows:

```powershell
$env:PYTHONPATH="."
python -m src.training.train_full_image
```

### Face Extraction

Generate face crops before training the face model:

macOS/Linux:

```bash
PYTHONPATH=. python3 -m src.face.face_detection
```

Windows:

```powershell
$env:PYTHONPATH="."
python -m src.face.face_detection
```

The face extraction step saves generated crops and metadata under `data/raw/`.

### Face Model

macOS/Linux:

```bash
PYTHONPATH=. python3 -m src.training.train_face_model
```

Windows:

```powershell
$env:PYTHONPATH="."
python -m src.training.train_face_model
```

### Image+Face Fusion

Export prediction probabilities:

```bash
PYTHONPATH=. python3 -m src.fusion.export_full_image_predictions
PYTHONPATH=. python3 -m src.fusion.export_face_predictions
```

Run fusion:

```bash
PYTHONPATH=. python3 -m src.fusion.fuse_predictions
```

Optional fusion weight search:

```bash
PYTHONPATH=. python3 -m src.fusion.tune_fusion_weights
```

On Windows, use `python -m ...` after setting `$env:PYTHONPATH="."`.

### Text Model

Train the frozen DistilBERT text model:

```bash
PYTHONPATH=. python3 -m src.text.train_text_model
```

On Windows:

```powershell
$env:PYTHONPATH="."
python -m src.text.train_text_model
```

### Text+Image+Face Multimodal Fusion

Run final multimodal fusion:

```bash
PYTHONPATH=. python3 -m src.fusion.fuse_text_multimodal_predictions
```

Optional text-heavy fusion weight search:

```bash
PYTHONPATH=. python3 -m src.fusion.tune_text_multimodal_fusion
```

On Windows, use `python -m ...` after setting `$env:PYTHONPATH="."`.

## 7. Generated Files

Do not commit generated files such as:

- model checkpoints
- prediction CSVs
- confusion matrix images
- classification reports
- training history plots
- face crop folders
- face metadata CSVs
- logs
- cached embeddings

These are ignored by `.gitignore`.

## 8. Troubleshooting

- `python3` not found on Windows: use `python`.
- `python` not found on macOS/Linux: use `python3`.
- `No module named 'src'`: run from the repository root and set `PYTHONPATH=.`.
- Face training says crops are missing: run `python -m src.face.face_detection` first.
- DistilBERT download fails: check internet access or whether the model is already cached locally.
- Matplotlib cache issues: set `MPLCONFIGDIR=/private/tmp` on macOS/Linux if needed.
