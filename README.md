# EEEM068 Human Sentiment Analysis

Group coursework project for **EEEM068 Applied Machine Learning**.

This repository implements sentiment classification for the **MSCTD English-German (En-De) subset**. The final system includes image, face, text, and late-fusion pipelines for three sentiment classes:

- `0`: neutral
- `1`: negative
- `2`: positive

## Overview

The project uses one aligned sample per image/text/sentiment line:

```text
image_id == text line index == sentiment line index == image filename
```

For example, sample `42` uses:

```text
data/raw/train_images/42.jpg
data/raw/ende/english_train.txt[42]
data/raw/ende/german_train.txt[42]
data/raw/ende/sentiment_train.txt[42]
```

Implemented components:

- Data pipeline for MSCTD En-De image/text/sentiment alignment
- Full-image sentiment model
- Face extraction and face-based sentiment model
- Face-count-aware image+face fusion
- Frozen DistilBERT English text sentiment model
- Text+image+face multimodal late fusion

## Dataset

Expected metadata files:

```text
data/raw/ende/
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

Expected image folders:

```text
data/raw/train_images/
data/raw/dev_images/
data/raw/test_images/
```

Face crops and metadata are generated from the raw images:

```text
data/raw/face_train_images/
data/raw/face_dev_images/
data/raw/face_test_images/
data/raw/face_metadata_train.csv
data/raw/face_metadata_dev.csv
data/raw/face_metadata_test.csv
```

Generated face files are ignored by git.

## Project Structure

```text
src/
  data/
    dataset.py
    dataloaders.py
    transforms.py
    test_dataset.py
    visualize_distribution.py
    visualize_samples.py
    visualize_transforms.py

  models/
    full_image_model.py
    face_model.py

  training/
    train_full_image.py
    train_face_model.py

  face/
    face_detection.py
    face_dataset.py

  fusion/
    export_full_image_predictions.py
    export_face_predictions.py
    fuse_predictions.py
    tune_fusion_weights.py
    fuse_text_multimodal_predictions.py
    tune_text_multimodal_fusion.py

  text/
    train_text_model.py
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, use:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run commands from the repository root. If needed, set `PYTHONPATH=.` before module commands.

## Commands

Check the shared dataset pipeline:

```bash
PYTHONPATH=. python -m src.data.test_dataset
```

Train the full-image model:

```bash
PYTHONPATH=. python -m src.training.train_full_image
```

Generate face crops:

```bash
PYTHONPATH=. python -m src.face.face_detection
```

Train the face model:

```bash
PYTHONPATH=. python -m src.training.train_face_model
```

Export predictions for image+face fusion:

```bash
PYTHONPATH=. python -m src.fusion.export_full_image_predictions
PYTHONPATH=. python -m src.fusion.export_face_predictions
```

Run image+face fusion:

```bash
PYTHONPATH=. python -m src.fusion.fuse_predictions
```

Tune image+face fusion weights:

```bash
PYTHONPATH=. python -m src.fusion.tune_fusion_weights
```

Train the frozen DistilBERT text model:

```bash
PYTHONPATH=. python -m src.text.train_text_model
```

Run text+image+face multimodal fusion:

```bash
PYTHONPATH=. python -m src.fusion.fuse_text_multimodal_predictions
```

Tune text+image+face multimodal fusion weights:

```bash
PYTHONPATH=. python -m src.fusion.tune_text_multimodal_fusion
```

## Final Results

| System | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Full-image | 0.3517 | 0.3392 |
| Face | 0.3560 | 0.3304 |
| Image+Face Fusion | 0.3661 | 0.3458 |
| Text | 0.5927 | 0.5826 |
| Text+Image+Face Fusion | 0.5988 | 0.5872 |

## Generated Outputs

Model checkpoints, prediction CSVs, confusion matrices, classification reports, training plots, logs, cached embeddings, raw images, generated face crops, and face metadata are ignored by git.

Important generated examples:

```text
*.pth
*.joblib
*_predictions.csv
*_classification_report.txt
*_confusion_matrix.png
*_training_history.png
outputs/
data/raw/face_*_images/
data/raw/face_metadata_*.csv
```
