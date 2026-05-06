# EEEM068 Human Sentiment Analysis

Group project for **EEEM068 Applied Machine Learning**.

This repository implements sentiment classification models using the **English–German (En–De) portion of the MSCTD dataset**.  
The focus is on image-based sentiment analysis using full images and associated metadata.


## Project goal
Build sentiment classification models using:
- face-based analysis
- full-image analysis
- fusion of both

## Current progress
- Repository created
- Folder structure set up
- Dataset metadata organised
- Initial data check script added
- Codebase reviewed and tested against coursework requirements (by Jyotsana Singh Rajawat)
- Experimental runs and technical report writing in progress (by Jyotsana Singh Rajawat)

## 1. Dataset

We use the **MSCTD** (Multi‑Sentiment Captioned Twitter Dataset), restricted to the **English–German (En–De)** portion.

The raw data files are stored under:

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

## 2. Repository Structure

data/
  raw/
    ende/                    # MSCTD En–De raw text and indices

src/
  data/
    dataset.py               # Dataset utilities / helpers
    full_image_dataset.py    # Dataset class for full‑image sentiment data
    test_full_image_dataset.py  # Unit tests for the full‑image dataset
    transforms.py            # Image transforms and augmentations

  models/
    full_image_model.py      # Full‑image sentiment model definition

  training/
    train_full_image.py      # Main training script for full‑image models

.gitignore
README.md
SETUP_GUIDE.md
requirements.txt





