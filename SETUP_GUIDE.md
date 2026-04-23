# Setup Guide (For Group Members)

This guide helps you run the project locally and avoid common setup issues.

## 1. Clone the repository

```bash
git clone https://github.com/hamzaf287/eeem068-human-sentiment-analysis.git
cd eeem068-human-sentiment-analysis
```

## 2. Create and activate a virtual environment

Use a virtual environment so everyone uses isolated dependencies.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```bat
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Place the dataset correctly

### Metadata files

Put all metadata text files in:

`data/raw/ende/`

Required patterns:

- `english_*.txt`
- `german_*.txt`
- `image_index_*.txt`
- `sentiment_*.txt`

### Image files

Put images into:

- `data/raw/train_images/`
- `data/raw/dev_images/`
- `data/raw/test_images/`

Expected naming format:

- `0.jpg`, `1.jpg`, `2.jpg`, ...

## 5. Verify your setup

Run:

```bash
python src/data_check.py
```

Expected behavior:

- Prints sample indices
- Prints sentiment label
- Confirms referenced image file exists

If this script runs without errors, your local setup is ready.

## 6. Daily workflow

1. Activate environment:

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bat
   venv\Scripts\activate
   ```

2. Pull latest changes:

   ```bash
   git pull
   ```

3. Do your work.
4. Commit and push:

   ```bash
   git add .
   git commit -m "meaningful message"
   git push
   ```

## 7. Team rules

- Do not upload dataset images to GitHub.
- Always work inside the virtual environment.
- Use clear and meaningful commit messages.

## 8. Quick troubleshooting

- `python` command not found: try `python3`.
- Dependency install fails: upgrade pip first with `python -m pip install --upgrade pip`.
- `data_check.py` fails: re-check file paths and image naming format.

## 9. Executing the train_full_image, train_face_model (MAC)

- ' source venv/bin/activate
  (venv) (base) syedmubeen@syeds-MacBook-Air-2 eeem068-human-sentiment-analysis % python -m src.training.train_full_image'

- ' python -m src.training.train_face_model '
