# Real-Time Age Classification with ResNet18 (Transfer Learning)

## Overview

This project uses a pre-trained **ResNet18** model for webcam-based **age-group classification** with transfer learning.
Each class represents a fixed age span (default: **3 years**).

## Setup

Install project dependencies:

```bash
pip install -r requirements.txt
```

## 1) Kaggle setup and UTKFace installation (full step-by-step)

### Step 1 — Install Kaggle CLI (one-time)

```bash
pip install kaggle
```

### Step 2 — Create Kaggle API credentials

1. Go to Kaggle → **Account**.
2. Click **Create New API Token**.
3. Save the downloaded `kaggle.json`.

Place the file at:

- Linux/macOS: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%/.kaggle/kaggle.json`

Set secure permissions on Linux/macOS:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3 — Download + prepare UTKFace with this project script

```bash
python src/download_utkface.py --source kaggle --output-dir datasets/UTKFace
```

Optional flags:

- Clean existing output before re-preparing:
  ```bash
  python src/download_utkface.py --source kaggle --output-dir datasets/UTKFace --clean
  ```
- Use a different temporary work directory:
  ```bash
  python src/download_utkface.py --source kaggle --work-dir datasets/_downloads
  ```
- Use a different Kaggle dataset slug:
  ```bash
  python src/download_utkface.py --source kaggle --kaggle-dataset owner/dataset-name
  ```

Alternative download source (without Kaggle CLI):

```bash
python src/download_utkface.py --source url --url <direct_archive_url> --output-dir datasets/UTKFace
```

> Downloaded datasets/archives are ignored by git via `.gitignore` (`datasets/`, archive extensions).

### Step 4 — Verify dataset structure

The output directory should contain files named like:

`25_0_2_20170116174525125.jpg`

Quick check:

```bash
ls datasets/UTKFace | head
```

## 2) Train the model (Transfer Learning on ResNet18)

Run training on the prepared UTKFace folder:

```bash
python src/train.py --utkface-root datasets/UTKFace --min-age 0 --max-age 80 --age-bin-size 3 --epochs 10
```

This initializes ResNet18 with pre-trained weights and fine-tunes the classifier head for configured age bins.

During training, the script now computes and logs validation metrics every epoch, then evaluates both validation and test splits at the end. It writes:

- `artifacts/metrics/training_history.csv` (table with epoch-by-epoch train/validation metrics)
- `artifacts/metrics/training_curves.png` (loss, validation accuracy, and validation F1 plots)
- `artifacts/metrics/test_confusion_matrix.png` (normalized test confusion matrix heatmap)

No retraining is required if you already have a checkpoint and only want metrics. Run evaluation-only mode:

```bash
python src/train.py \
  --utkface-root datasets/UTKFace \
  --model-path model_state.pth \
  --eval-only
```

In `--eval-only` mode, the script loads `model_state.pth`, computes validation/test metrics, prints metric tables, and writes artifacts to `artifacts/metrics` (or your custom `--metrics-dir`).

If you want a prediction to count as correct when it lands in a neighboring bin (for 3-year bins, e.g., predicted `21-23` and truth is `18-20` or `24-26`), use:

```bash
python src/train.py \
  --utkface-root datasets/UTKFace \
  --model-path model_state.pth \
  --eval-only \
  --tolerance-bins 1
```

`--tolerance-bins 1` means a 3-bin acceptance span (left neighbor, exact bin, right neighbor). Set `--tolerance-bins 0` for strict exact-bin metrics.

> Note: if you configured a different `--age-bin-size`, the tolerance still applies in **bin units** (not years).  
> Example: `--age-bin-size 5 --tolerance-bins 1` accepts ±1 neighboring 5-year bin.

You can customize split/output settings:

```bash
python src/train.py \
  --utkface-root datasets/UTKFace \
  --val-split 0.15 \
  --test-split 0.15 \
  --metrics-dir artifacts/metrics
```

## 3) Webcam inference

```bash
python src/webcam.py --model-path model_state.pth
```

During webcam mode, press `+` to zoom in and `-` to zoom out.

If `model_state.pth` is missing, you can auto-train first:

```bash
python src/webcam.py --train-if-missing --utkface-root datasets/UTKFace --min-age 0 --max-age 80 --age-bin-size 3
```

The webcam mode is inference-only.

## 4) Single-image inference (file picker)

```bash
python src/predict_image.py --model-path model_state.pth
```

The script prompts you to select an image file and prints the predicted age range.
