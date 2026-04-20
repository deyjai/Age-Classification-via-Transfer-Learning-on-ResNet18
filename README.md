# Real-Time Age Classification with ResNet18 (Transfer Learning)

## Overview

This project uses a pre-trained **ResNet18** model for webcam-based **age-group classification** with transfer learning.
Each class represents a fixed age span (default: **3 years**).

## Features

- Transfer learning with pre-trained ResNet18 (ImageNet weights)
- UTKFace dataset downloader/preparation script (Kaggle or direct URL)
- Training script for age-bin classification
- Real-time webcam inference
- Checkpoint save/load with class-name metadata
- Configurable age bins (default `0-2`, `3-5`, `6-8`, ...)

## Why ResNet18 for this project?

For webcam age-category prediction, **ResNet18** is a practical balance between accuracy and speed:

- It is deep enough to transfer useful facial features learned from ImageNet.
- It is lightweight compared with larger backbones (ResNet50/101), which helps real-time webcam inference on CPU/GPU.
- It is widely supported and stable in torchvision, making it easy to train and deploy.

## Architecture modifications for age classification

Starting from pre-trained `resnet18`, the model is adapted as follows:

1. **Transfer learning setup**: backbone layers are frozen (`feature_extract=True`) during initial training.
2. **New classification head**: the original `fc` layer is replaced with:
   - `Dropout(p=0.3)` for regularization
   - `Linear(in_features, num_age_bins)` to match your age categories
3. **Age-bin labels** are generated from:
   - `min_age`
   - `max_age`
   - `age_bin_size` (default: `3`)

Example: if `min_age=0`, `max_age=11`, and `age_bin_size=3`, classes become:
`0-2`, `3-5`, `6-8`, `9-11`.


## Model evaluation metrics (what to report and why)

Because this is a **multi-class age-bin classification** problem, evaluate on a held-out validation/test set with the following metrics:

1. **Accuracy**
   - Definition: fraction of predictions where the predicted age bin equals the true age bin.
   - Why it matters: gives a simple overall performance number and is easy to compare across experiments.
   - Limitation: can be misleading when some age bins have many more samples than others.

2. **Precision (macro + weighted)**
   - Definition (per class): `TP / (TP + FP)` for each age bin.
   - Macro precision: unweighted average across bins.
   - Weighted precision: average weighted by bin support.
   - Why it matters: tells you how often a predicted bin is correct, which is useful if false positives in specific age groups are costly.

3. **Recall (macro + weighted)**
   - Definition (per class): `TP / (TP + FN)` for each age bin.
   - Macro recall: unweighted average across bins.
   - Weighted recall: average weighted by bin support.
   - Why it matters: shows whether the model is missing certain age groups (e.g., under-represented bins). High recall means fewer missed true instances of each group.

4. **F1 score (macro + weighted)**
   - Definition: harmonic mean of precision and recall.
   - Macro F1: treats all bins equally.
   - Weighted F1: reflects dataset frequency.
   - Why it matters: balances false positives and false negatives, making it a strong single summary metric when class imbalance exists.

5. **Confusion matrix**
   - Definition: table of true bins vs predicted bins.
   - Why it matters: especially important for age estimation, because errors are ordinal. It reveals whether mistakes are mostly between neighboring bins (acceptable in many use cases) versus far-apart bins (more problematic).

### Recommended reporting set for this project

At minimum, report:

- Top-1 Accuracy
- Macro Precision, Macro Recall, Macro F1
- Weighted Precision, Weighted Recall, Weighted F1
- Confusion Matrix (with class labels like `0-2`, `3-5`, `6-8`, ...)

This combination gives:

- a global headline number (**accuracy**),
- fairness across age groups (**macro metrics**),
- population-level realism (**weighted metrics**), and
- interpretable error structure (**confusion matrix**).

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

What this script does:

- Downloads archive from `jangedoo/utkface-new` (default dataset ID).
- Extracts archive under `datasets/_downloads/extracted`.
- Recursively collects valid UTKFace image files.
- Writes a flat, training-ready image folder to `datasets/UTKFace`.

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
