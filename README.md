# Real-Time Age Classification with ResNet18 (Transfer Learning)

## Overview

This project uses a pre-trained **ResNet18** model for webcam-based **age-group classification** with transfer learning.
Each class can represent a fixed age span (default: **3 years**).

## Features

- Transfer learning with pre-trained ResNet18 (ImageNet weights)
- Optional webcam data collection
- Training script and real-time inference script
- Checkpoint save/load with class-name metadata
- Configurable age bins (default `0-2`, `3-5`, `6-8`, ...)

## Why ResNet18 for this project?

For webcam age-category prediction, **ResNet18** is a practical balance between accuracy and speed:

- It is deep enough to transfer useful facial features learned from ImageNet.
- It is lightweight compared with larger backbones (ResNet50/101), which helps real-time webcam inference on CPU/GPU.
- It is widely supported and stable in torchvision, making it easy to train and deploy.

### Architecture modifications for age classification

Starting from pre-trained `resnet18`, the model is adapted as follows:

1. **Transfer learning setup**: backbone layers are frozen (`feature_extract=True`) during initial training.
2. **New classification head**: the original `fc` layer is replaced with:
   - `Dropout(p=0.3)` for regularization
   - `Linear(in_features, num_age_bins)` to match your age categories
3. **Age-bin labels**: bins are generated automatically from:
   - `min_age`
   - `max_age`
   - `age_bin_size` (default: `3`)

So if `min_age=0`, `max_age=11`, and `age_bin_size=3`, classes become:
`0-2`, `3-5`, `6-8`, `9-11`.

## Setup

```bash
pip install -r requirements.txt
```

## 1) Install and Prepare UTKFace

1. Install Kaggle CLI (one-time):

```bash
pip install kaggle
```

2. Create a Kaggle account and API token:
   - Go to Kaggle -> Account -> **Create New API Token**
   - This downloads `kaggle.json`

3. Place credentials file:
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
   - **Windows:** `%USERPROFILE%\\.kaggle\\kaggle.json`

4. Restrict permissions (Linux/macOS):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

5. Download/prepare UTKFace into `datasets/UTKFace/` with one command:

```bash
python src/download_utkface.py --source kaggle --output-dir datasets/UTKFace
```

6. Alternative (if not using Kaggle CLI): use `--source url --url <direct_archive_url>`.
> Downloaded datasets/archives are ignored by git via `.gitignore` (`datasets/`, archive extensions).

7. The prepared folder will contain image names like:
   `25_0_2_20170116174525125.jpg`.
## 2) Train the Model (Transfer Learning on ResNet18)

Run training on the prepared UTKFace folder:

```bash
python src/train.py --utkface-root datasets/UTKFace --min-age 0 --max-age 80 --age-bin-size 3 --epochs 10
```

This starts with pre-trained ResNet18 weights and fine-tunes the final classification head on UTKFace age bins.

## 3) Use Webcam for Inference (Classification Only)

```bash
python src/webcam.py --model-path model_state.pth
```

If `model_state.pth` is missing, you can auto-train from UTKFace first:

```bash
python src/webcam.py --train-if-missing --utkface-root datasets/UTKFace --min-age 0 --max-age 80 --age-bin-size 3
```

The webcam mode is inference-only (no data collection).
