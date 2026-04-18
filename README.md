# Real-Time Image Classification with ResNet (Transfer Learning)

## Overview

This project uses a pre-trained ResNet18 model for webcam-based image classification with transfer learning.

## Features

- Transfer learning with ResNet18
- Optional webcam data collection
- Training script and real-time inference script
- Checkpoint save/load with class-name metadata

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --collect-data
```

> Press `c` to capture images and `q` to stop collection for each class.

## Run Inference

```bash
python src/webcam.py --model-path model_state.pth
```

Or auto-train if the checkpoint is missing:

```bash
python src/webcam.py --train-if-missing --collect-data
```
