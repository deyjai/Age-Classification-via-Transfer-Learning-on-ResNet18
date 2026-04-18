"""Utility helpers for data collection, preprocessing, and inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import torch
import torchvision.transforms as transforms


def ensure_class_directories(data_root: str | Path, class_names: Iterable[str]) -> None:
    """Create data directories for each class if they do not already exist."""
    data_root = Path(data_root)
    for class_name in class_names:
        (data_root / class_name).mkdir(parents=True, exist_ok=True)


def collect_data_for_class(
    class_name: str,
    data_root: str | Path = "data",
    num_samples: int = 10,
    camera_index: int = 0,
) -> None:
    """Collect images for a specific class using webcam snapshots.

    Press 'c' to capture the current frame.
    Press 'q' to stop early.
    """
    class_dir = Path(data_root) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    count = 0
    print(f"Collecting samples for class '{class_name}'.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from webcam.")
            break

        cv2.putText(
            frame,
            f"Class: {class_name} | Captured: {count}/{num_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Collect Data (press 'c' to capture, 'q' to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            out_path = class_dir / f"{class_name}_{count}.jpg"
            cv2.imwrite(str(out_path), frame)
            count += 1
            print(f"Captured {count}/{num_samples}: {out_path}")
            if count >= num_samples:
                break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_transforms(train: bool = True):
    """Get image transforms compatible with ResNet18."""
    base = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
    ]

    if train:
        base.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )

    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transforms.Compose(base)


def load_dataset_tensors(
    data_root: str | Path,
    class_names: list[str],
    image_transforms,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load class images from disk and convert to training tensors."""
    data_root = Path(data_root)

    data: list[torch.Tensor] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue

        for filename in os.listdir(class_dir):
            image_path = class_dir / filename
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            data.append(image_transforms(img))
            labels.append(class_index)

    if not data:
        raise ValueError(
            f"No valid images found under '{data_root}'. "
            "Collect data first or verify class directory names."
        )

    return torch.stack(data), torch.tensor(labels)


def prediction_to_label(pred_idx: int, class_names: list[str]) -> str:
    """Map a predicted class index to a class name."""
    if pred_idx < 0 or pred_idx >= len(class_names):
        return "unknown"
    return class_names[pred_idx]
