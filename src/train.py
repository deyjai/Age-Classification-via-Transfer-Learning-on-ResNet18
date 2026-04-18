"""Training entrypoint for ResNet18 transfer learning."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import create_resnet18_transfer_model
from utils import (
    generate_age_bins,
    get_transforms,
    load_utkface_tensors,
)

DEFAULT_MIN_AGE = 0
DEFAULT_MAX_AGE = 80
DEFAULT_BIN_SIZE = 3


def train_model(
    utkface_root: str = "datasets/UTKFace",
    class_names: list[str] | None = None,
    min_age: int = DEFAULT_MIN_AGE,
    max_age: int = DEFAULT_MAX_AGE,
    age_bin_size: int = DEFAULT_BIN_SIZE,
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 1e-3,
    model_path: str = "model_state.pth",
) -> str:
    """Train a transfer-learning ResNet18 classifier and save the checkpoint."""
    class_names = class_names or generate_age_bins(
        min_age=min_age,
        max_age=max_age,
        bin_size=age_bin_size,
    )

    transforms = get_transforms(train=True)
    data, labels = load_utkface_tensors(
        utkface_root=utkface_root,
        class_names=class_names,
        image_transforms=transforms,
        min_age=min_age,
        max_age=max_age,
        bin_size=age_bin_size,
    )

    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet18_transfer_model(num_classes=len(class_names), feature_extract=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
        },
        model_path,
    )
    print(f"Model checkpoint saved to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 transfer-learning classifier")
    parser.add_argument(
        "--utkface-root",
        default="datasets/UTKFace",
        help="Directory containing prepared UTKFace images.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE)
    parser.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE)
    parser.add_argument(
        "--age-bin-size",
        type=int,
        default=DEFAULT_BIN_SIZE,
        help="Width of each age category in years.",
    )
    parser.add_argument("--model-path", default="model_state.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        utkface_root=args.utkface_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_age=args.min_age,
        max_age=args.max_age,
        age_bin_size=args.age_bin_size,
        model_path=args.model_path,
    )
