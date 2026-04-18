"""Training entrypoint for ResNet18 transfer learning."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import create_resnet18_transfer_model
from utils import (
    collect_data_for_class,
    ensure_class_directories,
    get_transforms,
    load_dataset_tensors,
)

DEFAULT_CLASS_NAMES = ["class1", "class2", "class3"]


def train_model(
    data_root: str = "data",
    class_names: list[str] | None = None,
    num_samples: int = 10,
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 1e-3,
    collect_data: bool = False,
    model_path: str = "model_state.pth",
) -> str:
    """Train a transfer-learning ResNet18 classifier and save the checkpoint."""
    class_names = class_names or DEFAULT_CLASS_NAMES
    ensure_class_directories(data_root, class_names)

    if collect_data:
        for class_name in class_names:
            collect_data_for_class(class_name, data_root=data_root, num_samples=num_samples)

    transforms = get_transforms(train=True)
    data, labels = load_dataset_tensors(data_root, class_names, transforms)

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
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect webcam samples before training.",
    )
    parser.add_argument("--model-path", default="model_state.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        collect_data=args.collect_data,
        model_path=args.model_path,
    )
