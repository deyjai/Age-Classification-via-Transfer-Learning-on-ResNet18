"""Real-time webcam inference using a trained ResNet18 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from model import create_resnet18_transfer_model
from train import DEFAULT_BIN_SIZE, DEFAULT_MAX_AGE, DEFAULT_MIN_AGE, train_model
from utils import generate_age_bins, get_transforms, prediction_to_label


def load_model(model_path: str, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Load model and class names from a saved checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint.get(
        "class_names",
        generate_age_bins(DEFAULT_MIN_AGE, DEFAULT_MAX_AGE, DEFAULT_BIN_SIZE),
    )
    model = create_resnet18_transfer_model(num_classes=len(class_names), feature_extract=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names


def run_webcam_inference(model_path: str, camera_index: int = 0) -> None:
    """Run live webcam inference with the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)
    transforms = get_transforms(train=False)

    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from webcam.")
            break

        input_img = transforms(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            pred_idx = torch.argmax(output, dim=1).item()

        label = prediction_to_label(pred_idx, class_names)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.imshow("Predictions (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam classification")
    parser.add_argument("--model-path", default="model_state.pth")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--utkface-root", default=None)
    parser.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE)
    parser.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE)
    parser.add_argument("--age-bin-size", type=int, default=DEFAULT_BIN_SIZE)
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train a model first if checkpoint does not exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not Path(args.model_path).exists():
        if args.train_if_missing:
            print(f"Checkpoint '{args.model_path}' not found. Training a new model...")
            train_model(
                model_path=args.model_path,
                collect_data=args.collect_data,
                utkface_root=args.utkface_root,
                min_age=args.min_age,
                max_age=args.max_age,
                age_bin_size=args.age_bin_size,
            )
        else:
            raise FileNotFoundError(
                f"Checkpoint '{args.model_path}' not found. "
                "Run `python src/train.py` first or pass --train-if-missing."
            )

    run_webcam_inference(model_path=args.model_path, camera_index=args.camera_index)
