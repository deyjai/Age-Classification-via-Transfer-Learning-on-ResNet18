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


def apply_zoom(frame, zoom_factor: float):
    """Apply digital zoom by center-cropping and resizing back to frame size."""
    if zoom_factor <= 1.0:
        return frame

    height, width = frame.shape[:2]
    crop_width = max(int(width / zoom_factor), 1)
    crop_height = max(int(height / zoom_factor), 1)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    zoomed = frame[y1:y2, x1:x2]
    return cv2.resize(zoomed, (width, height), interpolation=cv2.INTER_LINEAR)


def run_webcam_inference(
    model_path: str,
    camera_index: int = 0,
    initial_zoom: float = 1.0,
    zoom_step: float = 0.1,
    max_zoom: float = 3.0,
) -> None:
    """Run live webcam inference with the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)
    transforms = get_transforms(train=False)

    cap = cv2.VideoCapture(camera_index)
    zoom_factor = max(1.0, min(initial_zoom, max_zoom))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from webcam.")
            break

        display_frame = apply_zoom(frame, zoom_factor)
        input_img = transforms(display_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            pred_idx = torch.argmax(output, dim=1).item()

        label = prediction_to_label(pred_idx, class_names)
        cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(
            display_frame,
            f"Zoom: {zoom_factor:.1f}x (+/-)",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Predictions (press 'q' to quit)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("+"), ord("=")):
            zoom_factor = min(max_zoom, zoom_factor + zoom_step)
        if key in (ord("-"), ord("_")):
            zoom_factor = max(1.0, zoom_factor - zoom_step)

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam classification")
    parser.add_argument("--model-path", default="model_state.pth")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--utkface-root", default="datasets/UTKFace")
    parser.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE)
    parser.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE)
    parser.add_argument("--age-bin-size", type=int, default=DEFAULT_BIN_SIZE)
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train a model first if checkpoint does not exist.",
    )
    parser.add_argument("--initial-zoom", type=float, default=1.0, help="Starting zoom factor.")
    parser.add_argument(
        "--zoom-step",
        type=float,
        default=0.1,
        help="Zoom increment/decrement used with +/- keys.",
    )
    parser.add_argument("--max-zoom", type=float, default=3.0, help="Maximum zoom factor.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not Path(args.model_path).exists():
        if args.train_if_missing:
            print(f"Checkpoint '{args.model_path}' not found. Training a new model...")
            train_model(
                model_path=args.model_path,
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

    run_webcam_inference(
        model_path=args.model_path,
        camera_index=args.camera_index,
        initial_zoom=args.initial_zoom,
        zoom_step=args.zoom_step,
        max_zoom=args.max_zoom,
    )
