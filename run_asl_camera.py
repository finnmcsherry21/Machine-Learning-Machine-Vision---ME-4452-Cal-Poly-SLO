"""Launch the trained ASL classifier in a single OpenCV camera window.

Run from the project folder:
    python run_asl_camera.py

Press Q while the camera window is active to close it.
"""

from __future__ import annotations

import csv
import string
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenCV is not installed in the Python environment running this script.\n"
        "In the VS Code terminal, with the same interpreter selected, run:\n"
        "    python -m pip install opencv-python"
    ) from error
import numpy as np
import torch
from torch import nn


# Camera settings. Change CAMERA_INDEX to 1 if a different camera is needed.
CAMERA_INDEX = 0
MIRROR_CAMERA_IMAGE = False
ROI_FRACTION = 0.65
WINDOW_NAME = "ASL Real-Time Sign Recognition"

# These must match the preprocessing used when the models were trained.
IMAGE_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = list(string.ascii_uppercase) + ["del", "nothing", "space"]

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
SELECTION_FILE = RESULTS_DIR / "final_model_selection.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineCNN(nn.Module):
    """The exact CNN architecture used by the saved baseline checkpoint."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(images))


def create_mobilenet_model() -> nn.Module:
    """Re-create the trained MobileNet architecture without downloading weights."""
    try:
        from torchvision.models import mobilenet_v3_small
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The saved final model is MobileNetV3-Small, which requires torchvision.\n"
            "Install it with: python -m pip install torchvision"
        ) from error

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
    return model


def get_selected_model() -> tuple[str, Path]:
    """Read the final model choice written by the notebook."""
    if not SELECTION_FILE.is_file():
        raise FileNotFoundError(
            f"Final model selection file was not found: {SELECTION_FILE}\n"
            "Run the notebook through final model selection first."
        )

    with SELECTION_FILE.open(newline="", encoding="utf-8") as selection_file:
        selection = next(csv.DictReader(selection_file), None)

    if not selection:
        raise ValueError(f"Final model selection file is empty: {SELECTION_FILE}")

    model_name = selection["Selected model"]
    saved_path = Path(selection["Checkpoint path"])

    # Use the saved absolute path when available. The relative fallback keeps
    # the launcher usable if the project folder has been moved.
    if saved_path.is_file():
        checkpoint_path = saved_path
    elif model_name == "Baseline CNN":
        checkpoint_path = CHECKPOINT_DIR / "baseline_cnn_run_01" / "best_model.pt"
    elif model_name == "MobileNetV3-Small":
        checkpoint_path = CHECKPOINT_DIR / "mobilenet_classifier_run_01" / "best_model.pt"
    else:
        raise ValueError(f"Unsupported selected model: {model_name!r}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected model checkpoint was not found: {checkpoint_path}")

    return model_name, checkpoint_path


def load_model() -> tuple[nn.Module, str]:
    """Load the notebook's selected trained model onto the available device."""
    model_name, checkpoint_path = get_selected_model()

    if model_name == "Baseline CNN":
        model = BaselineCNN(num_classes=len(CLASS_NAMES))
    elif model_name == "MobileNetV3-Small":
        model = create_mobilenet_model()
    else:
        raise ValueError(f"Unsupported selected model: {model_name!r}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, model_name


def get_center_square_region(frame):
    """Return the centered square recognition region and its coordinates."""
    frame_height, frame_width = frame.shape[:2]
    side_length = int(min(frame_height, frame_width) * ROI_FRACTION)
    center_x, center_y = frame_width // 2, frame_height // 2
    x_start = center_x - side_length // 2
    y_start = center_y - side_length // 2
    x_end, y_end = x_start + side_length, y_start + side_length
    return frame[y_start:y_end, x_start:x_end], (x_start, y_start, x_end, y_end)


def classify_region(model: nn.Module, region_bgr) -> tuple[str, float]:
    """Predict the class and confidence for an OpenCV BGR image region."""
    region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    resized_region = cv2.resize(
        region_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # This is equivalent to the notebook's torchvision Resize, ToTensor, and
    # Normalize validation transform: RGB pixels become C x H x W float32.
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(resized_region.transpose(2, 0, 1))
    ).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image_tensor = image_tensor.sub(mean).div(std).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    return CLASS_NAMES[predicted_index.item()], confidence.item()


def main() -> None:
    model, model_name = load_model()
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Change CAMERA_INDEX near the top of this file if needed."
        )

    print(f"Loaded {model_name} on {DEVICE}. Press Q in the camera window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Camera frame could not be read.")

            if MIRROR_CAMERA_IMAGE:
                frame = cv2.flip(frame, 1)

            region, (x_start, y_start, x_end, y_end) = get_center_square_region(frame)
            label, confidence = classify_region(model, region)

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
            cv2.putText(
                frame,
                f"Prediction: {label} ({confidence:.1%})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Place one sign in the blue box | Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
