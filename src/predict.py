"""Load local checkpoints and classify a single image."""

import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from torchvision.models.mobilenetv2 import MobileNetV2

from .dataset import CLASS_NAMES, build_cnn_transform
from .features import extract_hog
from .models import build_mobilenet_v2

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATHS = {
    "mobilenet_v2": MODEL_DIR / "mobilenet_v2.pth",
    "hog_svm": MODEL_DIR / "hog_svm.pkl",
}
MAX_IMAGE_PIXELS = 24_000_000


def decode_image(contents: bytes) -> Image.Image:
    """Read a JPEG or PNG, apply its orientation, and convert to RGB.

    Args:
        contents: Encoded image bytes from the upload.

    Returns:
        A decoded RGB image independent of the input buffer.

    Raises:
        ValueError: If the file is unreadable or exceeds 24 megapixels.
    """
    try:
        with Image.open(BytesIO(contents), formats=("JPEG", "PNG")) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Choose an image under 24 megapixels.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Choose an image under 24 megapixels.") from exc
    except (OSError, SyntaxError) as exc:
        message = "This file could not be read as a JPG or PNG."
        raise ValueError(message) from exc


def load_svm(path: Path) -> CalibratedClassifierCV:
    """Load the calibrated SVM from a trusted project artifact.

    Args:
        path: Path to the locally trained pickle file.

    Returns:
        The fitted SVM with probability columns in CIFAR-10 class order.

    Raises:
        ValueError: If the checkpoint uses a different class order.
    """
    with path.open("rb") as file:
        model = pickle.load(file)
    classes = getattr(model, "classes_", ())
    if tuple(classes) != tuple(range(len(CLASS_NAMES))):
        raise ValueError("The SVM checkpoint has an unexpected class order.")
    return model


def load_cnn(path: Path, device: torch.device) -> MobileNetV2:
    """Restore MobileNetV2 on the requested device in evaluation mode.

    Args:
        path: Path to the trained state dictionary.
        device: CPU or CUDA device for inference.

    Returns:
        A MobileNetV2 ready to classify images.
    """
    model = build_mobilenet_v2(pretrained=False)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    return model.to(device=device, memory_format=torch.channels_last).eval()


def predict_probabilities(
    model: CalibratedClassifierCV | MobileNetV2, image: Image.Image
) -> NDArray[np.float64]:
    """Predict probabilities for a single image using a loaded model.

    Args:
        model: A model returned by load_svm or load_cnn.
        image: An RGB PIL image.

    Returns:
        A ten-element vector in CIFAR-10 class order.
    """
    if isinstance(model, MobileNetV2):
        device = next(model.parameters()).device
        batch = build_cnn_transform()(image).unsqueeze(0)
        batch = batch.to(device=device, memory_format=torch.channels_last)
        with torch.inference_mode():
            probabilities = model(batch).softmax(dim=1)[0].cpu().numpy()
        return probabilities.astype(np.float64)

    cell_size = getattr(model, "hog_cell_size_", 8)
    features = extract_hog(image, cell_size=cell_size)
    return model.predict_proba(features[None, :])[0]
