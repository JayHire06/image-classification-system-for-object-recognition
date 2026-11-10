"""Shared image preprocessing for the HOG + SVM baseline."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.feature import hog

from .dataset import CIFAR_IMAGE_SIZE


def extract_hog(
    image: Image.Image, *, cell_size: int = 8
) -> NDArray[np.float32]:
    """Extract a HOG feature vector from a CIFAR-10 image or uploaded image.

    Args:
        image: A PIL image in any color mode or size.
        cell_size: Width and height of each HOG cell in pixels.

    Returns:
        A float32 descriptor of the RGB image at 32 by 32 pixels.
    """
    image = image.convert("RGB")
    size = (CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE)
    if image.size != size:
        image = image.resize(size, resample=Image.Resampling.BILINEAR)

    descriptor = hog(
        np.asarray(image),
        orientations=9,
        pixels_per_cell=(cell_size, cell_size),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        channel_axis=-1,
    )
    return np.asarray(descriptor, dtype=np.float32)


def extract_hog_dataset(
    samples: Iterable[tuple[Image.Image, int]],
    *,
    cell_size: int = 8,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Extract features and labels from an ordered, nonempty image dataset.

    Args:
        samples: Raw PIL images paired with integer class indices.
        cell_size: Width and height of each HOG cell in pixels.

    Returns:
        A feature matrix and a label vector in the original sample order.
    """
    features = []
    labels = []
    for image, label in samples:
        features.append(extract_hog(image, cell_size=cell_size))
        labels.append(label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)
