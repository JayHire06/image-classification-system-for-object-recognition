"""Shared CIFAR-10 splits and preprocessing for both classifiers."""

import argparse
import random
from collections.abc import Callable
from copy import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

SEED = 42
TRAIN_SIZE = 40_000
VALIDATION_SIZE = 10_000
TEST_SIZE = 10_000
CIFAR_IMAGE_SIZE = 32
CNN_IMAGE_SIZE = 224
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_cnn_transform(*, training: bool = False) -> transforms.Compose:
    """Build MobileNetV2 preprocessing for training or evaluation.

    Args:
        training: Apply random crop and horizontal flip before resizing.

    Returns:
        A transform from RGB PIL images to normalized float tensors.
    """
    steps = []
    if training:
        steps.extend(
            [
                transforms.RandomCrop(CIFAR_IMAGE_SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend(
        [
            transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(steps)


def load_cifar10_splits(
    root: str | Path = DEFAULT_DATA_DIR,
    *,
    train_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    eval_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    download: bool = True,
) -> tuple[Subset, Subset, CIFAR10]:
    """Load disjoint 40k/10k/10k splits using the fixed project seed.

    Args:
        root: Directory for the CIFAR-10 archive and extracted data.
        train_transform: Training transform. None returns raw images.
        eval_transform: Transform for validation and test images. None returns
            raw images.
        download: Download and extract CIFAR-10 if it is missing.

    Returns:
        Training and validation subsets of the official training data, followed
        by the official test dataset in its original order.

    Raises:
        ValueError: If the source dataset has unexpected split sizes.
    """
    training_data = CIFAR10(
        root, train=True, transform=train_transform, download=download
    )
    test_data = CIFAR10(root, train=False, transform=eval_transform)
    if (
        len(training_data) != TRAIN_SIZE + VALIDATION_SIZE
        or len(test_data) != TEST_SIZE
    ):
        raise ValueError(
            "CIFAR-10 must contain 50,000 train and 10,000 test images."
        )

    # Share the image storage while giving validation its own transform.
    validation_data = copy(training_data)
    validation_data.transform = eval_transform
    indices = torch.randperm(
        len(training_data), generator=torch.Generator().manual_seed(SEED)
    ).tolist()
    return (
        Subset(training_data, indices[:TRAIN_SIZE]),
        Subset(validation_data, indices[TRAIN_SIZE:]),
        test_data,
    )


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the worker's PyTorch seed.

    Args:
        worker_id: Worker index passed by the data loader.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    root: str | Path = DEFAULT_DATA_DIR,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build MobileNetV2 loaders with reproducible training shuffle.

    When num_workers is zero, seed PyTorch in the caller before iterating the
    training loader to reproduce augmentation in the main process.

    Args:
        root: Directory where torchvision stores CIFAR-10.
        batch_size: Maximum number of images per batch.
        num_workers: Number of loader subprocesses.
        pin_memory: Use pinned host memory for CUDA transfers.
        persistent_workers: Keep loader subprocesses between epochs.
        download: Download and extract CIFAR-10 if it is missing.

    Returns:
        Training, validation, and test loaders in that order.
    """
    datasets = load_cifar10_splits(
        root,
        train_transform=build_cnn_transform(training=True),
        eval_transform=build_cnn_transform(),
        download=download,
    )
    train, validation, test = (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == 0,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(SEED + split),
        )
        for split, dataset in enumerate(datasets)
    )
    return train, validation, test


def main() -> None:
    """Download CIFAR-10 and report split sizes and batch shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    train, validation, test = build_dataloaders(
        args.data_dir, batch_size=4, download=not args.no_download
    )
    images, labels = next(iter(train))
    print(
        f"Training: {len(train.dataset):,} | "
        f"Validation: {len(validation.dataset):,} | "
        f"Test: {len(test.dataset):,}"
    )
    print(
        f"First training batch: images {tuple(images.shape)}, "
        f"labels {tuple(labels.shape)}"
    )


if __name__ == "__main__":
    main()
