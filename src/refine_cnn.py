"""Test extra CIFAR-100 supervision and stronger CIFAR-10 fine-tuning."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models.mobilenetv2 import MobileNetV2

from .dataset import (
    DEFAULT_DATA_DIR,
    SEED,
    _seed_worker,
    build_cnn_transform,
    load_cifar10_splits,
)
from .evaluate import RESULTS_DIR
from .models import set_trainable_layers
from .predict import MODEL_PATHS, load_cnn
from .train_cnn import run_epoch, seed_everything

REFINED_PATH = MODEL_PATHS["mobilenet_v2"].with_name(
    "mobilenet_v2_refined.pth"
)
REPORT_PATH = RESULTS_DIR / "mobilenet_v2_refinement.json"
TOP_BLOCKS = 10
TARGET_LR = 1.5e-4
AUX_LR = 3e-4


class ExternalTestSet(Dataset):
    """The CIFAR-10.1 v6 images with standard CIFAR-10 preprocessing."""

    def __init__(self, root: Path) -> None:
        """Load the published NumPy arrays from the given dataset directory."""
        self.images = np.load(
            root / "cifar10.1_v6_data.npy", allow_pickle=False
        )
        self.labels = np.load(
            root / "cifar10.1_v6_labels.npy", allow_pickle=False
        )
        if (
            self.images.shape != (2000, 32, 32, 3)
            or self.images.dtype != np.uint8
            or self.labels.shape != (2000,)
            or self.labels.dtype.kind not in "iu"
            or not np.array_equal(np.bincount(self.labels), np.full(10, 200))
        ):
            raise ValueError("Expected the balanced CIFAR-10.1 v6 files.")
        self.transform = build_cnn_transform()

    def __len__(self) -> int:
        """Return the number of external test images."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one normalized RGB tensor and its class index."""
        return self.transform(Image.fromarray(self.images[index])), int(
            self.labels[index]
        )


def pixel_hash(image: NDArray[np.uint8]) -> bytes:
    """Fingerprint an exact RGB image."""
    return sha256(image.tobytes()).digest()


def difference_hash(image: NDArray[np.uint8]) -> bytes:
    """Fingerprint adjacent luminance changes on a small grayscale image."""
    grayscale = Image.fromarray(image).convert("L")
    grid = np.asarray(grayscale.resize((9, 8), Image.Resampling.LANCZOS))
    return np.packbits(grid[:, 1:] > grid[:, :-1]).tobytes()


def filter_auxiliary(
    images: NDArray[np.uint8], reference: tuple[NDArray[np.uint8], ...]
) -> tuple[list[int], dict[str, int]]:
    """Remove exact or matching perceptual hashes from auxiliary training.

    Args:
        images: Raw CIFAR-100 training images in official order.
        reference: CIFAR-10 and CIFAR-10.1 images to screen against.

    Returns:
        Kept indices and mutually exclusive removal counts.
    """
    exact = {pixel_hash(image) for dataset in reference for image in dataset}
    perceptual = {
        difference_hash(image) for dataset in reference for image in dataset
    }
    indices = []
    exact_removed = 0
    perceptual_removed = 0
    for index, image in enumerate(images):
        if pixel_hash(image) in exact:
            exact_removed += 1
        elif difference_hash(image) in perceptual:
            perceptual_removed += 1
        else:
            indices.append(index)
    return indices, {
        "available": len(images),
        "used": len(indices),
        "exact_removed": exact_removed,
        "perceptual_hash_removed": perceptual_removed,
    }


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    *,
    training: bool = False,
) -> DataLoader:
    """Create a seeded loader with reproducible training shuffling.

    Args:
        dataset: Images and labels to batch.
        args: Run arguments specifying batch size and worker count.
        device: Device used to determine whether to pin memory.
        training: Shuffle examples and keep workers alive between epochs.

    Returns:
        A loader for the requested dataset.
    """
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0 and training,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
        worker_init_fn=_seed_worker,
    )


def snapshot(model: MobileNetV2) -> dict[str, torch.Tensor]:
    """Copy a checkpoint into independent CPU tensors."""
    return {
        name: tensor.detach().to("cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def tune(
    model: MobileNetV2,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Fine-tune on CIFAR-10 and return the best validation checkpoint.

    Args:
        model: Initialized model with a ten-class head.
        train_loader: Augmented CIFAR-10 training batches.
        validation_loader: CIFAR-10 validation batches.
        device: Device hosting the model.
        epochs: Number of training epochs.
        name: Candidate label for progress output.

    Returns:
        The best checkpoint and its validation history.
    """
    set_trainable_layers(model, TOP_BLOCKS)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TARGET_LR,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    training_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    scoring_loss = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_weights = snapshot(model)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        train_metrics = run_epoch(
            model, train_loader, training_loss, device, scaler,
            optimizer=optimizer, top_blocks=TOP_BLOCKS,
        )
        validation = run_epoch(
            model, validation_loader, scoring_loss, device, scaler,
            mixed_precision=False,
        )
        score = validation["macro_f1"]
        history.append({
            "epoch": epoch, "train": train_metrics, "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": perf_counter() - start,
        })
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_weights = snapshot(model)
        scheduler.step()
        print(
            f"{name} epoch {epoch}/{epochs} | validation macro-F1 {score:.4f}"
            f" | accuracy {validation['accuracy']:.2%}",
            flush=True,
        )
    return best_weights, {
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1]["validation"],
        "epochs": history,
    }


def pretrain_auxiliary(
    model: MobileNetV2,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> list[dict[str, float]]:
    """Adapt the shared features using a temporary 100-class head.

    Args:
        model: Initialized model whose original head will be restored.
        loader: Screened CIFAR-100 training batches.
        device: Device hosting the model.
        epochs: Number of auxiliary training epochs.

    Returns:
        The auxiliary training metrics for each epoch.
    """
    target_head = model.classifier[-1]
    model.classifier[-1] = nn.Linear(model.last_channel, 100).to(device)
    set_trainable_layers(model, TOP_BLOCKS)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=AUX_LR
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    history = []
    for epoch in range(1, epochs + 1):
        metrics = run_epoch(
            model, loader, loss, device, scaler,
            optimizer=optimizer, top_blocks=TOP_BLOCKS,
        )
        history.append(metrics)
        print(
            f"CIFAR-100 epoch {epoch}/{epochs} | loss {metrics['loss']:.4f}",
            flush=True,
        )
    model.classifier[-1] = target_head
    return history


def save_run(
    weights: dict[str, torch.Tensor], report: dict[str, object]
) -> None:
    """Stage the selected checkpoint and comparison before publishing them.

    Args:
        weights: CPU state dictionary selected on validation.
        report: Run settings and metrics for all candidates.
    """
    with (
        TemporaryDirectory(dir=REFINED_PATH.parent) as model_stage,
        TemporaryDirectory(dir=REPORT_PATH.parent) as report_stage,
    ):
        model = Path(model_stage) / REFINED_PATH.name
        report_file = Path(report_stage) / REPORT_PATH.name
        torch.save(weights, model)
        report_file.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        model.replace(REFINED_PATH)
        report_file.replace(REPORT_PATH)


def main() -> None:
    """Train both candidates, select on validation, then score held-outs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--aux-dir", type=Path)
    parser.add_argument("--external-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--aux-epochs", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.aux_epochs) < 1:
        parser.error("Batch size and epoch counts must be positive.")
    if args.num_workers < 0:
        parser.error("Worker count must be nonnegative.")

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device.type.upper()}", flush=True)
    augmentation = build_cnn_transform(training=True)
    augmentation.transforms.insert(2, transforms.RandAugment(1, 5))
    train, validation, test = load_cifar10_splits(
        args.data_dir,
        train_transform=augmentation,
        eval_transform=build_cnn_transform(),
        download=not args.no_download,
    )
    external = ExternalTestSet(
        args.external_dir or args.data_dir / "cifar10.1"
    )
    aux = CIFAR100(
        args.aux_dir or args.data_dir,
        train=True,
        transform=augmentation,
        download=not args.no_download,
    )
    indices, audit = filter_auxiliary(
        aux.data, (train.dataset.data, test.data, external.images)
    )
    print(f"CIFAR-100 screening: {audit}", flush=True)
    cifar10_hashes = {
        pixel_hash(image)
        for dataset in (train.dataset.data, test.data)
        for image in dataset
    }
    external_indices = [
        i for i, image in enumerate(external.images)
        if pixel_hash(image) not in cifar10_hashes
    ]
    validation_loader = make_loader(validation, args, device)
    model = load_cnn(MODEL_PATHS["mobilenet_v2"], device)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    loss = nn.CrossEntropyLoss()
    weights = {"original": snapshot(model)}
    runs = {"original": {"validation": run_epoch(
        model, validation_loader, loss, device, scaler, mixed_precision=False
    )}}
    aux_history = []
    for name in ("cifar10_only", "cifar100_transfer"):
        seed_everything()
        model.load_state_dict(weights["original"])
        if name == "cifar100_transfer":
            aux_loader = make_loader(
                Subset(aux, indices), args, device, training=True
            )
            aux_history = pretrain_auxiliary(
                model, aux_loader, device, args.aux_epochs
            )
            del aux_loader
        seed_everything()
        train_loader = make_loader(train, args, device, training=True)
        weights[name], runs[name] = tune(
            model, train_loader, validation_loader, device, args.epochs, name
        )
        del train_loader

    selected = max(runs, key=lambda name: runs[name]["validation"]["macro_f1"])
    print(f"Selected on validation: {selected}", flush=True)
    for label, dataset in (
        ("cifar10_test", test),
        ("cifar10_1_v6", Subset(external, external_indices)),
    ):
        loader = make_loader(dataset, args, device)
        for name, checkpoint in weights.items():
            model.load_state_dict(checkpoint)
            metrics = run_epoch(
                model, loader, loss, device, scaler, mixed_precision=False
            )
            runs[name][label] = metrics
            print(f"{label} | {name} | {metrics['accuracy']:.2%}", flush=True)
    save_run(weights[selected], {
        "seed": SEED,
        "device": device.type,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cifar100_audit": audit,
        "cifar100_pretraining": aux_history,
        "cifar10_1_samples": len(external_indices),
        "selection_metric": "cifar10_validation_macro_f1",
        "selected": selected,
        "runs": runs,
    })
    print(f"Saved {REFINED_PATH.name} and {REPORT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
