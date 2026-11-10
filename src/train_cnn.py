"""Train and validate MobileNetV2 on the fixed CIFAR-10 splits."""

import argparse
import json
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import MobileNetV2

from .dataset import (
    CLASS_NAMES,
    CNN_IMAGE_SIZE,
    DEFAULT_DATA_DIR,
    SEED,
    build_dataloaders,
)
from .models import (
    PRETRAINED_WEIGHTS,
    build_mobilenet_v2,
    set_trainable_layers,
    set_training_mode,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "mobilenet_v2.pth"
HISTORY_PATH = PROJECT_DIR / "results" / "mobilenet_v2_history.json"


def seed_everything() -> None:
    """Seed Python, NumPy, and PyTorch and enable deterministic operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def run_epoch(
    model: MobileNetV2,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    optimizer: Optimizer | None = None,
    top_blocks: int = 0,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """Run one training or validation pass and aggregate sample metrics.

    Args:
        model: The configured MobileNetV2.
        loader: The training or validation data loader.
        loss_fn: Cross-entropy loss for class logits.
        device: Device hosting the model and batches.
        scaler: Gradient scaler, disabled when training on CPU.
        optimizer: Optimizer for training. Omit it for validation.
        top_blocks: Number of unfrozen backbone blocks during training.
        mixed_precision: Enable automatic mixed precision on CUDA.

    Returns:
        Mean loss and accuracy, with macro-F1 added for validation.
    """
    training = optimizer is not None
    if training:
        set_training_mode(model, top_blocks)
    else:
        model.eval()

    use_cuda = device.type == "cuda"
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    predictions = []
    targets = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=use_cuda,
                memory_format=torch.channels_last,
            )
            labels = labels.to(device, non_blocking=use_cuda)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=use_cuda and mixed_precision
            ):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predicted = logits.detach().argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.detach() * batch_size
            correct += (predicted == labels).sum()
            seen += batch_size
            if not training:
                predictions.append(predicted)
                targets.append(labels)

    metrics = {
        "loss": total_loss.item() / seen,
        "accuracy": correct.item() / seen,
    }
    if not training:
        metrics["macro_f1"] = float(
            f1_score(
                torch.cat(targets).cpu().numpy(),
                torch.cat(predictions).cpu().numpy(),
                labels=range(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        )
    return metrics


def save_artifacts(
    weights: dict[str, torch.Tensor], history: dict[str, object]
) -> None:
    """Write the best CPU state dict and the completed training history.

    Args:
        weights: The best validation checkpoint, stored as CPU tensors.
        history: Run settings, selected epoch, and per-epoch metrics.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=MODEL_PATH.parent) as model_temp,
        TemporaryDirectory(dir=HISTORY_PATH.parent) as history_temp,
    ):
        model_file = Path(model_temp) / MODEL_PATH.name
        history_file = Path(history_temp) / HISTORY_PATH.name
        torch.save(weights, model_file)
        history_file.write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        model_file.replace(MODEL_PATH)
        history_file.replace(HISTORY_PATH)


def main() -> None:
    """Run both training stages and save the best validation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-blocks", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    args = parser.parse_args()

    if min(args.batch_size, args.head_epochs, args.fine_tune_epochs) < 1:
        parser.error("Batch size and both epoch counts must be positive.")
    if args.num_workers < 0:
        parser.error("The worker count must be nonnegative.")
    if not 0 < args.fine_tune_lr < args.head_lr < float("inf"):
        parser.error("Fine-tuning must use a positive, lower learning rate.")

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Training on {device.type.upper()}", flush=True)
    model = build_mobilenet_v2()
    if not 1 <= args.fine_tune_blocks <= len(model.features):
        parser.error(f"Fine-tuning blocks must be 1 to {len(model.features)}.")
    model.to(device=device, memory_format=torch.channels_last)
    train, validation, _ = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        download=not args.no_download,
    )
    loss_fn = nn.CrossEntropyLoss()

    stages = (
        ("head", args.head_epochs, 0, args.head_lr),
        (
            "fine_tune",
            args.fine_tune_epochs,
            args.fine_tune_blocks,
            args.fine_tune_lr,
        ),
    )
    epochs = []
    best_weights = None
    best_epoch = 0
    best_f1 = -1.0
    for stage, stage_epochs, top_blocks, learning_rate in stages:
        if best_weights is not None:
            model.load_state_dict(best_weights)
        set_trainable_layers(model, top_blocks)
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = Adam(
            trainable_parameters,
            lr=learning_rate,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=use_cuda)
        for _ in range(stage_epochs):
            start = perf_counter()
            train_metrics = run_epoch(
                model,
                train,
                loss_fn,
                device,
                scaler,
                optimizer=optimizer,
                top_blocks=top_blocks,
            )
            validation_metrics = run_epoch(
                model, validation, loss_fn, device, scaler
            )
            score = validation_metrics["macro_f1"]
            epoch = len(epochs) + 1
            elapsed = perf_counter() - start
            epochs.append(
                {
                    "epoch": epoch,
                    "stage": stage,
                    "learning_rate": learning_rate,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "seconds": elapsed,
                }
            )
            if score > best_f1:
                best_f1 = score
                best_epoch = epoch
                best_weights = {
                    key: value.detach().to("cpu", copy=True)
                    for key, value in model.state_dict().items()
                }
            print(
                f"Epoch {epoch} ({stage}) | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"validation macro-F1 {score:.4f}, "
                f"accuracy {validation_metrics['accuracy']:.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

    history = {
        "model": "mobilenet_v2",
        "seed": SEED,
        "device": device.type,
        "mixed_precision": use_cuda,
        "pretrained_weights": PRETRAINED_WEIGHTS.name,
        "class_names": CLASS_NAMES,
        "image_size": CNN_IMAGE_SIZE,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "fine_tune_blocks": args.fine_tune_blocks,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "epochs": epochs,
    }
    if best_weights is None:
        raise RuntimeError("No validation checkpoint was produced.")
    save_artifacts(best_weights, history)
    print(f"Best validation macro-F1 {best_f1:.4f} at epoch {best_epoch}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Training history saved to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
