"""Compare the trained classifiers on the official CIFAR-10 test set."""

import argparse
import json
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .dataset import (
    CLASS_NAMES,
    DEFAULT_DATA_DIR,
    TEST_SIZE,
    build_cnn_transform,
)
from .features import extract_hog_dataset
from .predict import MODEL_PATHS, load_cnn, load_svm

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
CLASS_IDS = tuple(range(len(CLASS_NAMES)))


class ErrorCase(TypedDict):
    """An incorrect prediction in the official test set."""

    index: int
    actual: str
    predicted: str
    confidence: float


def calculate_metrics(
    targets: NDArray[np.int64], predictions: NDArray[np.int64]
) -> dict[str, object]:
    """Calculate classification metrics in the fixed CIFAR-10 class order.

    Args:
        targets: Ground-truth class indices.
        predictions: Predicted indices in the same sample order.

    Returns:
        Accuracy, macro-F1, per-class metrics, and confusion counts.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=CLASS_IDS, zero_division=0
    )
    return {
        "class_names": CLASS_NAMES,
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=CLASS_IDS
        ).tolist(),
    }


def select_error_cases(
    targets: NDArray[np.int64], probabilities: NDArray[np.floating]
) -> list[ErrorCase]:
    """Rank incorrect predictions by predicted-class probability.

    Args:
        targets: Ground-truth class indices in official test order.
        probabilities: One row per image and one column per class.

    Returns:
        Up to five errors, with ties resolved by original test order.
    """
    predictions = probabilities.argmax(axis=1)
    incorrect = np.flatnonzero(predictions != targets)
    confidence = probabilities[incorrect, predictions[incorrect]]
    order = np.argsort(-confidence, kind="stable")[:5]
    return [
        {
            "index": int(index),
            "actual": CLASS_NAMES[targets[index]],
            "predicted": CLASS_NAMES[predictions[index]],
            "confidence": float(probabilities[index, predictions[index]]),
        }
        for index in incorrect[order]
    ]


def hog_test_probabilities(test: CIFAR10) -> NDArray[np.float64]:
    """Run the calibrated SVM on raw test images.

    Args:
        test: The official test dataset with no image transform.

    Returns:
        Predicted probabilities in dataset and CIFAR-10 class order.

    """
    model = load_svm(MODEL_PATHS["hog_svm"])
    features, _ = extract_hog_dataset(
        test, cell_size=getattr(model, "hog_cell_size_", 8)
    )
    return model.predict_proba(features)


def cnn_test_probabilities(
    test: CIFAR10, batch_size: int, num_workers: int
) -> NDArray[np.float32]:
    """Run the saved MobileNetV2 with deterministic test preprocessing.

    Args:
        test: The official test dataset with no image transform.
        batch_size: Maximum number of images in each inference batch.
        num_workers: Number of data loader subprocesses.

    Returns:
        Softmax probabilities in dataset and CIFAR-10 class order.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model = load_cnn(MODEL_PATHS["mobilenet_v2"], device)

    transformed_test = copy(test)
    transformed_test.transform = build_cnn_transform()
    loader = DataLoader(
        transformed_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_cuda,
    )
    batches = []
    print(f"Running MobileNetV2 on {device.type.upper()}...", flush=True)
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(
                device,
                non_blocking=use_cuda,
                memory_format=torch.channels_last,
            )
            batches.append(model(images).softmax(dim=1).cpu())
    return torch.cat(batches).numpy()


def save_confusion_matrix(
    counts: NDArray[np.int64], model_name: str, path: Path
) -> None:
    """Render a confusion matrix using the same count scale for both models.

    Args:
        counts: Confusion counts with actual rows and predicted columns.
        model_name: Name to show in the chart title.
        path: Output PNG path.
    """
    import seaborn as sns
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.4, 7.2), layout="constrained")
    axis = figure.subplots()
    sns.heatmap(
        counts,
        ax=axis,
        cmap="Blues",
        vmin=0,
        vmax=int(counts.sum(axis=1).max()),
        annot=True,
        fmt="d",
        square=True,
        linewidths=0.4,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Images", "shrink": 0.85},
    )
    axis.set(
        title=f"{model_name} confusion matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.tick_params(axis="y", labelrotation=0)
    figure.savefig(path, dpi=160)


def save_error_grid(
    test: CIFAR10, cases: list[ErrorCase], model_name: str, path: Path
) -> None:
    """Render the original test images with their incorrect predictions.

    Args:
        test: The official test dataset with no image transform.
        cases: Selected errors in descending confidence order.
        model_name: Name to show in the figure title.
        path: Output PNG path.
    """
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.5, 3.3), layout="constrained")
    figure.suptitle(f"{model_name} highest-confidence errors", fontsize=16)
    if not cases:
        figure.text(0.5, 0.5, "No incorrect predictions", ha="center")
    else:
        axes = figure.subplots(1, len(cases), squeeze=False)[0]
        for axis, case in zip(axes, cases, strict=True):
            image, _ = test[case["index"]]
            axis.imshow(image, interpolation="nearest")
            axis.set_title(
                f"Actual {case['actual']}\n"
                f"Predicted {case['predicted']} · {case['confidence']:.2%}",
                fontsize=10,
            )
            axis.set(xlabel=f"Test image {case['index']}")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.savefig(path, dpi=160)


def save_evaluation(
    test: CIFAR10,
    probabilities: NDArray[np.floating],
    model_id: str,
    model_name: str,
) -> None:
    """Score one model and save its test report and figures.

    Args:
        test: The official test dataset with no image transform.
        probabilities: Probabilities in dataset and CIFAR-10 class order.
        model_id: Stable identifier for filenames and the dashboard.
        model_name: Display name for charts and the terminal.
    """
    targets = np.asarray(test.targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    cases = select_error_cases(targets, probabilities)
    report = {
        "model": model_id,
        "split": "test",
        "samples": len(test),
        **calculate_metrics(targets, predictions),
        "errors": cases,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=RESULTS_DIR) as staging:
        output = Path(staging)
        report_file = output / f"{model_id}_test_metrics.json"
        matrix_file = output / f"{model_id}_confusion_matrix.png"
        errors_file = output / f"{model_id}_error_analysis.png"
        report_file.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        counts = np.asarray(report["confusion_matrix"], dtype=np.int64)
        save_confusion_matrix(counts, model_name, matrix_file)
        save_error_grid(test, cases, model_name, errors_file)
        for path in (matrix_file, errors_file, report_file):
            path.replace(RESULTS_DIR / path.name)

    print(
        f"{model_name} | macro-F1 {report['macro_f1']:.4f} | "
        f"accuracy {report['accuracy']:.2%}",
        flush=True,
    )


def main() -> None:
    """Evaluate both saved checkpoints on the same official test images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Use a positive batch size and nonnegative worker count.")
    for path in MODEL_PATHS.values():
        if not path.is_file():
            parser.error(f"Missing trained model {path}")

    test = CIFAR10(args.data_dir, train=False, download=not args.no_download)
    if len(test) != TEST_SIZE or tuple(test.classes) != CLASS_NAMES:
        raise ValueError("Unexpected CIFAR-10 test size or class order.")
    print("Running HOG + SVM on CPU...", flush=True)
    save_evaluation(test, hog_test_probabilities(test), "hog_svm", "HOG + SVM")
    save_evaluation(
        test,
        cnn_test_probabilities(test, args.batch_size, args.num_workers),
        "mobilenet_v2",
        "MobileNetV2",
    )
    print(f"Test reports and figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
