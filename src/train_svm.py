"""Train, calibrate, and evaluate the HOG + linear SVM baseline."""

import argparse
import json
import pickle
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import Subset

from .dataset import DEFAULT_DATA_DIR, SEED, load_cifar10_splits
from .evaluate import calculate_metrics
from .features import extract_hog_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "hog_svm.pkl"
METRICS_PATH = PROJECT_DIR / "results" / "hog_svm_metrics.json"
SEARCH_SPACE = ((8, (1.0, 0.01)), (4, (0.01, 0.1)))
SELECTION_SIZE = 8_000


def seed_everything() -> None:
    """Set the fixed project seed for Python, NumPy, and PyTorch."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def select_and_fit_svm(
    training_data: Subset,
) -> tuple[LinearSVC, int, dict[str, object]]:
    """Choose HOG and SVM settings on a stratified training-only holdout.

    Args:
        training_data: The 40,000 training images and their labels.

    Returns:
        The winner refit on all training images, its HOG cell size, and
        search results with wall-clock timings.
    """
    start = perf_counter()
    labels = np.asarray(training_data.dataset.targets)[training_data.indices]
    fit_indices, selection_indices = train_test_split(
        np.arange(len(labels)),
        test_size=SELECTION_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    fit_labels = labels[fit_indices]
    selection_labels = labels[selection_indices]
    trials = []
    best_score = -1.0
    best_features = None
    best_cell_size = 8
    best_c = 1.0

    for cell_size, penalties in SEARCH_SPACE:
        print(f"Extracting HOG with {cell_size}px cells...", flush=True)
        features, _ = extract_hog_dataset(training_data, cell_size=cell_size)
        fit_features = features[fit_indices]
        selection_features = features[selection_indices]
        for penalty in penalties:
            svm = LinearSVC(
                C=penalty, dual=False, max_iter=10_000, random_state=SEED
            )
            fit_start = perf_counter()
            svm.fit(fit_features, fit_labels)
            fit_seconds = perf_counter() - fit_start
            score = float(
                f1_score(
                    selection_labels,
                    svm.predict(selection_features),
                    average="macro",
                    zero_division=0,
                )
            )
            trials.append(
                {
                    "hog_cell_size": cell_size,
                    "svm_c": penalty,
                    "macro_f1": score,
                    "fit_seconds": fit_seconds,
                }
            )
            print(
                f"  C={penalty:g}, selection macro-F1 {score:.4f} "
                f"({fit_seconds:.1f}s fit)",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_features = features
                best_cell_size = cell_size
                best_c = penalty
        del fit_features, selection_features

    search_seconds = perf_counter() - start
    print(
        f"Refitting {best_cell_size}px HOG + SVM (C={best_c:g}) on all "
        "40,000 training images...",
        flush=True,
    )
    svm = LinearSVC(C=best_c, dual=False, max_iter=10_000, random_state=SEED)
    start = perf_counter()
    svm.fit(best_features, labels)
    selection = {
        "holdout_size": SELECTION_SIZE,
        "trials": trials,
        "search_seconds": search_seconds,
        "refit_seconds": perf_counter() - start,
    }
    return svm, best_cell_size, selection


def save_artifacts(
    classifier: CalibratedClassifierCV, metrics: dict[str, object]
) -> None:
    """Stage the fitted classifier and metrics before replacing output files.

    Args:
        classifier: The fitted and calibrated SVM.
        metrics: JSON-serializable test metrics and run details.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODEL_PATH.parent) as model_temp:
        with TemporaryDirectory(dir=METRICS_PATH.parent) as metrics_temp:
            model_file = Path(model_temp) / MODEL_PATH.name
            metrics_file = Path(metrics_temp) / METRICS_PATH.name
            with model_file.open("wb") as output:
                pickle.dump(
                    classifier, output, protocol=pickle.HIGHEST_PROTOCOL
                )
            metrics_file.write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            model_file.replace(MODEL_PATH)
            metrics_file.replace(METRICS_PATH)


def main() -> None:
    """Run the baseline on the fixed CIFAR-10 splits and save its results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    seed_everything()
    train, validation, test = load_cifar10_splits(
        args.data_dir, download=not args.no_download
    )

    svm, cell_size, selection = select_and_fit_svm(train)
    timings = {
        "model_selection": selection["search_seconds"],
        "fit": selection["refit_seconds"],
    }

    print("Calibrating probabilities on validation data...", flush=True)
    start = perf_counter()
    validation_features, validation_labels = extract_hog_dataset(
        validation, cell_size=cell_size
    )
    timings["validation_hog"] = perf_counter() - start
    start = perf_counter()
    classifier = CalibratedClassifierCV(FrozenEstimator(svm), method="sigmoid")
    classifier.fit(validation_features, validation_labels)
    classifier.hog_cell_size_ = cell_size
    timings["calibration"] = perf_counter() - start
    del validation_features, validation_labels

    print("Evaluating on the official test split...", flush=True)
    start = perf_counter()
    test_features, test_labels = extract_hog_dataset(test, cell_size=cell_size)
    timings["test_hog"] = perf_counter() - start
    start = perf_counter()
    predictions = classifier.predict(test_features)
    timings["test_prediction"] = perf_counter() - start

    metrics = {
        "model": "hog_svm",
        "split": "test",
        "seed": SEED,
        **calculate_metrics(test_labels, predictions),
        "hog_cell_size": cell_size,
        "feature_count": int(svm.n_features_in_),
        "svm_c": svm.C,
        "selection": selection,
        "timings_seconds": timings,
    }
    save_artifacts(classifier, metrics)
    print(
        f"Test macro-F1 {metrics['macro_f1']:.4f} | "
        f"Accuracy {metrics['accuracy']:.4f}"
    )
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
