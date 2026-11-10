"""Generate the ES 335 site's data audit and paired test comparison."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from .dataset import CLASS_NAMES, DEFAULT_DATA_DIR, SEED, load_cifar10_splits
from .evaluate import (
    RESULTS_DIR,
    calculate_metrics,
    cnn_test_probabilities,
    hog_test_probabilities,
)

SITE_DATA = RESULTS_DIR.parent / "docs" / "data" / "study.json"
BOOTSTRAP_REPLICATES = 2000
N_CLASSES = len(CLASS_NAMES)


def audit_split(
    images: NDArray[np.uint8], labels: NDArray[np.int64]
) -> tuple[dict[str, object], list[bytes]]:
    """Summarize one split and fingerprint each raw RGB image.

    Args:
        images: CIFAR-10 image array in sample order.
        labels: Integer class labels in the same order.

    Returns:
        Counts and quality checks, followed by exact pixel fingerprints.
    """
    fingerprints = [sha256(image.tobytes()).digest() for image in images]
    frequencies = Counter(fingerprints)
    return {
        "samples": len(images),
        "shape": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "class_counts": np.bincount(labels, minlength=N_CLASSES).tolist(),
        "duplicate_rows": len(fingerprints) - len(frequencies),
        "constant_images": int(
            np.count_nonzero(np.ptp(images, axis=(1, 2, 3)) == 0)
        ),
    }, fingerprints


def paired_intervals(
    labels: NDArray[np.int64],
    baseline: NDArray[np.int64],
    candidate: NDArray[np.int64],
) -> dict[str, list[float]]:
    """Estimate intervals for paired differences by resampling test images.

    Args:
        labels: Ground-truth labels in test order.
        baseline: HOG + SVM labels in the same order.
        candidate: MobileNetV2 labels in the same order.

    Returns:
        Percentile intervals for MobileNetV2 minus HOG + SVM.
    """
    rng = np.random.default_rng(SEED)
    pairs = (N_CLASSES * labels + baseline, N_CLASSES * labels + candidate)
    differences = np.empty((BOOTSTRAP_REPLICATES, 2))
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(labels), size=len(labels))
        scores = []
        for pair in pairs:
            matrix = np.bincount(
                pair[indices], minlength=N_CLASSES**2
            ).reshape(N_CLASSES, N_CLASSES)
            diagonal = matrix.diagonal()
            support = matrix.sum(axis=0) + matrix.sum(axis=1)
            f1 = np.divide(
                2 * diagonal,
                support,
                out=np.zeros(N_CLASSES),
                where=support > 0,
            )
            scores.append(np.array([diagonal.sum() / len(labels), f1.mean()]))
        differences[replicate] = scores[1] - scores[0]
    accuracy, macro_f1 = np.percentile(
        differences, [2.5, 97.5], axis=0
    ).T
    return {"accuracy": accuracy.tolist(), "macro_f1": macro_f1.tolist()}


def main() -> None:
    """Audit the data, verify published counts, and write the site dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train, validation, test = load_cifar10_splits(
        args.data_dir, download=not args.no_download
    )

    audits = {}
    hashes = {}
    for name, split in (("train", train), ("validation", validation)):
        indices = split.indices
        raw = split.dataset
        audits[name], hashes[name] = audit_split(
            raw.data[indices], np.asarray(raw.targets, dtype=np.int64)[indices]
        )
    targets = np.asarray(test.targets, dtype=np.int64)
    audits["test"], hashes["test"] = audit_split(test.data, targets)
    training_hashes = set(hashes["train"])
    validation_hashes = set(hashes["validation"])
    seen_hashes = training_hashes | validation_hashes
    distinct_test = np.asarray(
        [fingerprint not in seen_hashes for fingerprint in hashes["test"]]
    )
    overlap = {
        "validation_in_train": sum(
            fingerprint in training_hashes
            for fingerprint in hashes["validation"]
        ),
        "test_in_official_train": int((~distinct_test).sum()),
    }
    print(f"Audited exact-pixel matches: {overlap}", flush=True)

    predictions = {
        "hog_svm": hog_test_probabilities(test).argmax(axis=1),
        "mobilenet_v2": cnn_test_probabilities(
            test, args.batch_size, args.num_workers
        ).argmax(axis=1),
    }
    reports = {}
    sensitivity = {}
    for model, labels in predictions.items():
        report_path = RESULTS_DIR / f"{model}_test_metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        observed = calculate_metrics(targets, labels)
        if report["confusion_matrix"] != observed["confusion_matrix"]:
            raise ValueError(f"Regenerate {report_path.name} on this device.")
        reports[model] = report
        filtered = calculate_metrics(
            targets[distinct_test], labels[distinct_test]
        )
        sensitivity[model] = {
            "accuracy": filtered["accuracy"],
            "macro_f1": filtered["macro_f1"],
        }
    training = json.loads(
        (RESULTS_DIR / "hog_svm_metrics.json").read_text(encoding="utf-8")
    )
    data = {
        "class_names": CLASS_NAMES,
        "audit": audits,
        "overlap": overlap,
        "sensitivity": sensitivity,
        "reports": reports,
        "svm_selection": training["selection"],
        "paired_95_intervals": paired_intervals(
            targets, predictions["hog_svm"], predictions["mobilenet_v2"]
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=SITE_DATA.parent) as staging:
        report_path = Path(staging) / SITE_DATA.name
        script_path = Path(staging) / "study-data.js"
        payload = json.dumps(data, indent=2)
        report_path.write_text(payload + "\n", encoding="utf-8")
        script_path.write_text(
            f"window.CIFAR10_STUDY = {payload};\n", encoding="utf-8"
        )
        report_path.replace(SITE_DATA)
        script_path.replace(SITE_DATA.with_name(script_path.name))
    print(f"Saved {SITE_DATA}")


if __name__ == "__main__":
    main()
