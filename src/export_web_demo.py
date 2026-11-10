"""Export the trained web model and an illustrated CIFAR-10 test gallery."""

import argparse
import gzip
import json
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from shutil import copyfile

import numpy as np
import onnx
import torch
from torchvision.datasets import CIFAR10

from .dataset import CLASS_NAMES, CNN_IMAGE_SIZE, DEFAULT_DATA_DIR
from .predict import MODEL_PATHS, load_cnn, load_svm, predict_probabilities

DOCS = Path(__file__).resolve().parents[1] / "docs"
SAMPLE_INDICES = (8, 10, 1382, 29, 48, 23)


def encode_compressed(payload: bytes) -> str:
    """Return a deterministic gzip payload encoded for a browser script.

    Args:
        payload: Model or WebAssembly bytes.

    Returns:
        Base64 text containing the compressed bytes.
    """
    return b64encode(gzip.compress(payload, mtime=0)).decode("ascii")


def export_runtime(dist: Path) -> None:
    """Package the pinned ONNX Runtime Web assets for offline use.

    Args:
        dist: The dist directory from onnxruntime-web 1.22.0.
    """
    vendor = DOCS / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    copyfile(dist / "ort.wasm.min.js", vendor / "ort.wasm.min.js")
    stem = dist / "ort-wasm-simd-threaded"
    runtime = {
        "loader": stem.with_suffix(".mjs").read_text(encoding="utf-8"),
        "wasm": encode_compressed(stem.with_suffix(".wasm").read_bytes()),
    }
    (vendor / "ort-assets.js").write_text(
        f"window.CIFAR10_ORT_ASSETS = {json.dumps(runtime)};\n",
        encoding="utf-8",
    )


def main() -> None:
    """Save the ONNX model and exact checkpoint scores for six test images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--refined", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(4)
    test = CIFAR10(
        args.data_dir, train=False, download=not args.no_download
    )
    device = torch.device("cpu")
    cnn_path = MODEL_PATHS["mobilenet_v2"]
    refinement = None
    if args.refined:
        cnn_path = cnn_path.with_name("mobilenet_v2_refined.pth")
        report_path = DOCS.parent / "results" / "mobilenet_v2_refinement.json"
        refinement = json.loads(report_path.read_text(encoding="utf-8"))
    cnn = load_cnn(cnn_path, device)
    svm = load_svm(MODEL_PATHS["hog_svm"])
    models = {"mobilenet_v2": cnn, "hog_svm": svm}
    samples = []
    for index in SAMPLE_INDICES:
        image, label = test[index]
        png = BytesIO()
        image.save(png, format="PNG", optimize=True)
        scores = {}
        for name, model in models.items():
            probabilities = predict_probabilities(model, image)
            top = np.argsort(-probabilities, kind="stable")[:3]
            scores[name] = [
                [CLASS_NAMES[i], float(probabilities[i])] for i in top
            ]
        samples.append(
            {
                "index": index,
                "actual": CLASS_NAMES[label],
                "png": b64encode(png.getvalue()).decode("ascii"),
                "scores": scores,
            }
        )

    data_dir = DOCS / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "classes": CLASS_NAMES,
        "samples": samples,
        "refinement": refinement,
    })
    (data_dir / "demo-samples.js").write_text(
        f"window.CIFAR10_DEMO = {payload};\n", encoding="utf-8"
    )

    export_buffer = BytesIO()
    dummy = torch.zeros(1, 3, CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)
    torch.onnx.export(
        cnn,
        dummy,
        export_buffer,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    model_bytes = export_buffer.getvalue()
    onnx.checker.check_model(model_bytes)
    (data_dir / "mobilenet-model.js").write_text(
        f'window.CIFAR10_ONNX = "{encode_compressed(model_bytes)}";\n',
        encoding="utf-8",
    )
    if args.runtime_dir is not None:
        export_runtime(args.runtime_dir)


if __name__ == "__main__":
    main()
