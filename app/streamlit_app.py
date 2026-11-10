"""Interactive CIFAR-10 classification and model evaluation."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from torchvision.models.mobilenetv2 import MobileNetV2

# Streamlit executes the entry point from app/ rather than as a package.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.dataset import CLASS_NAMES  # noqa: E402
from src.predict import (  # noqa: E402
    MODEL_PATHS,
    decode_image,
    load_cnn,
    load_svm,
    predict_probabilities,
)

RESULTS_DIR = PROJECT_DIR / "results"
MODEL_NAMES = {"mobilenet_v2": "MobileNetV2", "hog_svm": "HOG + SVM"}
TRAIN_COMMANDS = {
    "mobilenet_v2": "python -m src.train_cnn",
    "hog_svm": "python -m src.train_svm",
}

PAGE_STYLE = """
<style>
.stApp {
    --lab-ink: #17263e;
    --lab-muted: #4f627b;
    --lab-line: #d4deec;
    --lab-surface: #f2f6fc;
    --lab-accent: #315db0;
}
[data-testid="stMainBlockContainer"] {
    max-width: 1100px;
    padding-top: 4rem;
}
.lab-header {
    background: #192d50;
    border-radius: 15px;
    color: #fff;
    padding: clamp(1.35rem, 4vw, 2.55rem);
    position: relative;
    overflow: hidden;
}
.lab-header > div { position: relative; z-index: 1; }
.lab-header h1 {
    color: #fff;
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: clamp(2.05rem, 5vw, 3.45rem);
    font-weight: 740;
    letter-spacing: -.045em;
    line-height: 1.03;
    margin: .4rem 0 .65rem;
    padding: 0;
}
.lab-header p { color: #e1e9f9; margin: 0; max-width: 34rem; }
.lab-kicker {
    font-family: Consolas, monospace;
    font-size: .8rem;
    letter-spacing: .06em;
}
.lab-reticle {
    width: 116px;
    height: 116px;
    position: absolute;
    right: 2.25rem;
    top: 50%;
    transform: translateY(-50%);
    border: 1px solid #779ad2;
    border-radius: 12px;
    background:
      linear-gradient(#4d75b4, #4d75b4) 32px 32px / 24px 24px no-repeat,
      linear-gradient(#4d75b4, #4d75b4) 58px 32px / 24px 24px no-repeat,
      linear-gradient(#4d75b4, #4d75b4) 32px 58px / 24px 24px no-repeat,
      linear-gradient(#9fc0f9, #9fc0f9) 58px 58px / 24px 24px no-repeat;
}
[role="tab"][aria-selected="true"] { color: var(--lab-accent); }
[data-baseweb="tab-highlight"] { background-color: var(--lab-accent); }
.lab-placeholder {
    min-height: 11rem;
    padding: 1.4rem;
    display: grid;
    align-content: center;
    background: var(--lab-surface);
    border: 1px dashed var(--lab-line);
    border-radius: 12px;
    color: var(--lab-muted);
}
.lab-placeholder strong { color: var(--lab-ink); font-size: 1.1rem; }
.lab-placeholder p { margin: .3rem 0 0; }
.lab-result {
    background: var(--lab-surface);
    border: 1px solid var(--lab-line);
    border-radius: 12px;
    color: var(--lab-ink);
    padding: 1.25rem 1.4rem;
}
.lab-result p { margin: 0; }
.lab-result .lab-kicker { color: var(--lab-muted); }
.lab-result .lab-predicted {
    font-size: clamp(1.8rem, 4vw, 2.7rem);
    font-weight: 750;
    letter-spacing: -.04em;
    line-height: 1.12;
    margin: .25rem 0 1.15rem;
}
.lab-result .lab-probabilities {
    list-style: none;
    margin: .5rem 0 0;
    padding: 0;
}
.lab-probabilities li + li { margin-top: .6rem; }
.lab-probability-label {
    display: flex;
    justify-content: space-between;
    gap: .5rem;
    font-variant-numeric: tabular-nums;
}
.lab-track {
    background: var(--lab-line);
    border-radius: 3px;
    display: block;
    height: .48rem;
    margin-top: .28rem;
    overflow: hidden;
}
.lab-track span {
    background: var(--lab-accent);
    border-radius: inherit;
    display: block;
    height: 100%;
}
.st-key-image_input [data-testid="stImage"] img {
    max-height: 350px;
    object-fit: contain;
    background: var(--lab-surface);
    border-radius: 10px;
}
.st-key-evaluation_metrics {
    background: var(--lab-surface);
    border: 1px solid var(--lab-line);
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
.lab-scores {
    width: 100%;
    border-collapse: collapse;
    color: var(--lab-ink);
    font-size: .9rem;
    font-variant-numeric: tabular-nums;
}
.lab-scores th, .lab-scores td {
    padding: .5rem .55rem;
    border-bottom: 1px solid var(--lab-line);
    text-align: right;
}
.lab-scores th:first-child, .lab-scores td:first-child { text-align: left; }
.lab-scores thead { background: var(--lab-surface); }
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding-top: 3rem; }
    .lab-reticle { display: none; }
    .st-key-image_input [data-testid="stImage"] img { max-height: 190px; }
}
@media (prefers-color-scheme: dark) {
    .stApp {
        --lab-ink: #f1f5ff;
        --lab-muted: #bbc9e1;
        --lab-line: #42536e;
        --lab-surface: #202b3d;
        --lab-accent: #86aaff;
    }
}
</style>
"""

HEADER = """
<header class="lab-header">
  <div>
    <span class="lab-kicker">CIFAR-10 image lab</span>
    <h1>Image Classification System for Object Recognition</h1>
    <p>Classify an image into one of ten categories. Switch models to compare
       their predictions and see how each performed on the official test
       set.</p>
  </div>
  <span class="lab-reticle" aria-hidden="true"></span>
</header>
"""

PLACEHOLDER = """
<div class="lab-placeholder">
  <strong>Ready for an image</strong>
  <p>Upload a JPG or PNG to see the predicted class and top three
     probabilities.</p>
</div>
"""


@st.cache_resource(max_entries=4, show_spinner=False)
def load_app_model(
    model_id: str, checkpoint_version: int
) -> CalibratedClassifierCV | MobileNetV2:
    """Load and reuse the selected local checkpoint.

    Args:
        model_id: One of the two model identifiers.
        checkpoint_version: File modification time used as the cache key.

    Returns:
        The loaded classifier in inference mode.
    """
    path = MODEL_PATHS[model_id]
    if model_id == "hog_svm":
        return load_svm(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_cnn(path, device)


def render_prediction(probabilities: NDArray[np.float64]) -> None:
    """Display the winning class and a ranked horizontal bar chart.

    Args:
        probabilities: Class probabilities in CIFAR-10 order.
    """
    top_indices = np.argsort(-probabilities, kind="stable")[:3]
    bars = []
    for index in top_indices:
        name = CLASS_NAMES[index].capitalize()
        percentage = float(probabilities[index]) * 100
        bars.append(
            '<li><span class="lab-probability-label">'
            f"<span>{name}</span><strong>{percentage:.2f}%</strong></span>"
            '<span class="lab-track" aria-hidden="true">'
            f'<span style="width: {percentage:.6f}%"></span></span></li>'
        )
    predicted = CLASS_NAMES[top_indices[0]].capitalize()
    st.html(
        '<section class="lab-result" aria-label="Image prediction">'
        '<p class="lab-kicker">Predicted class</p>'
        f'<p class="lab-predicted">{predicted}</p>'
        '<strong>Top 3 class probabilities</strong>'
        f'<ol class="lab-probabilities">{"".join(bars)}</ol></section>'
    )


def render_classifier(model_id: str) -> None:
    """Show the upload, preview, and live predictions.

    Args:
        model_id: Identifier selected in the shared model control.
    """
    input_column, output_column = st.columns((1, 1), gap="large")
    with input_column, st.container(key="image_input"):
        st.subheader("Input image")
        uploaded = st.file_uploader(
            "Upload a JPG or PNG",
            type=("jpg", "jpeg", "png"),
            max_upload_size=12,
            key="image_upload",
        )
        if uploaded is None:
            with output_column:
                st.subheader("Prediction")
                st.html(PLACEHOLDER)
            return
        try:
            image = decode_image(uploaded.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        preview = image.copy()
        preview.thumbnail((640, 640))
        st.image(preview, width="stretch")
        st.caption(f"{image.width:,} × {image.height:,} pixels")

    with output_column:
        st.subheader("Prediction")
        try:
            path = MODEL_PATHS[model_id]
            with st.spinner(f"Classifying with {MODEL_NAMES[model_id]}..."):
                model = load_app_model(model_id, path.stat().st_mtime_ns)
                probabilities = predict_probabilities(model, image)
        except FileNotFoundError:
            st.warning("Train this model to enable predictions.")
            st.code(TRAIN_COMMANDS[model_id], language="bash")
            return
        except (
            OSError,
            EOFError,
            ValueError,
            RuntimeError,
            pickle.UnpicklingError,
        ):
            st.error("This model could not run. Check its saved checkpoint.")
            return
        render_prediction(probabilities)


def render_evaluation(model_id: str) -> None:
    """Display the saved test report for the selected model.

    Args:
        model_id: Identifier selected in the shared model control.
    """
    path = RESULTS_DIR / f"{model_id}_test_metrics.json"
    st.subheader(f"{MODEL_NAMES[model_id]} on CIFAR-10")
    st.caption("Official held-out test set")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.info("Run the evaluation to generate the test report.")
        st.code("python -m src.evaluate", language="bash")
        return
    except (OSError, ValueError):
        st.error("The test report could not be read. Run evaluation again.")
        return

    with st.container(key="evaluation_metrics"):
        accuracy, f1, samples = st.columns(3)
        accuracy.metric("Accuracy", f"{report['accuracy']:.2%}")
        f1.metric("Macro-F1", f"{report['macro_f1']:.4f}")
        samples.metric("Test images", f"{report['samples']:,}")

    matrix_column, class_column = st.columns((1.7, 1), gap="large")
    with matrix_column:
        st.subheader("Confusion matrix")
        matrix_path = RESULTS_DIR / f"{model_id}_confusion_matrix.png"
        if matrix_path.is_file():
            st.image(str(matrix_path), width="stretch")
        else:
            st.info("Run evaluation again to generate the confusion matrix.")
    with class_column:
        st.subheader("Per-class scores")
        scores = pd.DataFrame.from_dict(report["per_class"], orient="index")
        scores = scores.loc[list(CLASS_NAMES), ["precision", "recall"]]
        scores.index.name = "Class"
        scores.columns = ["Precision", "Recall"]
        st.html(
            scores.reset_index().to_html(
                index=False,
                classes="lab-scores",
                border=0,
                float_format="{:.1%}".format,
            )
        )


def main() -> None:
    """Render the classifier and evaluation tabs with one model selector."""
    st.set_page_config(
        page_title="Image Classification System for Object Recognition",
        layout="wide",
    )
    st.html(PAGE_STYLE)
    st.html(HEADER)
    model_id = st.radio(
        "Model",
        options=tuple(MODEL_NAMES),
        format_func=MODEL_NAMES.__getitem__,
        horizontal=True,
        key="selected_model",
    )
    classifier, evaluation = st.tabs(("Classify image", "Test results"))
    with classifier:
        render_classifier(model_id)
    with evaluation:
        render_evaluation(model_id)


if __name__ == "__main__":
    main()
