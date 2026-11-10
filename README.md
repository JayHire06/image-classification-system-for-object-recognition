# Image Classification System for Object Recognition

This project compares a HOG + SVM baseline with a MobileNetV2 transfer learning
model on CIFAR-10 using shared, seeded dataset splits.

## ES 335 project website

The static site in [`docs/`](docs/) is the course submission deliverable.
It explains the question, data audit, preprocessing, methods, evaluation,
results, limitations, and reproduction steps. Its result explorer compares
both models and lets visitors inspect errors for an actual class. The
browser demo offers six held-out images with saved predictions from both
models. The live upload option runs the exported MobileNetV2 in the browser.
The separate Streamlit app also supports live HOG + SVM uploads.

Open `docs/index.html` directly to use the interactive explorer offline,
or preview the site from this directory with
`python -m http.server 8502 --bind 127.0.0.1 --directory docs`, then open
[the local preview](http://127.0.0.1:8502). The site reads a checked-in
aggregate report and requires no Python service when hosted.

To publish, put the contents of this project directory at the root of the
course repository. In **Settings → Pages**, choose **Deploy from a branch**,
select the branch containing the project, and select **/docs**. Check the
published URL and submit it together with the repository URL.
[GitHub Pages publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

After training both models, regenerate the static data with the same
inference settings for both commands:

```powershell
python -m src.evaluate --batch-size 128 --num-workers 2
python -m src.course_report --batch-size 128 --num-workers 2
```

The report generator verifies the saved confusion counts against predictions
from both checkpoints. It audits split-level class counts and exact pixel
overlaps, and uses 2,000 paired bootstrap replicates to estimate uncertainty
in the model difference. The site includes aggregate reports, test indices,
and six attributed 32 × 32 CIFAR-10 test thumbnails. Keep the raw dataset,
any secrets, and generated error-image grids out of the published repository.

To rebuild the browser checkpoint and gallery after retraining, install
`onnx==1.20.1` and run `python -m src.export_web_demo` with the same data
directory. The bundled ONNX Runtime Web 1.22.0 files are loaded on the
first upload. The optional `--runtime-dir` flag accepts the `dist` folder
from that version of the official npm package. The browser applies the same
input size and ImageNet normalization as Python, with browser-side image
resampling. [ONNX Runtime Web deployment documentation](https://onnxruntime.ai/docs/tutorials/web/deploy.html)

## Set up and check the data pipeline

From the `object-recognition-project` directory, using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.dataset
```

The last command downloads CIFAR-10 on the first run. It prints the split sizes
and checks that the training loader can produce a batch. Use
`--data-dir PATH --no-download` to check an existing local copy.

If the default download fails on Windows, fetch the same archive from the
mirror used by Keras and rerun the check. Torchvision verifies the archive
before extracting it. [Keras CIFAR-10 source](https://github.com/keras-team/keras/blob/master/keras/src/datasets/cifar10.py)

```powershell
New-Item -ItemType Directory -Force data | Out-Null
$cifarUrl = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz"
curl.exe --fail --location --output data/cifar-10-python.tar.gz $cifarUrl
.\.venv\Scripts\python.exe -m src.dataset
```

| Split | Images | Source |
| --- | ---: | --- |
| Training | 40,000 | Official training set, seed 42 |
| Validation | 10,000 | Remaining official training images |
| Test | 10,000 | Official test set, original order |

`load_cifar10_splits()` returns raw 32×32 RGB PIL images by default, so the
classical pipeline can extract HOG features. `build_dataloaders()` uses the same
split membership with 224×224 tensors and ImageNet normalization for MobileNetV2.
Only the training loader shuffles and applies random augmentation.

Use the same package versions, hardware, and loader settings when comparing
repeat training runs. PyTorch documents reproducibility limits across releases
and platforms. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## Train and evaluate HOG + SVM

```powershell
.\.venv\Scripts\python.exe -m src.train_svm
```

For an existing dataset outside the project directory, pass
`--data-dir PATH --no-download`.

The training command compares four combinations of HOG cell size and SVM
regularization. It fits each candidate on 32,000 training images and
ranks them by macro-F1 on a stratified 8,000-image holdout from the training
split. It then refits the winner on all 40,000 training images, calibrates
probabilities on the 10,000 validation images, and scores the official
10,000-image test set.

- `models/hog_svm.pkl` stores the fitted classifier with `predict_proba()`
  and the HOG cell size used for its input features.
- `results/hog_svm_metrics.json` records test macro-F1, accuracy, per-class
  precision and recall, confusion counts, candidate scores, and timing. The
  confusion counts use true labels for rows and predicted labels for columns.

To classify a PIL image with the saved model, use its stored HOG cell size.
The first baseline checkpoint used 8-pixel cells. Probability columns follow
`model.classes_`.

```python
from src.features import extract_hog

cell_size = getattr(model, "hog_cell_size_", 8)
features = extract_hog(image, cell_size=cell_size)
probabilities = model.predict_proba(features[None, :])[0]
```

## Train MobileNetV2

```powershell
.\.venv\Scripts\python.exe -m src.train_cnn
```

The default run starts from torchvision's ImageNet V2 weights. It trains
the new 10-class head for two epochs with the backbone frozen, then
fine-tunes the head and the last four backbone blocks for three epochs at
a lower learning rate. Frozen blocks keep their original batch normalization
statistics. Training uses Adam, cross-entropy loss, and training-only
random crops and flips.

The script uses CUDA with mixed precision when available and CPU otherwise.
To enable CUDA on a compatible Windows NVIDIA setup, install the matching
builds after the base requirements. [PyTorch CUDA build options](https://pytorch.org/get-started/previous-versions/#v2100)

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Use `--data-dir PATH --no-download` for an existing dataset. Increase
`--num-workers` to preprocess images in parallel. See `--help` for epoch,
batch size, and learning rate options. The first training run also needs the
ImageNet weights, which torchvision downloads into its model cache.

The saved checkpoint was trained on an RTX 4060 Laptop GPU using
`--batch-size 128 --num-workers 2` and the default epoch and learning rate
settings.

- `models/mobilenet_v2.pth` stores the weights from the epoch with the best
  validation macro-F1 across both stages.
- `results/mobilenet_v2_history.json` records the training settings and
  per-epoch training and validation metrics.

To restore the checkpoint, construct `build_mobilenet_v2(pretrained=False)`
and call `load_state_dict(torch.load(path, map_location="cpu", weights_only=True))`.

## Refine MobileNetV2 with extra training data

`src.refine_cnn` compares two candidates starting from the original trained
MobileNetV2. Both fine-tune the last ten backbone blocks with the same
CIFAR-10 data, RandAugment, label smoothing, AdamW, and a cosine learning-rate
schedule. One candidate first adapts the backbone to the 100 CIFAR-100 labels
with a temporary head. It then restores the original ten-class head.

The experiment screens auxiliary training images for exact RGB matches and
matching perceptual hashes against CIFAR-10 and CIFAR-10.1. These checks can
miss near-duplicates. The best checkpoint is selected by CIFAR-10 validation
macro-F1 before final evaluation on the official test set and CIFAR-10.1 v6.
The official CIFAR-10 test set had already been scored in the initial study.
CIFAR-10.1 provides a separate benchmark for this experiment.

Download the external test files and run the refinement from the project
directory. Torchvision downloads and checks CIFAR-100 if needed.

```powershell
$source = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets"
New-Item -ItemType Directory -Force data/cifar10.1 | Out-Null
curl.exe --fail --location --output data/cifar10.1/cifar10.1_v6_data.npy "$source/cifar10.1_v6_data.npy"
curl.exe --fail --location --output data/cifar10.1/cifar10.1_v6_labels.npy "$source/cifar10.1_v6_labels.npy"
.\.venv\Scripts\python.exe -m src.refine_cnn
```

If the CIFAR-100 download fails, put the
[checksum-verified Keras mirror archive](https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz)
in `data/` as `cifar-100-python.tar.gz`. Leave downloads enabled so
torchvision can extract the archive.
[CIFAR dataset source](https://www.cs.toronto.edu/~kriz/cifar.html) · [CIFAR-10.1 data and paper](https://github.com/modestyachts/CIFAR-10.1)

The selected weights and full comparison are saved as
`models/mobilenet_v2_refined.pth` and
`results/mobilenet_v2_refinement.json`. Run
`python -m src.export_web_demo --refined` to use that checkpoint for
browser uploads and rebuild the held-out gallery scores. The site shows
the refinement results separately from the original benchmark comparison.

## Compare both models on the official test set

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

The evaluation command loads both saved models and scores the same 10,000
test images in CIFAR-10 order. Use `--data-dir PATH --no-download` for an
existing dataset. MobileNetV2 uses full-precision inference on CUDA when
available, with CPU as the fallback.

For each model, `results/` contains a `*_test_metrics.json` report with
accuracy, macro-F1, per-class precision and recall, and confusion counts.
The `*_confusion_matrix.png` plots use the same count scale, with actual
classes in rows and predicted classes in columns. The
`*_error_analysis.png` grids show the five highest-confidence errors,
ranked by the predicted class probability with ties in original test order.

## Run the Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. Upload a JPG or PNG, then use the
model selector to compare the predicted class and the top three probabilities.
The same selector controls the **Test results** tab, which shows the saved
accuracy, macro-F1, confusion matrix, and per-class scores. The app reads the
saved checkpoints and evaluation reports directly, so it can run without a
local copy of the CIFAR-10 dataset.
