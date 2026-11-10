"use strict";

(() => {
  const sampleList = document.getElementById("demo-samples");
  const fileInput = document.getElementById("demo-file");
  const modelControl = document.getElementById("demo-model");
  const preview = document.getElementById("demo-image");
  const caption = document.getElementById("demo-image-caption");
  const status = document.getElementById("demo-status");
  const prediction = document.getElementById("demo-prediction");
  const demo = window.CIFAR10_DEMO;
  const refinement = demo?.refinement;
  const cnnName = refinement && refinement.selected !== "original"
    ? "Refined MobileNetV2" : "MobileNetV2";
  const modelNames = { mobilenet_v2: cnnName, hog_svm: "HOG + SVM" };
  const size = 224;
  let selectedSample = null;
  let request = 0;
  let previewUrl = null;
  let sessionPromise;

  function text(tag, value, className) {
    const node = document.createElement(tag);
    node.textContent = value;
    if (className) node.className = className;
    return node;
  }

  function title(label) {
    return label.charAt(0).toUpperCase() + label.slice(1);
  }

  function setStatus(message, error = false) {
    status.textContent = message;
    status.classList.toggle("is-error", error);
  }

  function setPreview(src, alt, description, sample = false) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = src.startsWith("blob:") ? src : null;
    preview.hidden = !src;
    if (src) preview.src = src;
    else preview.removeAttribute("src");
    preview.alt = alt;
    preview.classList.toggle("is-sample", sample);
    caption.textContent = description;
  }

  function renderPrediction(scores, actual) {
    const predicted = scores[0][0];
    const outcome = actual
      ? `Actual label: ${title(actual)} · ${predicted === actual ? "Correct" : "Incorrect"}`
      : "Prediction from your uploaded photo";
    const rows = document.createElement("ol");
    rows.className = "demo-scores";
    rows.setAttribute("aria-label", "Top three class probabilities");
    for (const [label, probability] of scores) {
      const row = document.createElement("li");
      const value = 100 * probability;
      const line = text("div", "", "demo-score-label");
      line.append(text("span", title(label)), text("strong", `${value.toFixed(2)}%`));
      const track = text("div", "", "demo-score-track");
      track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("span");
      fill.style.width = `${value}%`;
      track.append(fill);
      row.append(line, track);
      rows.append(row);
    }
    prediction.replaceChildren(
      text("h3", title(predicted), "demo-predicted"),
      text("p", outcome, "demo-outcome"),
      text("strong", "Top three probabilities"),
      rows
    );
  }

  function renderSample() {
    if (!selectedSample) return;
    const model = modelControl.querySelector("input:checked").value;
    setStatus(`Saved prediction · ${modelNames[model]}`);
    renderPrediction(selectedSample.scores[model], selectedSample.actual);
  }

  function selectSample(sample, button) {
    request += 1;
    selectedSample = sample;
    modelControl.hidden = false;
    for (const item of sampleList.children) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    setPreview(
      `data:image/png;base64,${sample.png}`,
      `Held-out CIFAR-10 image labeled ${sample.actual}`,
      `CIFAR-10 test image ${sample.index} · 32 × 32 pixels`,
      true
    );
    renderSample();
  }

  function loadScript(path) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = path;
      script.onload = () => {
        script.remove();
        resolve();
      };
      script.onerror = () => {
        script.remove();
        reject(new Error("Could not load " + path));
      };
      document.head.append(script);
    });
  }

  async function inflate(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function initializeModel() {
    await Promise.all([
      loadScript("./vendor/ort.wasm.min.js"),
      loadScript("./vendor/ort-assets.js"),
      loadScript("./data/mobilenet-model.js"),
    ]);
    const runtime = window.ort;
    const assets = window.CIFAR10_ORT_ASSETS;
    const [wasm, model] = await Promise.all([
      inflate(assets.wasm), inflate(window.CIFAR10_ONNX),
    ]);
    delete window.CIFAR10_ORT_ASSETS;
    delete window.CIFAR10_ONNX;
    const loaderUrl = URL.createObjectURL(new Blob([assets.loader], { type: "text/javascript" }));
    const wasmUrl = URL.createObjectURL(new Blob([wasm], { type: "application/wasm" }));
    runtime.env.wasm.numThreads = 1;
    runtime.env.wasm.initTimeout = 15000;
    runtime.env.wasm.wasmPaths = { mjs: loaderUrl, wasm: wasmUrl };
    try {
      return await runtime.InferenceSession.create(model, { executionProviders: ["wasm"] });
    } finally {
      URL.revokeObjectURL(loaderUrl);
      URL.revokeObjectURL(wasmUrl);
    }
  }

  function getSession() {
    if (!sessionPromise) {
      sessionPromise = initializeModel().catch((error) => {
        sessionPromise = undefined;
        throw error;
      });
    }
    return sessionPromise;
  }

  function prepareInput(bitmap) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const context = canvas.getContext("2d", { alpha: false });
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, size, size);
    const rgba = context.getImageData(0, 0, size, size).data;
    const pixels = size * size;
    const data = new Float32Array(3 * pixels);
    const means = [0.485, 0.456, 0.406];
    const deviations = [0.229, 0.224, 0.225];
    for (let channel = 0; channel < 3; channel++) {
      for (let i = 0; i < pixels; i++) {
        data[channel * pixels + i] = (rgba[4 * i + channel] / 255 - means[channel]) / deviations[channel];
      }
    }
    return data;
  }

  function rankLogits(logits) {
    const largest = Math.max(...logits);
    const weights = Array.from(logits, (value) => Math.exp(value - largest));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return weights.map((weight, i) => [demo.classes[i], weight / total])
      .sort((a, b) => b[1] - a[1]).slice(0, 3);
  }

  async function decodeUpload(file) {
    if (file.size > 12 * 1024 * 1024) throw new Error("Choose a JPG or PNG under 12 MB.");
    const bytes = new Uint8Array(await file.slice(0, 8).arrayBuffer());
    const png = [137, 80, 78, 71, 13, 10, 26, 10];
    const isPng = png.every((byte, i) => bytes[i] === byte);
    const isJpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    if (!isPng && !isJpeg) throw new Error("This file could not be read as a JPG or PNG.");
    let bitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch {
      throw new Error("This file could not be read as a JPG or PNG.");
    }
    if (bitmap.width * bitmap.height > 24_000_000) {
      bitmap.close();
      throw new Error("Choose an image under 24 megapixels.");
    }
    return bitmap;
  }

  async function classifyUpload(file) {
    const current = ++request;
    selectedSample = null;
    modelControl.hidden = true;
    for (const button of sampleList.children) button.setAttribute("aria-pressed", "false");
    prediction.replaceChildren();
    setPreview("", "", "Your photo will appear here.");
    setStatus("Reading your photo…");

    let bitmap;
    try {
      bitmap = await decodeUpload(file);
    } catch (error) {
      if (current === request) setStatus(error.message, true);
      return;
    }
    try {
      if (current !== request) return;
      setPreview(URL.createObjectURL(file), "Your uploaded photo", `${file.name} · ${bitmap.width} × ${bitmap.height} pixels`);
      const data = prepareInput(bitmap);
      setStatus(sessionPromise ? `Classifying with ${cnnName}…` : `Starting ${cnnName} in your browser…`);
      const session = await getSession();
      if (current !== request) return;
      const tensor = new window.ort.Tensor("float32", data, [1, 3, size, size]);
      const output = await session.run({ image: tensor });
      if (current !== request) return;
      setStatus(`Live prediction · ${cnnName}`);
      renderPrediction(rankLogits(output.logits.data));
    } catch (error) {
      console.error("Browser inference failed", error);
      if (current === request) {
        setStatus("Live classification could not start. Try a current browser, or pick a test image above.", true);
      }
    } finally {
      bitmap.close();
    }
  }

  if (!demo?.samples.length) {
    fileInput.disabled = true;
    setStatus("The demo data could not be loaded. Keep the data folder beside index.html.", true);
    return;
  }

  document.getElementById("demo-cnn-label").textContent = cnnName;
  if (refinement) {
    const labels = {
      original: "Original MobileNetV2",
      cifar10_only: "CIFAR-10 fine-tuning",
      cifar100_transfer: "CIFAR-100 transfer + fine-tuning",
    };
    const audit = refinement.cifar100_audit;
    document.getElementById("refinement-summary").textContent =
      `The auxiliary stage used ${audit.used.toLocaleString("en-US")} CIFAR-100 training images after overlap screening. The demo uses ${labels[refinement.selected]}, selected on validation before test scoring.`;
    const rows = Object.entries(refinement.runs).map(([name, run]) => {
      const row = document.createElement("tr");
      const heading = text("th", labels[name]);
      heading.scope = "row";
      row.append(
        heading,
        text("td", run.validation.macro_f1.toFixed(4)),
        text("td", `${(100 * run.cifar10_test.accuracy).toFixed(2)}%`),
        text("td", `${(100 * run.cifar10_1_v6.accuracy).toFixed(2)}%`)
      );
      return row;
    });
    document.getElementById("refinement-runs").replaceChildren(...rows);
    document.getElementById("refinement").hidden = false;
  }

  for (const sample of demo.samples) {
    const button = document.createElement("button");
    button.type = "button";
    const thumbnail = document.createElement("img");
    thumbnail.alt = "";
    thumbnail.src = `data:image/png;base64,${sample.png}`;
    button.append(thumbnail, text("span", title(sample.actual)));
    button.addEventListener("click", () => selectSample(sample, button));
    sampleList.append(button);
  }
  modelControl.addEventListener("change", renderSample);
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    fileInput.value = "";
    if (file) classifyUpload(file);
  });
  selectSample(demo.samples[0], sampleList.firstElementChild);
})();
