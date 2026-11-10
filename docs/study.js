"use strict";

const byId = (id) => document.getElementById(id);
const integer = new Intl.NumberFormat("en-US");
const percent = (value, digits = 2) => (100 * value).toFixed(digits) + "%";
const title = (name) => name.charAt(0).toUpperCase() + name.slice(1);

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function tableRow(values, firstHeader = false) {
  const row = element("tr");
  for (const [index, value] of values.entries()) {
    const cell = element(firstHeader && index === 0 ? "th" : "td", value);
    if (cell.tagName === "TH") cell.scope = "row";
    row.append(cell);
  }
  return row;
}

function statistic(value, label) {
  const node = element("div");
  node.append(element("strong", value), element("span", label));
  return node;
}

function renderAudit(study) {
  const { audit, overlap } = study;
  byId("class-counts").replaceChildren(...study.class_names.map((name, i) =>
    tableRow([
      title(name),
      integer.format(audit.train.class_counts[i]),
      integer.format(audit.validation.class_counts[i]),
      integer.format(audit.test.class_counts[i]),
    ], true)
  ));
  const constant = Object.values(audit).reduce((total, split) => total + split.constant_images, 0);
  byId("audit-summary").textContent =
    `All three splits contain 32 × 32 RGB images stored as unsigned 8-bit pixels. The audit found ${constant} constant images and ${overlap.validation_in_train} validation images with an exact training-pixel match. ${overlap.test_in_official_train} official test images exactly match pixels in the official training partition. The hash check does not detect near-duplicates or labeling errors.`;
  byId("trials").replaceChildren(...study.svm_selection.trials.map((trial) =>
    tableRow([
      `${trial.hog_cell_size} px`,
      String(trial.svm_c),
      trial.macro_f1.toFixed(4),
      `${trial.fit_seconds.toFixed(1)} s`,
    ], true)
  ));
}

function renderMatrix(report) {
  const head = element("thead");
  const headings = element("tr");
  for (const name of ["Actual / predicted", ...report.class_names]) {
    const cell = element("th", name);
    cell.scope = "col";
    headings.append(cell);
  }
  head.append(headings);

  const body = element("tbody");
  for (const [index, counts] of report.confusion_matrix.entries()) {
    const row = tableRow([title(report.class_names[index]), ...counts], true);
    const button = element("button", title(report.class_names[index]));
    button.type = "button";
    button.setAttribute("aria-controls", "class-detail");
    button.addEventListener("click", () => {
      const selector = byId("actual-class");
      selector.value = String(index);
      selector.dispatchEvent(new Event("change"));
    });
    row.cells[0].replaceChildren(button);
    for (const [column, count] of counts.entries()) {
      const cell = row.cells[column + 1];
      const shade = Math.round(100 * Math.sqrt(count / 1000));
      cell.style.backgroundColor = `color-mix(in srgb, #17604b ${shade}%, #f3f8ee)`;
      if (shade > 62) cell.style.color = "#fff";
      if (column === index) cell.classList.add("diagonal");
    }
    body.append(row);
  }
  byId("matrix").replaceChildren(head, body);
}

function renderClass(study) {
  const model = document.querySelector('input[name="model"]:checked').value;
  const report = study.reports[model];
  const index = Number(byId("actual-class").value);
  const otherModel = model === "hog_svm" ? "mobilenet_v2" : "hog_svm";
  const otherName = otherModel === "hog_svm" ? "HOG + SVM" : "MobileNetV2";
  const otherCorrect = study.reports[otherModel].confusion_matrix[index][index];
  for (const [i, row] of Array.from(byId("matrix").tBodies[0].rows).entries()) {
    row.classList.toggle("selected-row", i === index);
    row.cells[0].firstElementChild.setAttribute("aria-pressed", String(i === index));
  }
  const actual = report.class_names[index];
  const scores = report.per_class[actual];
  const counts = report.confusion_matrix[index];
  const mistakes = counts
    .map((count, predicted) => ({ count, predicted }))
    .filter((entry) => entry.predicted !== index && entry.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 3);
  const list = element("ol");
  for (const { count, predicted } of mistakes) {
    list.append(element("li", `${title(report.class_names[predicted])} · ${count} images`));
  }
  byId("class-detail").replaceChildren(
    element("h3", `Actual class · ${title(actual)}`),
    element("p", `${counts[index]} of ${scores.support} correct`),
    element("p", `${otherName} got ${otherCorrect} of the same images correct.`),
    element("p", `Precision ${percent(scores.precision, 1)} · Recall ${percent(scores.recall, 1)}`),
    element("strong", mistakes.length ? "Most frequent wrong labels" : "No mistakes for this class"),
    list
  );
}

function renderResults(study) {
  const selected = document.querySelector('input[name="model"]:checked');
  const report = study.reports[selected.value];
  byId("model-summary").replaceChildren(
    statistic(report.macro_f1.toFixed(4), "Macro-F1"),
    statistic(percent(report.accuracy), "Accuracy"),
    statistic(integer.format(report.samples), "Test images")
  );
  renderMatrix(report);
  renderClass(study);
  byId("errors").replaceChildren(...report.errors.map((item) =>
    tableRow([item.index, title(item.actual), title(item.predicted), percent(item.confidence)], true)
  ));
}

function renderUncertainty(study) {
  const f1 = study.paired_95_intervals.macro_f1.map((x) => x.toFixed(4));
  const accuracy = study.paired_95_intervals.accuracy.map((x) => (x * 100).toFixed(2));
  byId("uncertainty").textContent =
    `Across ${integer.format(study.bootstrap_replicates)} paired bootstrap replicates, the 95% percentile interval for MobileNetV2 minus HOG + SVM was ${f1[0]} to ${f1[1]} macro-F1 and ${accuracy[0]} to ${accuracy[1]} percentage points of accuracy.`;
  const overlap = study.overlap.test_in_official_train;
  byId("sensitivity").textContent = overlap
    ? `Excluding the ${overlap} test images with exact pixel matches in the official training partition gives macro-F1 of ${study.sensitivity.mobilenet_v2.macro_f1.toFixed(4)} for MobileNetV2 and ${study.sensitivity.hog_svm.macro_f1.toFixed(4)} for HOG + SVM.`
    : "The exact-pixel audit found no official test images duplicated in the official training partition. This check cannot rule out visually similar images.";
}

function showRepositoryLink() {
  const suffix = ".github.io";
  if (!location.hostname.endsWith(suffix)) return;
  const owner = location.hostname.slice(0, -suffix.length);
  const repository = location.pathname.split("/").filter(Boolean)[0] || location.hostname;
  const link = byId("repository");
  link.href = `https://github.com/${encodeURIComponent(owner)}/${encodeURIComponent(repository)}`;
  link.hidden = false;
}

function start() {
  showRepositoryLink();
  try {
    const study = window.CIFAR10_STUDY;
    renderAudit(study);
    renderUncertainty(study);
    const mini = study.reports.mobilenet_v2.confusion_matrix.flat().map((count) => {
      const cell = element("span");
      cell.style.backgroundColor = `color-mix(in srgb, #b8e9ba ${Math.sqrt(count / 1000) * 100}%, #285349)`;
      return cell;
    });
    byId("mini-matrix").replaceChildren(...mini);
    const selector = byId("actual-class");
    study.class_names.forEach((name, index) => {
      selector.add(new Option(title(name), String(index), false, name === "cat"));
    });
    selector.addEventListener("change", () => renderClass(study));
    document.querySelectorAll('input[name="model"]').forEach((control) => {
      control.addEventListener("change", () => renderResults(study));
    });
    renderResults(study);
    byId("load-status").hidden = true;
    byId("results-content").hidden = false;
  } catch {
    byId("load-status").textContent = "The benchmark data could not be loaded. Keep the data folder beside index.html and reopen the page.";
  }
}

start();
