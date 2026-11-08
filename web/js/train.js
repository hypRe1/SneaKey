// Section 3: train a key classifier on chosen recordings and compare it with the active model on held-out ones.
import { $, api, el, keyLabel, pct, roleBadge, showError } from "./common.js";
import { refreshSessions, state } from "./app.js";

const KIND_INFO = {
  svm: "RBF support-vector machine on 464 hand-made features per press (64 band energies + a 10-step log-mel patch). Trains in seconds to minutes and copes well with little data. In the original project's large-data comparison it reached 53% validation key accuracy against the CNN's 60%.",
  lda: "Shrinkage linear discriminant analysis on the same features as the SVM. Trains in seconds; in the original comparison it matched the SVM (53%). A good quick baseline.",
  cnn: "The shipped convolutional network and recipe (log-mel input, ±3 ms shift augmentation, 30 epochs). Best with plenty of data (60% vs 53% on ~20k presses) but slowest: minutes for a few thousand presses.",
};
const KEYS_SHOWN = "abcdefghijklmnopqrstuvwxyz".split("").map((c) => `Key${c.toUpperCase()}`).concat(["Space", "Comma", "Period", "Backspace"]);

let kind = "svm";
let use = {};             // session id -> "train" | "validation" | "none"
let poller = null;
let initialised = false;
let lastShownResult = null;
let micChosen = false;

const shown = (s) => $("trainMic").value === "all" || s.microphone === $("trainMic").value;

export async function enter() {
  if (!initialised) init();
  const usable = state.sessions.filter((s) => s.has_audio);
  document.querySelector('[data-empty="train"]').classList.toggle("hidden", usable.length > 0);
  document.querySelector('[data-content="train"]').classList.toggle("hidden", usable.length === 0);
  if (!micChosen) {        // default: phone recordings if there are any, since those are what the demo uses
    $("trainMic").value = usable.some((s) => s.microphone === "phone") ? "phone" : "all";
    micChosen = true;
  }
  for (const s of usable) if (!(s.id in use)) use[s.id] = selectable(s) && shown(s) ? "train" : "none";
  if (!Object.values(use).includes("validation")) autoSplit(false);
  renderSessions();
  renderModels();
  poll();
}

export function leave() { clearTimeout(poller); poller = null; }

function init() {
  initialised = true;
  for (const button of $("kind").children) button.addEventListener("click", () => setKind(button.dataset.kind));
  $("autoSplit").addEventListener("click", () => autoSplit(true));
  $("minPresses").addEventListener("change", renderSummary);
  $("trainMic").addEventListener("change", () => { autoSplit(true); });
  $("startTraining").addEventListener("click", start);
  $("cancelTraining").addEventListener("click", () => api("api/training/cancel", { method: "POST" }).catch((e) => showError(e.message)));
  setKind("svm");
}

const selectable = (s) => s.role !== "locked" && s.role !== "test";

function setKind(value) {
  kind = value;
  for (const button of $("kind").children) button.classList.toggle("on", button.dataset.kind === kind);
  for (const label of document.querySelectorAll("[data-param]")) label.classList.toggle("hidden", label.dataset.param !== kind);
  $("kindInfo").textContent = KIND_INFO[kind];
}

// Newest ~20% of usable recordings (at least one, if there are two or more) for validation; the rest for training.
function autoSplit(render = true) {
  for (const s of state.sessions) if (!shown(s)) use[s.id] = "none";
  const usable = state.sessions.filter((s) => s.has_audio && selectable(s) && shown(s)).sort((a, b) => a.id.localeCompare(b.id));
  const nValidation = usable.length >= 2 ? Math.max(1, Math.round(usable.length * 0.2)) : 0;
  usable.forEach((s, i) => { use[s.id] = i >= usable.length - nValidation ? "validation" : "train"; });
  if (render) renderSessions();
}

function renderSessions() {
  const usable = state.sessions.filter((s) => s.has_audio && shown(s));
  $("trainSessions").replaceChildren(...usable.map((s) => {
    const select = el("select", { disabled: !selectable(s), onchange: (e) => { use[s.id] = e.target.value; renderSummary(); } },
      ...["train", "validation", "none"].map((v) => new Option({ train: "Training", validation: "Validation", none: "Not used" }[v], v, false, use[s.id] === v)));
    return el("tr", {}, el("td", { class: "mono" }, s.id.replace("session_", ""), " ", roleBadge(s.role)),
      el("td", {}, s.microphone || "laptop"), el("td", {}, s.participant || "—"), el("td", {}, s.keydowns ?? ""), el("td", {}, selectable(s) ? select : el("span", { class: "muted" }, "excluded")));
  }));
  renderSummary();
}

let coverageToken = 0;
async function renderSummary() {
  const chosen = (v) => state.sessions.filter((s) => use[s.id] === v);
  const train = chosen("train"), validation = chosen("validation");
  const presses = (list) => list.reduce((n, s) => n + (s.keydowns || 0), 0);
  $("trainSummary").replaceChildren(
    el("b", {}, `${train.length} training`), ` recordings (~${presses(train)} presses) · `,
    el("b", {}, `${validation.length} validation`), ` (~${presses(validation)} presses)`,
    validation.length ? "" : el("span", { class: "muted" }, " · without validation recordings there is nothing to compare on"));
  $("startTraining").disabled = train.length === 0;
  // A detector trained on the laptop microphone doesn't transfer: retrain it whenever phone recordings are used.
  $("retrainDetector").checked = train.some((s) => s.microphone === "phone");
  const token = ++coverageToken;
  let counts = {};
  try { counts = train.length ? await api(`api/key-counts?sessions=${train.map((s) => s.id).join(",")}`) : {}; } catch { return; }
  if (token !== coverageToken) return;
  const minimum = Math.max(1, +$("minPresses").value || 5);
  $("trainCoverage").replaceChildren(...KEYS_SHOWN.map((code) => {
    const n = counts[code] ?? 0;
    const cell = el("div", { class: `key-cell${n >= 40 ? " enough" : ""}`, title: n < minimum ? `${code}: ${n} presses, too few: left out of training` : `${code}: ${n} presses` },
      el("div", { class: "ch", style: n < minimum ? "color:var(--faint);text-decoration:line-through" : "" }, keyLabel(code)), el("div", { class: "n" }, n), el("div", { class: "fill" }));
    cell.lastChild.style.width = `${Math.min(100, (100 * n) / 40)}%`;
    return cell;
  }));
}

// ---- running ----

async function start() {
  showError("");
  const ids = (v) => Object.entries(use).filter(([, u]) => u === v).map(([id]) => id);
  const params = kind === "svm" ? { C: +$("paramC").value } : kind === "cnn" ? { epochs: +($("paramEpochs")?.value ?? 30) } : {};
  try {
    await api("api/training", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, params, train: ids("train"), validation: ids("validation"), name: $("modelName").value,
        end_to_end: $("endToEnd").checked, min_presses: +$("minPresses").value, retrain_detector: $("retrainDetector").checked }) });
  } catch (error) {
    return showError(`Couldn't start training: ${error.message}`);
  }
  lastShownResult = null;
  poll();
}

async function poll() {
  clearTimeout(poller);
  let status;
  try { status = await api("api/training"); } catch { status = { state: "idle" }; }
  showStatus(status);
  if (status.state === "running" && !document.getElementById("view-train").classList.contains("hidden")) poller = setTimeout(poll, 700);
}

function showStatus(status) {
  const running = status.state === "running";
  $("startTraining").disabled = running || !Object.values(use).includes("train");
  $("cancelTraining").disabled = !running;
  if (status.state === "idle") { $("trainStage").textContent = "Idle."; return; }
  $("trainStage").replaceChildren(running ? el("span", { class: "spinner" }) : "", `${status.stage}`,
    status.error ? el("span", { style: "color:var(--bad)" }, ` · ${status.error}`) : "");
  $("trainProgress").style.width = `${(status.progress || 0) * 100}%`;
  $("trainElapsed").textContent = `${status.elapsed_s?.toFixed(0) ?? 0} s`;
  const log = $("trainLog");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = status.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
  if (status.state === "done" && status.result && lastShownResult !== status.id) {
    lastShownResult = status.id;
    showResult(status.result);
    renderModels();
  }
}

function showResult({ model, evaluation }) {
  const box = $("trainResult");
  box.classList.remove("hidden");
  if (!evaluation.candidate) {
    box.replaceChildren(el("div", { class: "panel-head" }, el("h2", {}, `Trained: ${model.name}`)),
      el("p", { class: "muted" }, "No validation recordings were chosen, so the model couldn't be compared. It's saved below."),
      el("button", { class: "primary", onclick: () => activate(model.id) }, "Activate this model"));
    return;
  }
  const c = evaluation.candidate, a = evaluation.active;
  const row = (label, f, better = "higher") => {
    const cv = f(c), av = f(a);
    const winner = cv == null || av == null ? null : (better === "higher" ? cv > av : cv < av) ? "new" : cv === av ? null : "active";
    const cell = (v, who) => el("td", { class: "num", style: winner === who ? "color:var(--ok);font-weight:600" : "" }, v == null ? "–" : pct(v));
    return el("tr", {}, el("td", {}, label), cell(cv, "new"), cell(av, "active"));
  };
  const e2e = (x) => x.end_to_end;
  const newBetter = c.accuracy > a.accuracy;
  box.replaceChildren(
    el("div", { class: "panel-head" }, el("h2", {}, `Result: ${model.name}`),
      el("span", { class: "panel-sub" }, `${model.training_presses} training presses · ${evaluation.presses} validation presses · trained in ${model.fit_seconds.toFixed(0)} s`)),
    a.warning ? el("div", { class: "notice", style: "margin-bottom:10px" }, a.warning) : null,
    model.left_out_keys.length ? el("div", { class: "small muted", style: "margin-bottom:8px" }, `Keys left out (too few examples): ${model.left_out_keys.map(keyLabel).join(" ")}`) : null,
    el("table", { class: "data", style: "max-width:640px" },
      el("thead", {}, el("tr", {}, el("th", {}, "On the validation recordings"), el("th", {}, "New model"), el("th", {}, `Active: ${a.model.name}`))),
      el("tbody", {},
        row("Key accuracy (true press times)", (x) => x.accuracy),
        row("Balanced accuracy", (x) => x.balanced),
        row("Top-3", (x) => x.top3),
        ...(evaluation.detection ? [
          el("tr", {}, el("td", {}, "Press detection F1 (detector only)"),
            el("td", { class: "num", style: evaluation.detection.new.f1 > evaluation.detection.active.f1 ? "color:var(--ok);font-weight:600" : "" }, pct(evaluation.detection.new.f1)),
            el("td", { class: "num", style: evaluation.detection.active.f1 > evaluation.detection.new.f1 ? "color:var(--ok);font-weight:600" : "" }, pct(evaluation.detection.active.f1)))] : []),
        ...(e2e(c) ? [row("Presses detected, recall (end to end)", (x) => e2e(x).recall),
          row("Detected presses that are real, precision (end to end)", (x) => e2e(x).precision),
          row("Key accuracy, detected presses (end to end)", (x) => e2e(x).key_accuracy_matched),
          row("Character error, keys only", (x) => e2e(x).raw_cer, "lower"),
          row("Character error, with dictionary", (x) => e2e(x).dictionary_cer, "lower")] : []))),
    el("div", { class: "toolbar", style: "margin-top:12px" },
      el("button", { class: newBetter ? "primary" : "", onclick: () => activate(model.id) }, "Activate the new model"),
      el("span", { class: "small muted" }, newBetter ? "The new model scores higher on key accuracy." : "The active model still scores higher; you can keep it.")),
    el("details", { style: "margin-top:10px" }, el("summary", {}, "Per-key recall of the new model"),
      el("div", { class: "scroll" }, el("table", { class: "data" }, el("thead", {}, el("tr", {}, el("th", {}, "Key"), el("th", {}, "Presses"), el("th", {}, "Recall"), el("th", {}, "Most confused with"))),
        el("tbody", {}, ...c.per_key.map((r) => el("tr", {}, el("td", { class: "k" }, keyLabel(r.key)), el("td", {}, r.support), el("td", {}, pct(r.recall, 0)),
          el("td", { class: "k" }, r.confused_with ? keyLabel(r.confused_with) : "–"))))))));
}

// ---- models ----

async function renderModels() {
  let list = [];
  try { list = await api("api/models"); } catch (error) { return showError(error.message); }
  $("modelRows").replaceChildren(...list.map((m) => el("tr", {},
    el("td", {}, m.name, m.active ? el("span", { class: "badge unseen", style: "margin-left:8px" }, "active") : ""),
    el("td", { class: "mono" }, m.kind), el("td", {}, m.microphone || (m.id === "original-cnn" ? "laptop" : "–")),
    el("td", {}, `${(m.trained_on || []).length} recordings${m.training_presses ? ` · ${m.training_presses} presses` : ""}`),
    el("td", { class: "num" }, m.validation ? pct(m.validation.accuracy) : m.id === "original-cnn" ? "77.4% (original test session)" : "–"),
    el("td", { class: "small muted" }, m.compared_with ? `${m.compared_with.model.name}: ${pct(m.compared_with.accuracy)}` : ""),
    el("td", { class: "small muted" }, m.created ? new Date(m.created).toLocaleString() : "shipped"),
    el("td", {}, m.active ? "" : el("button", { class: "small", onclick: () => activate(m.id) }, "Activate"),
      m.active || m.id === "original-cnn" ? "" : el("button", { class: "small danger", style: "margin-left:6px", onclick: () => remove(m) }, "Delete")))));
}

async function activate(id) {
  try {
    await api(`api/models/${id}/activate`, { method: "POST" });
  } catch (error) { return showError(error.message); }
  await refreshSessions();       // roles depend on which model is active
  renderSessions();
  renderModels();
}

async function remove(model) {
  if (!confirm(`Delete "${model.name}"? This can't be undone.`)) return;
  try { await api(`api/models/${model.id}`, { method: "DELETE" }); } catch (error) { return showError(error.message); }
  renderModels();
}
