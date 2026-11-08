// Section 3: run the audio-only pipeline on a recording and evaluate it against what was typed.
import { $, alignChars, api, colours, el, fetchSamples, isLetter, keyLabel, pct, showError } from "./common.js";
import { computeSpectrogram } from "./spectrogram.js";
import { Timeline } from "./timeline.js";
import { heatmap, probabilityBars } from "./charts.js";
import { refreshSessions, sessionInfo } from "./app.js";
import * as player from "./player.js";

const HEAT_KEYS = [..."abcdefghijklmnopqrstuvwxyz"].map((c) => `Key${c.toUpperCase()}`).concat(["Space", "Comma", "Period", "Backspace"]);

let currentId = null;
let result = null;
let timeline = null;
let audio = null;
let selectedRow = null;
let initialised = false;

export async function enter(id) {
  if (!initialised) init();
  if (id === currentId && result) { timeline?.resize(); return; }
  currentId = id;
  result = null;
  selectedRow = null;
  timeline?.destroy();
  timeline = null;
  audio = null;
  $("predictResults").classList.add("hidden");
  showRole(sessionInfo(id));
  api("api/models").then((list) => {
    const active = list.find((m) => m.active);
    $("activeModelName").textContent = active?.name ?? "active model";
    // A model only works for the microphone it was trained on.
    const modelMic = active?.microphone || (active?.id === "original-cnn" ? "laptop" : null);
    const sessionMic = info?.microphone || "laptop";
    if (id === currentId && modelMic && modelMic !== "mixed" && modelMic !== sessionMic) {
      $("micNotice").textContent = `This is a ${sessionMic} recording, but the active model (${active.name}) was trained on ${modelMic} recordings, so expect poor results. ` +
        `Train a model on ${sessionMic} recordings (with the press detector retrained) on the Train page, then activate it.`;
      $("micNotice").classList.remove("hidden");
    }
  }).catch(() => {});
  $("micNotice").classList.add("hidden");
  const info = sessionInfo(id);
  if (info?.has_prediction) return run(false);
  $("predictStatus").replaceChildren("Not evaluated yet. Press ", el("b", {}, "Run prediction"), ". It takes ~10–20 s for a 90 s recording; results are cached.");
  $("predictStatus").classList.remove("hidden");
}

export function leave() { player.stop(); }

// "Highlight differences": one setting for the Text and DeepSeek panels, remembered between visits.
function setDiffHighlight(on) {
  document.getElementById("view-predict").classList.toggle("no-diff", !on);
  for (const box of document.querySelectorAll("[data-diff-toggle]")) box.checked = on;
  try { localStorage.setItem("sneakey-highlight-diff", on ? "1" : "0"); } catch { /* storage unavailable */ }
}

function init() {
  initialised = true;
  let highlight = true;
  try { highlight = localStorage.getItem("sneakey-highlight-diff") !== "0"; } catch { /* default on */ }
  setDiffHighlight(highlight);
  for (const box of document.querySelectorAll("[data-diff-toggle]")) box.addEventListener("change", () => setDiffHighlight(box.checked));
  $("runPredict").addEventListener("click", () => currentId && run(true));
  $("errorsOnly").addEventListener("change", renderWords);
  $("llmRun").addEventListener("click", () => runLlm());
  $("llmStyle").addEventListener("change", () => loadStoredLlm());
}

function showRole(info) {
  const notes = {
    trained: "The key classifier was trained on this recording, so its scores are optimistic and don't measure generalisation.",
    validation: "This recording is in the validation split (used for choosing settings), so its scores are mildly optimistic.",
    locked: "This recording is locked for a final evaluation (data/splits.json). Remove it from \"locked\" to evaluate it.",
  };
  const note = notes[info?.role];
  $("roleNotice").textContent = note ?? "";
  $("roleNotice").classList.toggle("hidden", !note);
  $("runPredict").disabled = info?.role === "locked";
}

async function run(refresh) {
  const id = currentId;
  const started = performance.now();
  $("predictStatus").classList.remove("hidden");
  const tick = setInterval(() => {
    $("predictStatus").replaceChildren(el("span", { class: "spinner" }),
      `Running detector → CNN → word decoder… ${((performance.now() - started) / 1000).toFixed(0)} s`);
  }, 250);
  $("runPredict").disabled = true;
  try {
    const data = await api(`api/sessions/${id}/predict${refresh ? "?refresh=true" : ""}`, { method: "POST" });
    if (id !== currentId) return;
    result = data;
    $("predictStatus").classList.add("hidden");
    $("predictResults").classList.remove("hidden");
    render();
    refreshSessions();
  } catch (error) {
    $("predictStatus").replaceChildren(el("span", { class: "muted" }, "Prediction failed."));
    showError(error.message);
  } finally {
    clearInterval(tick);
    $("runPredict").disabled = sessionInfo(id)?.role === "locked";
  }
}

// ---- rendering ----

function render() {
  renderCards();
  renderText();
  loadStoredLlm();
  renderWords();
  renderPerKey();
  renderConfusion();
  renderSummary();
  renderTimeline();
}

function card(label, value, detail, fraction) {
  return el("div", { class: "card" }, el("div", { class: "l" }, label), el("div", { class: "v" }, value),
    el("div", { class: "d" }, detail),
    fraction != null ? el("div", { class: "meter-bar" }, el("div", { style: `width:${Math.max(0, Math.min(1, fraction)) * 100}%` })) : null);
}

function renderCards() {
  const e = result.evaluation;
  const d = e.detection, a = e.audio_only, l = e.logged_timing, t = e.text;
  const wordsCorrect = e.word_alignment.filter((r) => r.op === "equal").length;
  const typedWords = e.word_alignment.filter((r) => r.typed != null).length;
  $("metricCards").replaceChildren(
    card("Presses detected", pct(d.recall), `${d.matched} of ${d.typed_presses} typed · precision ${pct(d.precision)} · ${d.extra} extra`, d.recall),
    card("Key accuracy · detected", pct(a.key_accuracy_matched), `of matched presses (audio only) · ${pct(a.correct_of_typed)} of all typed`, a.key_accuracy_matched),
    card("Key accuracy · logged timing", pct(l.accuracy), `classifier alone · top-3 ${pct(l.top3)} · balanced ${pct(l.balanced)}`, l.accuracy),
    card("Character error", t.dictionary_rates ? pct(t.dictionary_rates.cer) : "–",
      t.raw_rates ? `after dictionary · ${pct(t.raw_rates.cer)} from keys alone` : "no typed text", t.dictionary_rates ? 1 - t.dictionary_rates.cer : null),
    card("Words correct", `${wordsCorrect}`, `of ${typedWords} typed · WER ${t.dictionary_rates ? pct(t.dictionary_rates.wer) : "–"}`, typedWords ? wordsCorrect / typedWords : null),
  );
}

function diffSpans(typed, predicted) {
  const out = [];
  for (const [op, a, b] of alignChars(typed, predicted)) {
    if (op === "eq") out.push(b);
    else if (op === "sub") out.push(el("span", { class: "sub", title: `typed "${a}"` }, b));
    else if (op === "ins") out.push(el("span", { class: "ins", title: "extra character" }, b));
    else out.push(el("span", { class: "del", title: "missing character" }, a === " " ? "␣" : a));
  }
  return out;
}

function renderText() {
  const t = result.evaluation.text;
  const typed = t.typed.toLowerCase();
  $("textCompare").replaceChildren(
    el("div", { class: "label" }, "Typed"), el("div", { class: "text" }, typed || "(no typed text)"),
    el("div", { class: "label" }, "Predicted", el("br"), el("span", { class: "faint" }, "with dictionary")), el("div", { class: "text" }, diffSpans(typed, t.dictionary)),
    el("div", { class: "label" }, "Predicted", el("br"), el("span", { class: "faint" }, "keys only")), el("div", { class: "text" }, diffSpans(typed, t.raw)),
    el("div", { class: "label diff-legend" }), el("div", { class: "legend diff-legend" },
      el("span", {}, el("i", { style: "background:var(--bad)" }), "wrong character"), el("span", {}, el("i", { style: "background:var(--warn)" }), "extra character"),
      el("span", {}, el("i", { style: "background:var(--faint)" }), "missing (struck through)")));
}

const softmax = (scores) => { const m = Math.max(...scores); const w = scores.map((s) => Math.exp(s - m)); const z = w.reduce((a, b) => a + b, 0); return w.map((v) => v / z); };

function renderWords() {
  const rows = result.evaluation.word_alignment;
  const errorsOnly = $("errorsOnly").checked;
  $("words").replaceChildren(...rows.map((row, index) => {
    if (errorsOnly && row.op === "equal") return null;
    const piece = row.piece != null ? result.words.words[row.piece] : null;
    const confidence = piece?.scores?.length ? softmax(piece.scores)[0] : null;
    const button = el("button", { class: `word ${row.op}${selectedRow === index ? " selected" : ""}`, onclick: () => selectWord(index),
      title: row.op === "delete" ? "Typed word with no decoded word" : row.op === "insert" ? "Decoded word with no typed word" : `confidence ${pct(confidence)}` },
      el("div", { class: "p" }, row.predicted ?? row.typed),
      el("div", { class: "t" }, row.op === "equal" ? " " : row.typed ?? "—"),
      el("div", { class: "conf" }, el("div", { style: `width:${(confidence ?? 0) * 100}%` })));
    return button;
  }));
}

function selectWord(index, { scroll = false } = {}) {
  selectedRow = index;
  renderWords();
  const row = result.evaluation.word_alignment[index];
  const box = $("wordDetail");
  box.classList.remove("hidden");
  if (row.piece == null) {
    box.replaceChildren(el("h3", {}, `Missed word: “${row.typed}”`),
      el("p", { class: "muted small" }, "No decoded word was aligned to this typed word. Its presses were missed by the detector or merged into a neighbouring word by the decoder."));
    return;
  }
  const piece = result.words.words[row.piece];
  const truths = piece.presses.map((i) => result.evaluation.truth_for_prediction[i]);
  const probs = piece.presses.map((i) => result.probabilities[i]);
  const names = result.key_names;
  const confidence = softmax(piece.scores);
  const heatCanvas = el("canvas", { class: "heat" });

  const cards = piece.presses.map((pressIndex, k) => {
    const row = probs[k];
    const order = row.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const top = order.slice(0, 5).map(([p, i]) => ({ key: names[i], probability: p }));
    const truth = truths[k]?.code;
    if (truth && !top.some((x) => x.key === truth)) {
      const i = names.indexOf(truth);
      top.push({ key: truth, probability: i >= 0 ? row[i] : 0 });
    }
    const predicted = names[order[0][1]];
    const status = !truth ? "extra" : truth === predicted ? "right" : "wrong";
    const rank = truth ? order.findIndex(([, i]) => names[i] === truth) + 1 : null;
    return el("div", { class: `press-card ${status}` },
      el("div", { class: "head" }, el("b", {}, keyLabel(predicted)),
        el("span", { class: "muted" }, truth ? `typed ${keyLabel(truth)}${status === "wrong" ? ` · rank ${rank || ">49"}` : ""}` : "no typed key")),
      probabilityBars(top, truth, { compact: true }),
      el("div", { class: "faint", style: "margin-top:4px" }, `${(result.presses[pressIndex].time_ms / 1000).toFixed(2)} s`));
  });

  box.replaceChildren(
    el("div", { class: "panel-head" },
      el("h3", {}, `“${piece.text}”`, row.typed && row.typed !== piece.text ? el("span", { class: "muted" }, `  typed “${row.typed}”`) : null),
      el("div", { class: "toolbar" },
        el("span", { class: "small muted" }, `keys: `, el("span", { class: "mono" }, piece.raw), piece.from_dictionary ? " → dictionary" : " (kept raw: no dictionary word fitted)"),
        el("button", { class: "small", onclick: () => showOnTimeline(piece) }, "Show on timeline"),
        el("button", { class: "small", onclick: () => playPiece(piece) }, "▶ Play"))),
    el("div", { class: "small muted", style: "margin-bottom:8px" }, "Decoder candidates: ",
      ...[piece.text, ...piece.alternatives].slice(0, confidence.length).map((w, i) => el("span", { class: "mono", style: i ? "" : "color:var(--text)" }, `${w} ${pct(confidence[i], 0)}${i < confidence.length - 1 ? " · " : ""}`))),
    el("div", { class: "small muted", style: "margin:10px 0 4px" }, "Character probability distribution per press (rows: keys, columns: presses in time order; green box = what was typed, blue = top prediction)"),
    heatCanvas,
    el("div", { class: "press-grid", style: "margin-top:12px" }, cards));

  const columns = HEAT_KEYS.map((key) => names.indexOf(key));
  const matrix = HEAT_KEYS.map((_, r) => probs.map((row) => (columns[r] >= 0 ? Math.sqrt(row[columns[r]]) : 0)));
  const c = colours();
  const marks = [];
  probs.forEach((row, k) => {
    const top = HEAT_KEYS.indexOf(names[row.indexOf(Math.max(...row))]);
    if (top >= 0) marks.push({ row: top, col: k, colour: c.accent });
    const truth = HEAT_KEYS.indexOf(truths[k]?.code);
    if (truth >= 0) marks.push({ row: truth, col: k, colour: c.ok });
  });
  heatCanvas.style.maxWidth = `${Math.max(160, 70 + probs.length * 46)}px`;
  heatmap(heatCanvas, matrix, { height: 330, range: [0, 1], yLabels: HEAT_KEYS.map(keyLabel), labelWidth: 18, marks,
    xLabels: probs.map((_, k) => (truths[k] ? keyLabel(truths[k].code) : "+")), labelHeight: 14 });
  if (scroll) box.scrollIntoView({ block: "nearest" });
}

// ---- timeline ----

async function renderTimeline() {
  const id = currentId;
  $("predictTimeline").replaceChildren(el("div", { class: "empty" }, el("span", { class: "spinner" }), "Loading audio…"));
  try {
    const { samples, sampleRate } = await fetchSamples(`api/sessions/${id}/audio.wav`);
    const spectrogram = await computeSpectrogram(samples, sampleRate);
    if (id !== currentId) return;
    audio = { samples, sampleRate };
    const c = colours();
    const e = result.evaluation;
    const truthMarkers = e.true_presses.map((p, i) => ({
      t: p.time_ms / 1000, label: keyLabel(p.code), id: `t${i}`, hollow: p.matched < 0,
      colour: p.matched < 0 ? c.faint : p.correct ? c.ok : c.bad, selectable: false,
    }));
    const predictedMarkers = result.presses.map((p, i) => {
      const truth = e.truth_for_prediction[i];
      return { t: p.time_ms / 1000, label: keyLabel(p.key), id: `p${i}`, press: i,
        colour: !truth ? c.warn : truth.code === p.key ? c.ok : c.bad };
    });
    timeline = new Timeline($("predictTimeline"), { duration: samples.length / sampleRate, onSelect: (m) => m?.press != null && selectPress(m.press) });
    timeline.setLanes([
      { type: "waveform", samples, sampleRate, label: "Waveform", height: 44 },
      { type: "spectrogram", spectrogram, label: "Spectrogram (raw, noise floor subtracted)", height: 130 },
      { type: "markers", markers: truthMarkers, label: "Typed", height: 32 },
      { type: "markers", markers: predictedMarkers, label: "Predicted", height: 32 },
    ]);
    timeline.centreOn((result.presses[0]?.time_ms ?? 0) / 1000 + 2, 5);
  } catch (error) {
    $("predictTimeline").replaceChildren(el("div", { class: "error" }, error.message));
  }
}

function selectPress(pressIndex) {
  const rows = result.evaluation.word_alignment;
  const index = rows.findIndex((r) => r.piece != null && result.words.words[r.piece].presses.includes(pressIndex));
  if (index >= 0) selectWord(index, { scroll: true });
}

function showOnTimeline(piece) {
  if (!timeline || !piece.presses.length) return;
  const t0 = result.presses[piece.presses[0]].time_ms / 1000, t1 = result.presses[piece.presses.at(-1)].time_ms / 1000;
  timeline.setView(t0 - 0.3, Math.max(1, t1 - t0 + 0.6));
  $("predictTimeline").scrollIntoView({ block: "center", behavior: "smooth" });
}

function playPiece(piece) {
  if (!audio || !piece.presses.length) return;
  const t0 = result.presses[piece.presses[0]].time_ms / 1000, t1 = result.presses[piece.presses.at(-1)].time_ms / 1000;
  player.play(audio.samples, audio.sampleRate, t0 - 0.15, t1 + 0.25, { onPosition: (t) => timeline?.setCursor(t) });
}

// ---- per-key and summary ----

function renderPerKey() {
  $("perKey").replaceChildren(...result.evaluation.per_key.map((r) => el("tr", {},
    el("td", { class: "k" }, keyLabel(r.key), el("span", { class: "faint" }, isLetter(r.key) ? "" : `  ${r.key}`)),
    el("td", {}, r.support),
    el("td", {}, pct(r.recall, 0)),
    el("td", { style: "width:30%" }, el("div", { class: "meter-bar", style: "height:6px;background:var(--surface-3);border-radius:3px;overflow:hidden" },
      el("div", { style: `height:100%;width:${r.recall * 100}%;background:${r.recall >= 0.7 ? "var(--ok)" : r.recall >= 0.4 ? "var(--warn)" : "var(--bad)"}` }))),
    el("td", { class: "k" }, r.confused_with ? keyLabel(r.confused_with) : "–"))));
}

function renderConfusion() {
  const conf = result.evaluation.confusion;
  if (!conf) return;
  // Keep keys that were typed at least 3 times so the matrix stays readable.
  const support = conf.matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const keep = conf.keys.map((_, i) => i).filter((i) => support[i] >= 3);
  const matrix = keep.map((r) => keep.map((k) => conf.matrix[r][k] / Math.max(support[r], 1)));
  const labels = keep.map((i) => keyLabel(conf.keys[i]));
  heatmap($("confusion"), matrix, { height: Math.max(220, Math.min(460, keep.length * 13 + 16)), range: [0, 1], xLabels: labels, yLabels: labels, labelWidth: 18, labelHeight: 14 });
}

async function renderSummary() {
  let summary = {};
  try { summary = await api("api/summary"); } catch { return; }
  const order = ["unseen", "test", "validation", "trained"];
  $("summary").replaceChildren(...order.filter((r) => summary[r]).map((role) => {
    const s = summary[role];
    return el("tr", {}, el("td", {}, el("span", { class: `badge ${role}` }, role)), el("td", {}, s.sessions), el("td", {}, s.typed_presses),
      el("td", {}, pct(s.recall)), el("td", {}, pct(s.precision)), el("td", {}, pct(s.key_accuracy_matched)),
      el("td", {}, pct(s.logged_accuracy)), el("td", {}, pct(s.logged_top3)), el("td", {}, pct(s.raw_cer)), el("td", {}, pct(s.dictionary_cer)));
  }));
}

// ---- DeepSeek decoder (experimental, sends data off the laptop) ----

let llmConfigured = null;
let llmShown = null;     // the result currently displayed

function llmQuery() {
  const [style, guesses] = $("llmStyle").value.split("|");
  return `style=${style}&guesses=${guesses}`;
}

// Shows a stored result for the chosen prompt, if any. Never sends anything.
async function loadStoredLlm() {
  const id = currentId;
  llmShown = null;
  $("llmResult").classList.add("hidden");
  if (llmConfigured === null) {
    try { llmConfigured = (await api("api/llm")).configured; } catch { llmConfigured = false; }
  }
  const locked = sessionInfo(id)?.role === "locked";
  $("llmRun").disabled = !llmConfigured || locked;
  $("llmRun").textContent = "Decode with DeepSeek";
  $("llmStatus").textContent = !llmConfigured
    ? "No DeepSeek API key configured: add DEEPSEEK_API_KEY=... to data/deepseek.env (git-ignored), then restart the server."
    : locked ? "Locked recording: not available." : "Not decoded with this prompt yet. Nothing is sent until you press the button.";
  try {
    const stored = await api(`api/sessions/${id}/llm-decode?${llmQuery()}`);
    if (id !== currentId) return;
    showLlm(stored, true);
  } catch { /* nothing stored */ }
}

async function runLlm() {
  const id = currentId;
  const presses = result?.presses?.length ?? 0;
  const ok = confirm(`Send the candidate letters for ${presses} detected key presses of ${id}` +
    `${$("llmStyle").value.startsWith("ensemble") ? " (plus the dictionary decoder's word guesses)" : ""} to api.deepseek.com?\n\n` +
    "The audio and typed text are not sent, but the text can largely be inferred from what is. Only continue if the participant agreed.");
  if (!ok) return;
  $("llmRun").disabled = true;
  const started = performance.now();
  const tick = setInterval(() => {
    $("llmStatus").replaceChildren(el("span", { class: "spinner" }), `Waiting for DeepSeek… ${((performance.now() - started) / 1000).toFixed(0)} s`);
  }, 250);
  try {
    const data = await api(`api/sessions/${id}/llm-decode?${llmQuery()}${llmShown ? "&refresh=true" : ""}`, { method: "POST" });
    if (id === currentId) showLlm(data, false);
  } catch (error) {
    $("llmStatus").replaceChildren(el("span", { style: "color:var(--bad)" }, `DeepSeek decoding failed: ${error.message}`));
  } finally {
    clearInterval(tick);
    $("llmRun").disabled = !llmConfigured;
  }
}

function showLlm(data, stored) {
  llmShown = data;
  $("llmRun").textContent = "Decode again (re-send)";
  const tokens = data.usage ? data.usage.prompt_tokens + data.usage.completion_tokens : 0;
  $("llmStatus").textContent = `${stored ? "Stored result, not re-sent." : "Decoded just now."} ${data.presses_sent} presses sent · ${tokens.toLocaleString()} tokens · model ${data.model}`;
  const rows = [
    ["DeepSeek", data.rates, data.words_right?.llm],
    ["Dictionary (local, used by the app)", data.dictionary.rates, data.words_right?.dictionary],
    ["Keys only", data.raw.rates, data.words_right?.raw],
  ];
  const best = (i, better) => {
    const values = rows.map((r) => (i === 2 ? r[2] : r[1]?.[["cer", "wer"][i]]));
    return better === "lower" ? Math.min(...values) : Math.max(...values);
  };
  const cell = (value, i, text, better) => el("td", { class: "num", style: value === best(i, better) ? "color:var(--ok);font-weight:600" : "" }, text);
  const typed = (data.typed || "").toLowerCase();
  const used = data.models_used ? `Models whose candidates were combined: ${data.models_used.join(", ")}` +
    (data.models_skipped_trained_on_session?.length ? ` (skipped, trained on this recording: ${data.models_skipped_trained_on_session.join(", ")})` : "") : "";
  $("llmResult").replaceChildren(
    el("table", { class: "data", style: "max-width:720px" },
      el("thead", {}, el("tr", {}, el("th", {}, "Decoder"), el("th", {}, "Character error"), el("th", {}, "Word error"), el("th", {}, "Words exactly right"))),
      el("tbody", {}, ...rows.map(([name, rates, right]) => el("tr", {}, el("td", {}, name),
        cell(rates?.cer, 0, rates ? pct(rates.cer) : "–", "lower"), cell(rates?.wer, 1, rates ? pct(rates.wer) : "–", "lower"),
        cell(right, 2, right != null ? `${right} / ${data.typed_words}` : "–", "higher"))))),
    el("div", { class: "compare", style: "margin-top:12px" },
      el("div", { class: "label" }, "Typed"), el("div", { class: "text" }, typed || "(no typed text)"),
      el("div", { class: "label" }, "DeepSeek"), el("div", { class: "text" }, typed ? diffSpans(typed, data.text) : data.text)),
    el("details", { style: "margin-top:10px" }, el("summary", {}, "What was sent"),
      el("p", { class: "small muted" }, used),
      el("pre", { class: "small mono", style: "white-space:pre-wrap;max-height:260px;overflow:auto;color:var(--muted)" },
        `SYSTEM:\n${data.system_prompt}\n\nFIRST REQUEST:\n${data.prompt_example}`)));
  $("llmResult").classList.remove("hidden");
}
