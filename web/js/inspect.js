// Section 2: inspect a recording. Raw vs processed spectrograms, the detector's novelty curve, ground-truth
// keystrokes on their sounds, and a per-keystroke inspector showing exactly what the models compute.
import { $, api, colours, el, fetchSamples, keyLabel, secs, showError } from "./common.js";
import { computeSpectrogram } from "./spectrogram.js";
import { Timeline } from "./timeline.js";
import { heatmap, lineChart, probabilityBars } from "./charts.js";
import * as player from "./player.js";

const VARIANTS = { raw: "Raw", detector: "Detector 180 Hz", model: "Model input 700 Hz", click: "Click band 1.2 kHz" };
const MODIFIER_CODES = /^(Shift|Control|Alt|Meta|CapsLock|OS)/;

let current = null;        // {id, session, signals, audio: {variant: {samples, sampleRate, spectrogram}}, timeline, presses}
let selection = { top: "raw", bottom: "model" };
let initialised = false;
let loadToken = 0;

export async function enter(id) {
  if (!initialised) init();
  if (current?.id === id) { current.timeline.resize(); return; }
  const token = ++loadToken;
  current?.timeline?.destroy();
  current = null;
  $("analyseTimeline").replaceChildren(el("div", { class: "empty" }, el("span", { class: "spinner" }), "Loading recording…"));
  $("keystroke").replaceChildren(el("div", { class: "empty" }, "Select a keystroke on the timeline."));
  try {
    const [session, signals] = await Promise.all([api(`api/sessions/${id}`), api(`api/sessions/${id}/signals`)]);
    const audio = {};
    await Promise.all([...new Set(["raw", selection.top, selection.bottom])].map((v) => loadVariant(id, v, audio)));
    if (token !== loadToken) return;
    current = { id, session, signals, audio };
    build();
  } catch (error) {
    if (token === loadToken) showError(`Couldn't load ${id}: ${error.message}`);
  }
}

export function leave() { player.stop(); }

async function loadVariant(id, variant, into) {
  if (into[variant]) return into[variant];
  const { samples, sampleRate } = await fetchSamples(`api/sessions/${id}/audio.wav?filter=${variant}`);
  const spectrogram = await computeSpectrogram(samples, sampleRate);
  return (into[variant] = { samples, sampleRate, spectrogram });
}

function init() {
  initialised = true;
  for (const [slot, id] of [["top", "specTop"], ["bottom", "specBottom"]]) {
    $(id).replaceChildren(...Object.entries(VARIANTS).map(([key, label]) =>
      el("button", { "data-variant": key, onclick: () => chooseVariant(slot, key) }, label.split(" ")[0])));
  }
  $("whiten").addEventListener("change", () => { if (current) { current.timeline.whiten = $("whiten").checked; current.timeline.draw(); } });
  $("contrast").addEventListener("input", () => { if (current) { current.timeline.contrast = +$("contrast").value; current.timeline.draw(); } });
  $("playRaw").addEventListener("click", () => playView("raw"));
  $("playModel").addEventListener("click", () => playView("model"));
  document.addEventListener("keydown", onKey);
  markVariantButtons();
}

async function chooseVariant(slot, variant) {
  selection[slot] = variant;
  markVariantButtons();
  if (!current) return;
  await loadVariant(current.id, variant, current.audio);
  setLanes();
}

function markVariantButtons() {
  for (const [slot, id] of [["top", "specTop"], ["bottom", "specBottom"]]) {
    for (const button of $(id).children) {
      button.classList.toggle("on", button.dataset.variant === selection[slot]);
      button.title = VARIANTS[button.dataset.variant];
    }
  }
}

// ---- timeline ----

function build() {
  const { signals, session } = current;
  const c = colours();
  current.presses = [];
  const markers = [];
  signals.events.forEach((event, index) => {
    if (event.time_ms == null) return;
    const t = event.time_ms / 1000;
    if (event.type === "keyup") { markers.push({ t, thin: true, colour: c.release, selectable: false, id: `u${index}` }); return; }
    const colour = event.action === "char" ? (event.correct ? c.ok : c.bad) : c.warn;
    const marker = { t, label: keyLabel(event.code), colour, id: `d${index}`, event, index };
    markers.push(marker);
    current.presses.push(marker);
  });
  current.presses.sort((a, b) => a.t - b.t);
  current.markers = markers;
  current.timeline = new Timeline($("analyseTimeline"), {
    duration: signals.duration_ms / 1000,
    onSelect: (marker, t) => marker ? inspect(marker) : inspectNearestCandidate(t),
  });
  current.timeline.whiten = $("whiten").checked;
  current.timeline.contrast = +$("contrast").value;
  setLanes();
  showRecordingInfo(session, signals);
  drawSpectrumChart();
  if (current.presses.length) current.timeline.centreOn(current.presses[0].t + 1.5, 4);
}

function setLanes() {
  const { audio, signals } = current;
  const c = colours();
  const candidatePoints = signals.candidates.map((k) => ({ t: k.time_ms / 1000, colour: k.press_probability >= signals.press_threshold ? c.accent : c.candidate }));
  current.timeline.setLanes([
    { type: "waveform", samples: audio.raw.samples, sampleRate: audio.raw.sampleRate, label: "Waveform (raw, auto-scaled)", height: 64 },
    { type: "spectrogram", spectrogram: audio[selection.top].spectrogram, label: `${VARIANTS[selection.top]} · 0–16 kHz` },
    { type: "spectrogram", spectrogram: audio[selection.bottom].spectrogram, label: `${VARIANTS[selection.bottom]} · 0–16 kHz` },
    { type: "curve", values: signals.novelty, threshold: signals.flux_threshold, max: signals.flux_threshold * 4, points: candidatePoints,
      label: `Detector novelty (flux ÷ median) · dashed = threshold ${signals.flux_threshold} · dots = candidates, blue = classified as press`, height: 70 },
    { type: "markers", markers: current.markers, label: "Ground truth", height: 34 },
  ]);
}

function playView(variant) {
  if (!current) return;
  const a = current.audio[variant] ?? current.audio.raw;
  const { start, length } = current.timeline.view;
  player.play(a.samples, a.sampleRate, start, start + length, { onPosition: (t) => current?.timeline.setCursor(t) });
}

function onKey(event) {
  if (!current || document.getElementById("view-analyse").classList.contains("hidden")) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (event.key === " ") { event.preventDefault(); player.isPlaying() ? player.stop() : playView("raw"); }
  if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
    event.preventDefault();
    const presses = current.presses;
    const now = current.timeline.selected?.t ?? current.timeline.view.start;
    const next = event.key === "ArrowRight" ? presses.find((p) => p.t > now + 1e-4) : [...presses].reverse().find((p) => p.t < now - 1e-4);
    if (next) { current.timeline.select(next); inspect(next); }
  }
}

// ---- keystroke inspector ----

function inspectNearestCandidate(t) {
  const candidate = current.signals.candidates.reduce((best, k) => (Math.abs(k.time_ms / 1000 - t) < Math.abs((best?.time_ms ?? Infinity) / 1000 - t) ? k : best), null);
  if (!candidate || Math.abs(candidate.time_ms / 1000 - t) > 0.02) return;
  const marker = { t: candidate.time_ms / 1000, id: `c${candidate.time_ms}` };
  current.timeline.select(marker, { centre: false });
  inspect(marker, candidate);
}

let inspectToken = 0;
async function inspect(marker, candidate = null) {
  const token = ++inspectToken;
  const event = marker.event;
  $("ksPosition").textContent = secs(marker.t * 1000);
  $("keystroke").replaceChildren(el("div", { class: "muted small" }, el("span", { class: "spinner" }), "Running the models at this position…"));
  let detail;
  try {
    detail = await api(`api/sessions/${current.id}/keystroke?time_ms=${marker.t * 1000}`);
  } catch (error) {
    return $("keystroke").replaceChildren(el("div", { class: "error" }, error.message));
  }
  if (token !== inspectToken) return;
  const truth = event?.code;
  const lag = event ? event.time_ms - event.logged_ms : null;
  const correctTop = truth && detail.top[0].key === truth;

  const patchCanvas = el("canvas", { class: "heat" });
  const waveCanvas = el("canvas", { class: "heat" });
  const info = el("dl", { class: "kv" },
    ...(event ? [
      el("dt", {}, "Typed"), el("dd", {}, `${event.key === " " ? "Space" : event.key} (${event.code})${event.repeat ? " · auto-repeat" : ""}`),
      el("dt", {}, "Intended"), el("dd", {}, event.action === "char" ? `${JSON.stringify(event.target_char)} ${event.correct ? "✓" : "✗ typo"}` : event.action),
      el("dt", {}, "Logged at"), el("dd", {}, secs(event.logged_ms)),
      el("dt", {}, "Heard at"), el("dd", {}, `${secs(event.time_ms)} (+${lag.toFixed(0)} ms input delay)`),
      el("dt", {}, "Click level"), el("dd", {}, event.snr_db != null ? `${event.snr_db} dB above background` : "–"),
    ] : [el("dt", {}, "Detector candidate"), el("dd", {}, `no logged key here · press probability ${(candidate?.press_probability * 100).toFixed(0)}%`)]),
    el("dt", {}, "Press classifier"), el("dd", {}, `${(detail.press_probability * 100).toFixed(0)}% press (threshold ${(current.signals.press_threshold * 100).toFixed(0)}%)`));

  $("keystroke").replaceChildren(
    el("div", { class: "grid two" },
      el("div", {},
        el("div", { style: "display:flex;align-items:baseline;gap:14px;margin-bottom:10px" },
          el("span", { class: "big-key" }, truth ? keyLabel(truth) : "?"),
          el("span", { class: correctTop ? "badge unseen" : truth ? "badge locked" : "badge" },
            `Model: ${keyLabel(detail.top[0].key)} ${(detail.top[0].probability * 100).toFixed(0)}%`)),
        info,
        el("div", { class: "toolbar", style: "margin-top:12px" },
          el("button", { class: "small", onclick: () => playAround(marker.t, "raw") }, "▶ Raw"),
          el("button", { class: "small", onclick: () => playAround(marker.t, "model") }, "▶ Processed"),
          el("button", { class: "small", onclick: () => current.timeline.centreOn(marker.t, 0.4) }, "Zoom to key"))),
      el("div", {},
        el("h3", { style: "margin-bottom:8px" }, `Key probabilities · ${detail.classifier.name}`),
        probabilityBars(detail.top, truth))),
    el("h3", { style: "margin:16px 0 6px" }, "What the key classifier sees"),
    el("div", { class: "small muted", style: "margin-bottom:6px" },
      detail.input_view.description),
    patchCanvas,
    el("h3", { style: "margin:16px 0 6px" }, "Waveform ±60 ms"),
    waveCanvas,
    el("div", { class: "legend", style: "margin-top:4px" }, el("span", {}, el("i", { style: "background:var(--faint)" }), "Raw"), el("span", {}, el("i", { style: "background:var(--warn)" }), "700 Hz high-passed (model input)")),
    el("details", { style: "margin-top:12px" }, el("summary", {}, "Press-classifier features"),
      el("div", { class: "features", style: "margin-top:6px" }, ...Object.entries(detail.features).flatMap(([k, v]) => [el("span", {}, k), el("span", {}, v.toFixed(3))]))));

  // patch: frames x mels -> rows = mels (low at bottom), cols = frames
  // matrix: (time steps, bands) -> rows = bands (low frequencies at the bottom), columns = time
  const view = detail.input_view;
  const frames = view.matrix.length;
  const matrix = view.matrix[0].map((_, band) => view.matrix.map((row) => row[band]));
  const [t0, t1] = view.time_ms;
  const xLabels = Array.from({ length: frames }, (_, i) => {
    const t = Math.round(t0 + ((t1 - t0) * i) / frames);
    const next = Math.round(t0 + ((t1 - t0) * (i + 1)) / frames);
    return Math.floor(t / 15) !== Math.floor(next / 15) || i === 0 ? `${t}` : "";
  });
  // Colour range from the 2nd-98th percentile: a few near-empty low mel bands would otherwise flatten the rest.
  const sorted = matrix.flat().sort((a, b) => a - b);
  const range = [sorted[Math.floor(sorted.length * 0.02)], sorted[Math.floor(sorted.length * 0.98)]];
  heatmap(patchCanvas, matrix, { height: 200, flipY: true, xLabels, labelHeight: 14, range });
  const c = colours();
  const xs = detail.waveform.raw.map((_, i) => i);
  const norm = (values) => { const m = Math.max(...values.map(Math.abs)) || 1; return values.map((v) => v / m); };
  lineChart(waveCanvas, xs, [{ values: norm(detail.waveform.raw), colour: c.faint }, { values: norm(detail.waveform.model), colour: c.warn }], { height: 130 });
}

function playAround(t, variant) {
  const a = current.audio[variant] ?? current.audio.raw;
  player.play(a.samples, a.sampleRate, t - 0.12, t + 0.2, { onPosition: (p) => current?.timeline.setCursor(p) });
}

// ---- side panels ----

function showRecordingInfo(session, signals) {
  const presses = signals.events.filter((e) => e.type === "keydown");
  const levels = presses.map((e) => e.snr_db).filter((v) => v != null).sort((a, b) => a - b);
  const rows = [
    ["Session", session.id], ["Participant", session.participant?.id || session.name || "—"],
    ["Duration", secs(signals.duration_ms)], ["Key presses", presses.length],
    ["Typing speed", session.stats?.wpm ? `${session.stats.wpm.toFixed(0)} WPM, ${session.stats.accuracy_keystrokes_pct}% accurate` : "–"],
    ["Input delay", `${signals.press_lag_ms} ms press, ${signals.release_lag_ms} ms release (measured from audio)`],
    ["Click level", levels.length ? `median ${levels[Math.floor(levels.length / 2)]} dB above background` : "–"],
    ["Detector", `${signals.candidates.length} candidates, ${signals.candidates.filter((k) => k.press_probability >= signals.press_threshold).length} classified as presses`],
    ["Setup", [session.conditions?.keyboard, session.conditions?.microphone, session.conditions?.environment].filter(Boolean).join(" · ") || "–"],
    ["Text", session.generator?.seed != null ? `generated, seed ${session.generator.seed} (${session.generator.mode})` : "–"],
  ];
  $("recordingInfo").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, String(v))]));
  $("fluxThreshold").textContent = signals.flux_threshold;
}

// Average spectrum of the background (median level per frequency) and of the keystrokes (level 5 ms after each
// true press), raw and processed. Shows what the high-pass and noise floor remove and where key clicks carry energy.
function drawSpectrumChart() {
  const c = colours();
  const raw = current.audio.raw.spectrogram, model = (current.audio.model ?? current.audio.raw).spectrogram;
  const toDb = (level) => level / 2 - 130;
  const keyMean = (s) => {
    const sums = new Float64Array(s.bins);
    let n = 0;
    for (const p of current.presses) {
      const frame = Math.round((p.t + 0.005 - s.size / 2 / s.sampleRate) * s.sampleRate / s.hop);
      if (frame < 0 || frame >= s.frames) continue;
      for (let b = 0; b < s.bins; b++) sums[b] += toDb(s.levels[frame * s.bins + b]);
      n++;
    }
    return Array.from(sums, (v) => v / Math.max(n, 1));
  };
  const bins = Array.from({ length: raw.bins }, (_, b) => b * raw.binHz);
  const keep = bins.map((hz, i) => i).filter((i) => bins[i] >= 100 && bins[i] <= 20000);
  const pick = (arr) => keep.map((i) => arr[i]);
  lineChart($("spectrumChart"), pick(bins), [
    { values: pick(Array.from(raw.noiseFloor, toDb)), colour: c.faint },
    { values: pick(keyMean(raw)), colour: c.accent },
    { values: pick(keyMean(model)), colour: c.warn, dashed: true },
  ], { height: 190, logX: true, xTicks: [100, 300, 1000, 3000, 10000], xLabel: "Hz", yLabel: "dB" });
}
