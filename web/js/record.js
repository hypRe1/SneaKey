// Section 1: the typing test. Records raw microphone audio and every keydown/keyup, maps each key event onto the
// audio's sample clock, and saves the session (same on-disk format as the original recorder).
import { $, api, el, keyLabel, roleBadge, showError } from "./common.js";
import { CONSTRAINTS, SAMPLE_RATE, openMicrophone } from "./microphone.js";
import { refreshSessions } from "./app.js";

const KEYS_SHOWN = "abcdefghijklmnopqrstuvwxyz".split("").map((c) => `Key${c.toUpperCase()}`).concat(["Space", "Comma", "Period", "Backspace"]);
const ENOUGH_PRESSES = 400;
const FIELDS = ["participant", "keyboard", "microphone", "environment"];

let test = null;           // {text, sentences, seed, mode, version, repeated_trigrams}
let target = "";
let typed = [];
let session = null;        // set while recording
let initialised = false;
window.__recordState = () => ({ events: session ? session.events.length : null, typed: typed.length, target: target.length });

export function enter() {
  if (!initialised) init();
  refreshSessions().then(showCoverage);
}

export function leave() {
  if (session) discard();     // never keep recording off-screen
}

function init() {
  initialised = true;
  for (const id of FIELDS) {
    $(id).value = localStorage.getItem(`sneakey-${id}`) || "";
    $(id).addEventListener("change", () => localStorage.setItem(`sneakey-${id}`, $(id).value));
  }
  $("newText").addEventListener("click", newText);
  $("length").addEventListener("change", newText);
  $("mode").addEventListener("change", newText);
  $("start").addEventListener("click", start);
  $("finish").addEventListener("click", finish);
  $("discard").addEventListener("click", discard);
  const typing = $("typing");
  typing.addEventListener("keydown", onKeyDown);
  typing.addEventListener("keyup", onKeyUp);
  typing.addEventListener("click", () => typing.focus());
  window.addEventListener("beforeunload", (event) => { if (session) event.preventDefault(); });
  newText();
}

// ---- the text ----

async function newText() {
  try {
    test = await api(`api/sentences?chars=${$("length").value}&mode=${$("mode").value}`);
  } catch (error) {
    showError(`Couldn't generate a test: ${error.message}`);
    return;
  }
  target = test.text;
  typed = [];
  $("seed").textContent = test.seed;
  $("repeats").textContent = test.repeated_trigrams;
  $("frames").replaceChildren(...test.sentences.map((s) => el("span", { title: s.text }, s.frame.replaceAll("_", " "))));
  drawText();
  updateStats();
}

function drawText() {
  const words = [];
  let word = null;
  for (let i = 0; i < target.length; i++) {
    if (!word || target[i - 1] === " ") { word = el("span", { class: "w" }); words.push(word); }
    const char = el("span", { class: "c" }, target[i]);
    if (i < typed.length) char.classList.add(typed[i] === target[i] ? "ok" : "bad");
    if (i === typed.length && session) char.classList.add("cur");
    word.append(char);
  }
  $("text").replaceChildren(...words);
}

function updateStats() {
  const keystrokes = session?.keystrokes ?? { chars: 0, correct: 0 };
  const seconds = session?.firstKeyTime ? (Date.now() - session.firstKeyTime) / 1000 : 0;
  const correctChars = typed.filter((c, i) => c === target[i]).length;
  $("wpm").textContent = seconds > 3 ? Math.round(correctChars / 5 / (seconds / 60)) : 0;
  $("accuracy").textContent = keystrokes.chars ? `${Math.round((100 * keystrokes.correct) / keystrokes.chars)}%` : "–";
  $("elapsed").textContent = `${seconds.toFixed(1)}s`;
  $("presses").textContent = session ? session.events.filter((e) => e.event_type === "keydown").length : 0;
  $("progress").style.width = `${target.length ? (100 * typed.length) / target.length : 0}%`;
}

async function showCoverage() {
  let counts = {};
  try { counts = await api("api/key-counts"); } catch { return; }
  $("coverage").replaceChildren(...KEYS_SHOWN.map((code) => {
    const n = counts[code] ?? 0;
    const cell = el("div", { class: `key-cell${n >= ENOUGH_PRESSES ? " enough" : ""}`, title: `${code}: ${n} presses` },
      el("div", { class: "ch" }, keyLabel(code)), el("div", { class: "n" }, n), el("div", { class: "fill" }));
    cell.lastChild.style.width = `${Math.min(100, (100 * n) / ENOUGH_PRESSES)}%`;
    return cell;
  }));
}

export function showSessions(sessions) {
  $("sessionCount").textContent = `${sessions.length} saved`;
  $("sessionRows").replaceChildren(...sessions.map((s) => el("tr", {},
    el("td", {}, el("a", { href: `#analyse/${s.id}`, class: "mono" }, s.id.replace("session_", ""))),
    el("td", {}, s.participant || "—"),
    el("td", { class: "num" }, s.duration_s ? `${s.duration_s.toFixed(0)} s` : "no audio"),
    el("td", { class: "num" }, s.keydowns ?? ""),
    el("td", {}, roleBadge(s.role)))));
  if (!sessions.length) $("sessionRows").replaceChildren(el("tr", {}, el("td", { colspan: 5, class: "muted" }, "No recordings yet.")));
}

// ---- recording ----

async function start() {
  showError("");
  if (!$("consent").checked) return showError("Confirm the participant's consent before recording.");
  if (!$("participant").value.trim()) return showError("Enter a participant ID (a pseudonym is fine).");
  if (!target) await newText();
  $("saved").classList.add("hidden");
  typed = [];
  session = {
    batches: [], firstFrame: null, clockPairs: [], events: [], startedAt: new Date(), firstKeyTime: null,
    keystrokes: { chars: 0, correct: 0 }, test,
  };
  setButtons(true);
  $("recState").textContent = "Starting microphone…";
  try {
    session.microphone = await openMicrophone(onAudio);
  } catch (error) {
    session = null;
    setButtons(false);
    $("recState").textContent = "Not recording";
    return showError(error.message);
  }
  // A suspended or silent capture would otherwise go unnoticed until saving: check audio is actually arriving.
  const arrived = await new Promise((resolve) => {
    const started = Date.now();
    const poll = setInterval(() => {
      if (!session) { clearInterval(poll); resolve(false); }
      else if (session.batches.length) { clearInterval(poll); resolve(true); }
      else if (Date.now() - started > 2500) { clearInterval(poll); resolve(false); }
    }, 50);
  });
  if (!arrived) {
    const failed = session;
    session = null;
    await failed?.microphone.stop().catch(() => {});
    setButtons(false);
    $("recState").textContent = "Not recording";
    return showError("No audio is arriving from the microphone. Check the input device and that the browser may use it, then try again.");
  }
  session.clockTimer = setInterval(sampleClocks, 250);
  session.statsTimer = setInterval(updateStats, 200);
  sampleClocks();
  $("recState").textContent = "Recording";
  $("recState").classList.add("on");
  $("typing").classList.remove("idle");
  $("typingHint").classList.add("hidden");
  drawText();
  $("typing").focus();
}

function onAudio({ samples, startFrame, peak }) {
  if (!session) return;
  if (session.firstFrame === null) session.firstFrame = startFrame;
  session.batches.push(samples);
  $("level").style.width = `${Math.min(100, Math.sqrt(peak) * 100)}%`;
}

// Key events are timed on the page clock (performance.now); the audio has its own clock. Every 250 ms we note
// the same moment on both and at the end fit a line from one to the other.
function sampleClocks() {
  const context = session.microphone.context;
  const stamp = context.getOutputTimestamp ? context.getOutputTimestamp() : null;
  if (stamp && stamp.performanceTime > 0) session.clockPairs.push({ perf: stamp.performanceTime, frame: stamp.contextTime * context.sampleRate });
  else session.clockPairs.push({ perf: performance.now(), frame: context.currentTime * context.sampleRate });
}

function onKeyDown(event) {
  if (!session || session.finishing) return;
  if ([" ", "Backspace", "'", "/"].includes(event.key)) event.preventDefault();
  const row = describeKey(event);
  if (!event.repeat && !event.ctrlKey && !event.altKey && !event.metaKey) {
    if (event.key === "Backspace") {
      row.action = typed.length ? "backspace" : "backspace_empty";
      typed.pop();
    } else if (event.key.length === 1 && typed.length < target.length) {
      row.action = "char";
      row.typed_char = event.key;
      row.correct = event.key === target[typed.length] ? 1 : 0;
      typed.push(event.key);
      session.keystrokes.chars++;
      session.keystrokes.correct += row.correct;
      session.firstKeyTime ??= new Date();
    }
  }
  row.cursor_after = typed.length;
  session.events.push(row);
  drawText();
  updateStats();
  if (typed.length === target.length && typed.join("") === target) finish();
}

function onKeyUp(event) {
  if (!session) return;
  session.events.push({ ...describeKey(event), action: "", target_char: "" });
}

function describeKey(event) {
  return {
    event_type: event.type, key: event.key, code: event.code, location: event.location,
    time_ms: Math.round(performance.now()), event_time: event.timeStamp,
    repeat: event.repeat ? 1 : 0, shift: event.shiftKey ? 1 : 0, ctrl: event.ctrlKey ? 1 : 0, alt: event.altKey ? 1 : 0, meta: event.metaKey ? 1 : 0,
    action: "ignored", target_char: target[typed.length] ?? "", typed_char: "", correct: "", cursor_before: typed.length,
  };
}

async function stopRecording() {
  if (!session || session.finishing) return null;
  session.finishing = true;
  $("finish").disabled = $("discard").disabled = true;
  // Keep recording briefly: the microphone hears a key 30-60 ms after the browser reports it, and the last
  // release comes later still.
  await new Promise((resolve) => setTimeout(resolve, 500));
  clearInterval(session.clockTimer);
  clearInterval(session.statsTimer);
  sampleClocks();
  const recorded = session;
  session = null;
  await recorded.microphone.stop();
  $("recState").textContent = "Not recording";
  $("recState").classList.remove("on");
  $("level").style.width = "0";
  $("typing").classList.add("idle");
  $("typingHint").classList.remove("hidden");
  setButtons(false);
  return recorded;
}

async function discard() {
  await stopRecording();
  typed = [];
  drawText();
  updateStats();
}

async function finish() {
  $("recState").textContent = "Saving…";
  const recorded = await stopRecording();
  if (!recorded) return;
  try {
    const id = await save(recorded);
    showSaved(id, recorded);
    await refreshSessions();
    showCoverage();
    newText();
  } catch (error) {
    showError(`Saving failed: ${error.message}`);
  }
}

function showSaved(id, recorded) {
  const presses = recorded.events.filter((e) => e.event_type === "keydown").length;
  const seconds = recorded.batches.reduce((sum, b) => sum + b.length, 0) / SAMPLE_RATE;
  $("saved").replaceChildren(
    el("div", { class: "panel-head" }, el("h2", {}, "Session saved"),
      el("div", { class: "toolbar" }, el("a", { href: `#analyse/${id}` }, "Analyse →"), el("a", { href: `#predict/${id}` }, "Predict →"))),
    el("div", { class: "small muted" }, el("span", { class: "mono" }, id), ` · ${seconds.toFixed(1)} s of audio · ${presses} key presses · ${$("wpm").textContent} WPM`));
  $("saved").classList.remove("hidden");
}

async function save(recorded) {
  const audio = joinBatches(recorded.batches);
  if (audio.length < SAMPLE_RATE / 2) throw new Error("the recording has less than half a second of audio");
  const clock = fitClock(recorded.clockPairs, recorded.firstFrame);
  const audioStartPerf = -clock.intercept / clock.slope;
  const events = recorded.events.map(({ event_time, ...row }) => ({
    ...row,
    event_time_ms: Math.round(event_time),
    t_rel_ms: (event_time - audioStartPerf).toFixed(3),
    sample_index_est: Math.round(clock.slope * event_time + clock.intercept),
  }));
  const typedText = typed.join("");
  const correctChars = [...typedText].filter((c, i) => c === target[i]).length;
  const minutes = recorded.firstKeyTime ? (Date.now() - recorded.firstKeyTime) / 60000 : 0;
  const metadata = {
    schema_version: 2,
    session_id: crypto.randomUUID(),
    name: $("participant").value.trim(),
    participant: { id: $("participant").value.trim(), consent_confirmed: true },
    conditions: { keyboard: $("keyboard").value.trim(), microphone: $("microphone").value.trim(), environment: $("environment").value.trim() },
    generator: { version: recorded.test.version, seed: recorded.test.seed, mode: recorded.test.mode,
      repeated_trigrams: recorded.test.repeated_trigrams, sentences: recorded.test.sentences },
    target_text: target,
    typed_text: typedText,
    target_words: target.split(" ").length,
    target_chars: target.length,
    session: {
      started_iso: recorded.startedAt.toISOString(),
      first_keystroke_iso: recorded.firstKeyTime?.toISOString() ?? null,
      finished_iso: new Date().toISOString(),
      completed_paragraph: typedText === target,
    },
    audio: { sample_rate_hz: SAMPLE_RATE, processing: "none (float to int16 conversion only)" },
    capture: { requested_constraints: CONSTRAINTS, applied_track_settings: recorded.microphone.trackSettings, user_agent: navigator.userAgent },
    sync: {
      primary_mapping: "output_timestamp",
      method: "Pairs of (performance.now, AudioContext time) every 250 ms, least-squares line from page time to audio sample.",
      ctx_frame_of_sample_0: recorded.firstFrame,
      slope_samples_per_ms: clock.slope,
      intercept_samples: clock.intercept,
      clock_pairs: recorded.clockPairs.length,
      note: "sample_index_est excludes the microphone's input delay (typically 30-60 ms); the analysis measures it.",
    },
    stats: {
      wpm: minutes > 0 ? +(correctChars / 5 / minutes).toFixed(2) : null,
      accuracy_keystrokes_pct: recorded.keystrokes.chars ? +((100 * recorded.keystrokes.correct) / recorded.keystrokes.chars).toFixed(2) : null,
    },
  };
  const { id } = await api("api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ metadata, events }) });
  try {
    await api(`api/sessions/${id}/audio`, { method: "PUT", body: audio.buffer });
  } catch (error) {
    await api(`api/sessions/${id}`, { method: "DELETE" }).catch(() => {});   // don't leave a session without audio
    throw error;
  }
  return id;
}

function joinBatches(batches) {
  const audio = new Int16Array(batches.reduce((sum, b) => sum + b.length, 0));
  let offset = 0;
  for (const b of batches) { audio.set(b, offset); offset += b.length; }
  return audio;
}

// Least-squares line: audio sample = slope * page time (ms) + intercept. Falls back to the nominal 48 samples/ms
// when the recording is too short to measure the slope or the fit is implausible.
export function fitClock(pairs, firstFrame) {
  const points = pairs.map((p) => ({ x: p.perf, y: p.frame - firstFrame }));
  const meanX = points.reduce((s, p) => s + p.x, 0) / points.length;
  const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
  let covariance = 0, variance = 0;
  for (const p of points) { covariance += (p.x - meanX) * (p.y - meanY); variance += (p.x - meanX) ** 2; }
  const nominal = SAMPLE_RATE / 1000;
  let slope = variance > 0 ? covariance / variance : nominal;
  const spanMs = Math.max(...points.map((p) => p.x)) - Math.min(...points.map((p) => p.x));
  if (spanMs < 5000 || Math.abs(slope / nominal - 1) > 0.001) slope = nominal;
  return { slope, intercept: meanY - slope * meanX };
}

function setButtons(recording) {
  $("start").disabled = recording;
  $("finish").disabled = !recording;
  $("discard").disabled = !recording;
  for (const id of ["newText", "length", "mode", ...FIELDS, "consent"]) $(id).disabled = recording;
}
