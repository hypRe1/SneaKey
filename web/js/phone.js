// The phone side of a phone recording (see sneakey/phone.py). Keeps the microphone open, reports its level, and
// records when the laptop says so: audio is uploaded in ~0.5 s chunks tagged with their audio-frame numbers, and
// timing data (page clock <-> audio frame pairs, round-trip pings to the server) goes up when the recording ends.
const code = location.pathname.split("/").filter(Boolean).pop();
const base = `/api/phone/d/${encodeURIComponent(code)}`;
const $ = (id) => document.getElementById(id);
const CONSTRAINTS = { channelCount: { ideal: 1 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false };
const CHUNK_SAMPLES = 24000;

let context = null, track = null;
let level = 0;
let recording = null;          // {id, pending: [], pendingStart, clockPairs, uploads: Promise, sent}
const pings = [];              // kept from the start, so a recording has measurements from before it began

function setState(text, detail = "", kind = "") {
  $("state").className = `state ${kind}`;
  $("state").replaceChildren(text, Object.assign(document.createElement("small"), { textContent: detail }));
}

function showError(message) {
  $("error").textContent = message;
  $("error").classList.toggle("hidden", !message);
}

async function call(path, options = {}) {
  const response = await fetch(base + path, { cache: "no-store", ...options });
  if (!response.ok) {
    let detail = `${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

$("enable").addEventListener("click", async () => {
  showError("");
  if (!navigator.mediaDevices?.getUserMedia) {
    return showError("This browser won't allow the microphone on this page. Open the https:// link exactly as shown on the laptop.");
  }
  try {
    try { context = new AudioContext({ sampleRate: 48000 }); } catch { context = new AudioContext(); }
    await context.resume();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: CONSTRAINTS });
    track = stream.getAudioTracks()[0];
    await context.audioWorklet.addModule("/js/capture-worklet.js");
    const node = new AudioWorkletNode(context, "capture", { numberOfOutputs: 0, channelCount: 1, channelCountMode: "explicit" });
    context.createMediaStreamSource(stream).connect(node);
    node.port.onmessage = (event) => { if (!event.data.flushed) onAudio(event.data); else recording?.onFlushed?.(); };
    window.__flush = () => new Promise((resolve) => { if (recording) recording.onFlushed = resolve; node.port.postMessage("flush"); });
  } catch (error) {
    return showError(error.name === "NotAllowedError" ? "Microphone permission was denied. Allow it in the browser's site settings and reload."
      : `Couldn't start the microphone: ${error.message}`);
  }
  try {
    await call("/hello", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_agent: navigator.userAgent, sample_rate: context.sampleRate, track_settings: track.getSettings() }) });
  } catch (error) {
    return showError(`Couldn't reach the laptop: ${error.message}`);
  }
  try { await navigator.wakeLock?.request("screen"); } catch { /* not supported: keep the screen on manually */ }
  $("enable").classList.add("hidden");
  $("facts").replaceChildren(...[["Code", code], ["Sample rate", `${context.sampleRate} Hz`],
    ["Processing", `echo cancel ${track.getSettings().echoCancellation ? "ON" : "off"}, noise suppression ${track.getSettings().noiseSuppression ? "ON" : "off"}, auto gain ${track.getSettings().autoGainControl ? "ON" : "off"}`]]
    .flatMap(([k, v]) => [Object.assign(document.createElement("span"), { textContent: k }), Object.assign(document.createElement("span"), { textContent: v })]));
  setState("Connected · waiting", "Start the recording on the laptop.");
  pingLoop();
  pollLoop();
  setInterval(() => { $("meter").style.width = `${Math.min(100, Math.sqrt(level) * 100)}%`; level *= 0.85; }, 100);
});

function onAudio({ samples, startFrame, peak }) {
  level = Math.max(level, peak);
  const rec = recording;
  if (!rec || !rec.active) return;
  // Start a new chunk if this batch doesn't follow on from the pending one (keeps frame numbers exact).
  if (rec.pending.length && rec.pendingStart + rec.pendingLength !== startFrame) flushChunk(rec);
  if (!rec.pending.length) rec.pendingStart = startFrame;
  rec.pending.push(samples);
  rec.pendingLength += samples.length;
  if (rec.pendingLength >= CHUNK_SAMPLES) flushChunk(rec);
}

function flushChunk(rec) {
  if (!rec.pending.length) return;
  const chunk = new Int16Array(rec.pendingLength);
  let offset = 0;
  for (const part of rec.pending) { chunk.set(part, offset); offset += part.length; }
  const start = rec.pendingStart;
  rec.pending = [];
  rec.pendingLength = 0;
  rec.sentSamples += chunk.length;
  // Uploads go one after another; each is retried a few times before giving up.
  rec.uploads = rec.uploads.then(async () => {
    for (let attempt = 0; attempt < 4; attempt++) {
      try {
        await fetch(`${base}/audio?recording_id=${rec.id}&start_frame=${start}`, { method: "POST", body: chunk.buffer }).then((r) => { if (!r.ok) throw new Error(r.status); });
        return;
      } catch { await new Promise((resolve) => setTimeout(resolve, 300 * (attempt + 1))); }
    }
    rec.failedChunks++;
  });
}

function sampleClock(rec) {
  const stamp = context.getOutputTimestamp ? context.getOutputTimestamp() : null;
  if (stamp && stamp.performanceTime > 0) rec.clockPairs.push({ perf: stamp.performanceTime, frame: stamp.contextTime * context.sampleRate });
  else rec.clockPairs.push({ perf: performance.now(), frame: context.currentTime * context.sampleRate });
}

async function pingLoop() {
  for (;;) {
    const t0 = performance.now();
    try {
      const { server } = await call("/clock");
      pings.push({ t0, t1: performance.now(), server });
      if (pings.length > 4000) pings.splice(0, 1000);
    } catch { /* missed ping: fine */ }
    await new Promise((resolve) => setTimeout(resolve, recording?.active ? 400 : 1000));
  }
}

async function pollLoop() {
  for (;;) {
    try {
      const reply = await call(`/poll?level=${level.toFixed(4)}`);
      if (reply.command === "record" && (!recording || recording.id !== reply.recording_id)) startRecording(reply.recording_id);
      else if (reply.command === "stop" && recording?.active && recording.id === reply.recording_id) await stopRecording();
      if (!recording?.active && reply.command === "wait" && !$("state").textContent.startsWith("Done")) setState("Connected · waiting", "Start the recording on the laptop.");
      showError("");
    } catch (error) {
      showError(`Lost contact with the laptop (${error.message}). Keep this page open; it reconnects automatically.`);
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
}

function startRecording(id) {
  recording = { id, active: true, pending: [], pendingStart: 0, pendingLength: 0, clockPairs: [], uploads: Promise.resolve(),
    sentSamples: 0, failedChunks: 0, startedAt: performance.now() };
  sampleClock(recording);
  recording.clockTimer = setInterval(() => sampleClock(recording), 250);
  recording.displayTimer = setInterval(() => setState("Recording", `${((performance.now() - recording.startedAt) / 1000).toFixed(0)} s · keep still`, "recording"), 500);
  setState("Recording", "keep still", "recording");
}

async function stopRecording() {
  const rec = recording;
  setState("Sending audio…", "Keep this page open.");
  await window.__flush();                 // audio still buffered in the worklet
  rec.active = false;
  clearInterval(rec.clockTimer);
  clearInterval(rec.displayTimer);
  sampleClock(rec);
  flushChunk(rec);
  await rec.uploads;
  const from = rec.startedAt - 30000;
  await call(`/finish?recording_id=${rec.id}`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_rate: context.sampleRate, clock_pairs: rec.clockPairs, pings: pings.filter((p) => p.t0 >= from),
      failed_chunks: rec.failedChunks }) });
  setState("Done", rec.failedChunks ? `${rec.failedChunks} audio chunks were lost (stored as silence).` : "Waiting for the next recording.");
}
