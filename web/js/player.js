// Plays a stretch of a recording (Float32 samples) and reports the playback position for the timeline cursor.
//
// Keyboard clicks recorded by a laptop microphone are quiet (peaks around -30 dBFS) and the high-passed versions
// quieter still, so each played stretch is normalised: its peak is raised to about -3 dBFS (at most +45 dB), with a
// limiter as a safety net. The recording itself is never changed.
import { showError } from "./common.js";

const TARGET_PEAK = 0.7;
const MAX_GAIN_DB = 45;
export const VOLUME_RANGE_DB = [-24, 30];    // the volume slider, relative to the normalised level

let context = null;
let current = null;
let volumeDb = readVolume();

function readVolume() {
  try {
    const saved = Number(localStorage.getItem("sneakey-volume-db"));
    return Number.isFinite(saved) && localStorage.getItem("sneakey-volume-db") !== null ? saved : 12;
  } catch { return 12; }
}

export const getVolumeDb = () => volumeDb;

// Applies immediately, including to a clip that is already playing.
export function setVolumeDb(db) {
  volumeDb = Math.max(VOLUME_RANGE_DB[0], Math.min(VOLUME_RANGE_DB[1], db));
  try { localStorage.setItem("sneakey-volume-db", String(volumeDb)); } catch { /* storage unavailable */ }
  if (current) current.amp.gain.setTargetAtTime(current.baseGain * 10 ** (volumeDb / 20), context.currentTime, 0.02);
}

export function play(samples, sampleRate, fromSeconds, toSeconds, { onPosition } = {}) {
  stop();
  context ??= new AudioContext();
  // A context can start (or end up) suspended, e.g. if the browser didn't count the action as a user gesture.
  // Resuming here, inside the click or key handler, is what browsers allow.
  if (context.state !== "running") context.resume().catch(() => {});
  const from = Math.max(0, Math.floor(fromSeconds * sampleRate));
  const to = Math.min(samples.length, Math.ceil(toSeconds * sampleRate));
  if (to <= from) return;
  const segment = samples.subarray(from, to);
  let peak = 0;
  for (let i = 0; i < segment.length; i++) { const v = Math.abs(segment[i]); if (v > peak) peak = v; }
  const gain = Math.min(TARGET_PEAK / Math.max(peak, 1e-6), 10 ** (MAX_GAIN_DB / 20));

  const buffer = context.createBuffer(1, segment.length, sampleRate);
  buffer.copyToChannel(segment, 0);
  const source = new AudioBufferSourceNode(context, { buffer });
  const amp = new GainNode(context, { gain: gain * 10 ** (volumeDb / 20) });
  const limiter = new DynamicsCompressorNode(context, { threshold: -2, knee: 0, ratio: 20, attack: 0.001, release: 0.05 });
  // The compressor isn't a hard ceiling (it overshot full scale at high volume settings), so a gentle tanh soft
  // clipper keeps the output strictly below 1 and avoids harsh digital clipping when the volume is turned up.
  const ceiling = new WaveShaperNode(context, { curve: softClipCurve(), oversample: "2x" });
  source.connect(amp).connect(limiter).connect(ceiling).connect(context.destination);
  const startedAt = context.currentTime;
  source.start();
  current = { source, timer: null, amp, baseGain: gain };
  const tick = () => {
    if (!current || current.source !== source) return;
    onPosition?.(fromSeconds + (context.currentTime - startedAt));
    current.timer = requestAnimationFrame(tick);
  };
  tick();
  source.onended = () => { if (current?.source === source) { stop(); onPosition?.(null); } };

  // If the browser still refuses to start audio, say so instead of failing silently.
  setTimeout(() => {
    if (context.state !== "running" && current?.source === source) {
      stop();
      onPosition?.(null);
      showError("The browser blocked audio playback. Click the page once and press play again, or allow sound for this site in the browser's site settings.");
    }
  }, 600);
}

let curve = null;
function softClipCurve() {
  if (curve) return curve;
  curve = new Float32Array(2049);
  const k = 1.5;
  for (let i = 0; i < curve.length; i++) {
    const x = (i / (curve.length - 1)) * 2 - 1;      // the shaper maps inputs in [-1, 1]; beyond is clamped
    curve[i] = (0.95 * Math.tanh(k * x)) / Math.tanh(k);
  }
  return curve;
}

export function stop() {
  if (!current) return;
  cancelAnimationFrame(current.timer);
  try { current.source.stop(); } catch { /* already stopped */ }
  current = null;
}

export const isPlaying = () => current !== null;
