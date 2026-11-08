// A zoomable, stacked time view of one recording, used by the Analyse and Predict views.
//
//   const timeline = new Timeline(container, { duration, onSelect });
//   timeline.setLanes([{ type: "waveform", samples, height }, { type: "spectrogram", spectrogram, label },
//                      { type: "curve", values (one per ms), threshold, label }, { type: "markers", markers, label }]);
//
// Markers are {t (seconds), label, colour, hollow, thin, id, selectable}. Every lane shares one time axis;
// scroll zooms around the pointer, dragging pans, clicking selects the nearest selectable marker.
import { COLOURS } from "./spectrogram.js";
import { colours } from "./common.js";

const AXIS = 18;          // px for the time axis at the bottom
const LABEL_W = 0;        // lane labels are drawn inside the lanes
const MAX_HZ = 16000;

export class Timeline {
  constructor(container, { duration, onSelect, onView } = {}) {
    this.container = container;
    this.duration = duration;            // seconds
    this.onSelect = onSelect;
    this.onView = onView;
    this.view = { start: 0, length: Math.min(4, duration) };
    this.lanes = [];
    this.selected = null;
    this.cursor = null;
    this.contrast = 30;
    this.whiten = true;
    this.overview = Object.assign(document.createElement("canvas"), { className: "overview" });
    this.main = Object.assign(document.createElement("canvas"), { className: "main" });
    container.replaceChildren(this.overview, this.main);
    this.bindEvents();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
  }

  destroy() { this.resizeObserver.disconnect(); }

  setLanes(lanes) {
    this.lanes = lanes;
    for (const lane of lanes) if (lane.type === "waveform") lane.blocks ??= blockPeaks(lane.samples);
    this.overviewLane = lanes.find((l) => l.type === "waveform") ?? null;
    this.resize();
  }

  // ---- view ----

  setView(start, length) {
    length = Math.max(0.05, Math.min(this.duration, length));
    start = Math.max(0, Math.min(this.duration - length, start));
    this.view = { start, length };
    this.draw();
    this.onView?.(this.view);
  }

  centreOn(t, length = this.view.length) { this.setView(t - length / 2, length); }

  select(marker, { centre = true, notify = false } = {}) {
    this.selected = marker;
    if (marker && centre && (marker.t < this.view.start || marker.t > this.view.start + this.view.length)) this.centreOn(marker.t);
    else this.draw();
    if (notify) this.onSelect?.(marker);
  }

  setCursor(t) { this.cursor = t; this.draw(); }

  xOf(t) { return LABEL_W + ((t - this.view.start) / this.view.length) * (this.width - LABEL_W); }
  tOf(x) { return this.view.start + ((x - LABEL_W) / (this.width - LABEL_W)) * this.view.length; }

  // ---- layout ----

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.width = this.container.clientWidth;
    if (!this.width) return;
    this.height = this.lanes.reduce((h, lane) => h + (lane.height ?? defaultHeight(lane)), 0) + AXIS;
    for (const [canvas, h] of [[this.overview, 46], [this.main, this.height]]) {
      canvas.width = Math.round(this.width * ratio);
      canvas.height = Math.round(h * ratio);
      canvas.style.height = `${h}px`;
      canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    this.overviewImage = null;
    this.draw();
  }

  // ---- drawing ----

  draw() {
    if (!this.width || !this.lanes.length) return;
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => { this.drawMain(); this.drawOverview(); });
  }

  drawOverview() {
    const ctx = this.overview.getContext("2d");
    const c = colours();
    const w = this.width, h = 46;
    if (!this.overviewImage && this.overviewLane) {
      const off = new OffscreenCanvas(w, h);
      const o = off.getContext("2d");
      o.fillStyle = c.faint;
      drawPeaks(o, this.overviewLane, 0, this.duration, w, 0, h, this.overviewLane.sampleRate);
      this.overviewImage = off;
    }
    ctx.clearRect(0, 0, w, h);
    if (this.overviewImage) ctx.drawImage(this.overviewImage, 0, 0, w, h);
    const x0 = (this.view.start / this.duration) * w, x1 = ((this.view.start + this.view.length) / this.duration) * w;
    ctx.fillStyle = c.accent + "33";
    ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    ctx.strokeStyle = c.accent;
    ctx.strokeRect(x0 + 0.5, 0.5, Math.max(2, x1 - x0) - 1, h - 1);
    if (this.cursor != null) { ctx.fillStyle = c.text; ctx.fillRect((this.cursor / this.duration) * w, 0, 1, h); }
  }

  drawMain() {
    const ctx = this.main.getContext("2d");
    const c = colours();
    ctx.clearRect(0, 0, this.width, this.height);
    let y = 0;
    this.laneBoxes = [];
    for (const lane of this.lanes) {
      const h = lane.height ?? defaultHeight(lane);
      this.laneBoxes.push({ lane, y, h });
      ctx.save();
      ctx.beginPath(); ctx.rect(0, y, this.width, h); ctx.clip();
      if (lane.type === "waveform") this.drawWaveform(ctx, lane, y, h, c);
      else if (lane.type === "spectrogram") this.drawSpectrogram(ctx, lane, y, h);
      else if (lane.type === "curve") this.drawCurve(ctx, lane, y, h, c);
      else if (lane.type === "markers") this.drawMarkers(ctx, lane, y, h, c);
      if (lane.label) {
        ctx.font = "11px system-ui, sans-serif";
        const tw = ctx.measureText(lane.label).width;
        ctx.fillStyle = c.bg + "cc"; ctx.fillRect(4, y + 3, tw + 8, 15);
        ctx.fillStyle = c.muted; ctx.fillText(lane.label, 8, y + 14);
      }
      ctx.restore();
      ctx.fillStyle = c.border; ctx.fillRect(0, y + h - 1, this.width, 1);
      y += h;
    }
    // Guide line for the selected marker and the playback cursor, through every lane.
    if (this.selected) {
      ctx.fillStyle = c.accent + "aa";
      ctx.fillRect(Math.round(this.xOf(this.selected.t)), 0, 1, y);
    }
    if (this.cursor != null) { ctx.fillStyle = c.text; ctx.fillRect(Math.round(this.xOf(this.cursor)), 0, 1, y); }
    this.drawAxis(ctx, y, c);
  }

  drawAxis(ctx, y, c) {
    const { start, length } = this.view;
    const step = niceStep(length / Math.max(1, this.width / 90));
    ctx.font = "10.5px system-ui, sans-serif";
    ctx.fillStyle = c.faint;
    for (let t = Math.ceil(start / step) * step; t <= start + length; t += step) {
      const x = Math.round(this.xOf(t));
      ctx.fillRect(x, y, 1, 4);
      ctx.fillText(step < 1 ? `${t.toFixed(step < 0.01 ? 3 : 2)}s` : `${t.toFixed(0)}s`, x + 3, y + 13);
    }
  }

  drawWaveform(ctx, lane, y, h, c) {
    ctx.fillStyle = lane.colour ?? c.accent;
    drawPeaks(ctx, lane, this.view.start, this.view.start + this.view.length, this.width, y, h, lane.sampleRate);
  }

  drawSpectrogram(ctx, lane, y, h) {
    const s = lane.spectrogram;
    if (!s) return;
    const key = `${this.view.start}|${this.view.length}|${this.width}|${h}|${this.contrast}|${this.whiten}`;
    if (lane.cacheKey !== key) {
      lane.cacheKey = key;
      const image = spectrogramImage(s, this.view.start, this.view.start + this.view.length, Math.round(this.width), h,
        this.contrast, this.whiten);
      lane.bitmap = new OffscreenCanvas(image.width, image.height);
      lane.bitmap.getContext("2d").putImageData(image, 0, 0);
    }
    ctx.drawImage(lane.bitmap, 0, y, this.width, h);
  }

  drawCurve(ctx, lane, y, h, c) {
    const { values } = lane;                  // one value per millisecond
    const max = lane.max ?? Math.max(lane.threshold * 3, 1);
    const toY = (v) => y + h - 3 - (Math.min(v, max) / max) * (h - 8);
    ctx.strokeStyle = lane.colour ?? c.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < this.width; x++) {
      const a = Math.max(0, Math.floor(this.tOf(x) * 1000)), b = Math.min(values.length, Math.max(a + 1, Math.floor(this.tOf(x + 1) * 1000)));
      let v = 0;
      for (let i = a; i < b; i++) if (values[i] > v) v = values[i];
      x ? ctx.lineTo(x, toY(v)) : ctx.moveTo(x, toY(v));
    }
    ctx.stroke();
    if (lane.threshold != null) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = c.warn;
      ctx.beginPath(); ctx.moveTo(0, toY(lane.threshold) + 0.5); ctx.lineTo(this.width, toY(lane.threshold) + 0.5); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const p of lane.points ?? []) {
      if (p.t < this.view.start || p.t > this.view.start + this.view.length) continue;
      ctx.fillStyle = p.colour;
      ctx.beginPath(); ctx.arc(this.xOf(p.t), y + 6, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  drawMarkers(ctx, lane, y, h, c) {
    const { start, length } = this.view;
    const pxPerSecond = this.width / length;
    const showLabels = pxPerSecond > 60;
    ctx.font = "600 12px ui-monospace, monospace";
    ctx.textAlign = "center";
    for (const m of lane.markers) {
      if (m.t < start - 0.05 || m.t > start + length + 0.05) continue;
      const x = Math.round(this.xOf(m.t));
      const selected = this.selected && this.selected.id === m.id;
      ctx.fillStyle = m.colour;
      ctx.strokeStyle = m.colour;
      if (m.thin) { ctx.globalAlpha = 0.7; ctx.fillRect(x, y + h * 0.55, 1, h * 0.45); ctx.globalAlpha = 1; continue; }
      if (showLabels && m.label) {
        const w = Math.max(14, ctx.measureText(m.label).width + 6);
        const top = y + 4, bh = h - 8;
        ctx.globalAlpha = m.hollow ? 0.9 : 0.22;
        if (m.hollow) { ctx.setLineDash([3, 2]); ctx.strokeRect(x - w / 2 + 0.5, top + 0.5, w - 1, bh - 1); ctx.setLineDash([]); }
        else ctx.fillRect(x - w / 2, top, w, bh);
        ctx.globalAlpha = 1;
        if (selected) { ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.strokeRect(x - w / 2 - 1, top - 1, w + 2, bh + 2); ctx.lineWidth = 1; }
        ctx.fillStyle = m.hollow ? c.muted : m.colour;
        ctx.fillText(m.label, x, top + bh / 2 + 4);
      } else {
        ctx.globalAlpha = m.hollow ? 0.5 : 1;
        ctx.fillRect(x, y + 4, selected ? 3 : 1.5, h - 8);
        ctx.globalAlpha = 1;
      }
    }
    ctx.textAlign = "start";
  }

  // ---- interaction ----

  bindEvents() {
    this.main.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = this.main.getBoundingClientRect();
      const t = this.tOf(event.clientX - rect.left);
      if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        const delta = (event.deltaX || event.deltaY) / this.width * this.view.length;
        return this.setView(this.view.start + delta, this.view.length);
      }
      const factor = Math.exp(event.deltaY * 0.0015);
      const length = Math.max(0.05, Math.min(this.duration, this.view.length * factor));
      this.setView(t - ((t - this.view.start) / this.view.length) * length, length);
    }, { passive: false });

    let drag = null;
    this.main.addEventListener("pointerdown", (event) => {
      drag = { x: event.clientX, start: this.view.start, moved: false };
      this.main.setPointerCapture(event.pointerId);
    });
    this.main.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x;
      if (Math.abs(dx) > 3) { drag.moved = true; this.main.classList.add("dragging"); }
      if (drag.moved) this.setView(drag.start - (dx / this.width) * this.view.length, this.view.length);
    });
    this.main.addEventListener("pointerup", (event) => {
      this.main.classList.remove("dragging");
      if (drag && !drag.moved) this.clickAt(event);
      drag = null;
    });
    const seek = (event) => {
      const rect = this.overview.getBoundingClientRect();
      this.centreOn(((event.clientX - rect.left) / rect.width) * this.duration);
    };
    this.overview.addEventListener("pointerdown", (event) => { seek(event); this.overview.setPointerCapture(event.pointerId); this.seeking = true; });
    this.overview.addEventListener("pointermove", (event) => this.seeking && seek(event));
    this.overview.addEventListener("pointerup", () => { this.seeking = false; });
  }

  clickAt(event) {
    const rect = this.main.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    let best = null;
    // Prefer markers in the lane that was clicked, then any lane.
    const box = this.laneBoxes?.find((b) => y >= b.y && y < b.y + b.h);
    const pools = [box?.lane.type === "markers" ? [box.lane] : [], this.lanes.filter((l) => l.type === "markers")];
    for (const lanes of pools) {
      for (const lane of lanes) for (const m of lane.markers) {
        if (m.selectable === false) continue;
        const d = Math.abs(this.xOf(m.t) - x);
        if (d < 12 && (!best || d < best.d)) best = { d, m };
      }
      if (best) break;
    }
    if (best) this.select(best.m, { centre: false, notify: true });
    else this.onSelect?.(null, this.tOf(x));
  }
}

function defaultHeight(lane) {
  return { waveform: 70, spectrogram: 150, curve: 64, markers: 30 }[lane.type];
}

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (raw <= m * p) return m * p;
  return 10 * p;
}

// Min/max of every 64 samples, so whole-recording waveforms draw quickly.
function blockPeaks(samples) {
  const n = Math.ceil(samples.length / 64);
  const min = new Float32Array(n), max = new Float32Array(n);
  for (let b = 0; b < n; b++) {
    let lo = 1, hi = -1;
    for (let i = b * 64, end = Math.min(samples.length, i + 64); i < end; i++) { const v = samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    min[b] = lo; max[b] = hi;
  }
  // Display gain: scale so the loudest 0.05% of blocks fill the lane.
  const sorted = Float32Array.from(max).sort();
  const scale = 1 / Math.max(1e-4, sorted[Math.floor(sorted.length * 0.9995)] || 1);
  return { min, max, scale };
}

function drawPeaks(ctx, lane, from, to, width, y, h, sampleRate = 48000) {
  const { samples, blocks } = lane;
  const scale = (lane.gain ?? 1) * blocks.scale;
  const mid = y + h / 2;
  const perPx = ((to - from) * sampleRate) / width;
  for (let x = 0; x < width; x++) {
    const a = Math.floor((from * sampleRate) + x * perPx), b = Math.max(a + 1, Math.floor(from * sampleRate + (x + 1) * perPx));
    let lo = 0, hi = 0;
    if (perPx > 256) {
      for (let k = Math.max(0, a >> 6), end = Math.min(blocks.min.length, (b >> 6) + 1); k < end; k++) { if (blocks.min[k] < lo) lo = blocks.min[k]; if (blocks.max[k] > hi) hi = blocks.max[k]; }
    } else {
      for (let i = Math.max(0, a), end = Math.min(samples.length, b); i < end; i++) { const v = samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    const top = Math.max(y, mid - hi * scale * (h / 2 - 2)), bottom = Math.min(y + h, mid - lo * scale * (h / 2 - 2));
    ctx.fillRect(x, top, 1, Math.max(1, bottom - top));
  }
}

// Spectrogram pixels for a time range. Levels are bytes in 0.5 dB steps; "whiten" subtracts each frequency's
// median level over the recording (its noise floor).
function spectrogramImage(s, from, to, width, height, contrastDb, whiten) {
  const image = new ImageData(width, height);
  const topBin = Math.min(s.bins - 1, Math.floor(MAX_HZ / s.binHz));
  const globalFloor = whiten ? 0 : median(s.noiseFloor);
  const scale = 255 / (contrastDb * 2);
  const framesPerSecond = s.sampleRate / s.hop;
  const offset = s.size / 2 / s.sampleRate;
  for (let x = 0; x < width; x++) {
    const f0 = Math.max(0, Math.floor((from + ((to - from) * x) / width - offset) * framesPerSecond));
    const f1 = Math.min(s.frames, Math.max(f0 + 1, Math.floor((from + ((to - from) * (x + 1)) / width - offset) * framesPerSecond)));
    for (let row = 0; row < height; row++) {
      const bin = Math.round(((height - 1 - row) / (height - 1)) * topBin);
      let level = 0;
      for (let f = f0; f < f1; f++) { const v = s.levels[f * s.bins + bin]; if (v > level) level = v; }
      const floor = whiten ? s.noiseFloor[bin] : globalFloor;
      const value = f0 < s.frames ? Math.max(0, Math.min(255, ((level - floor) * scale) | 0)) : 0;
      const p = (row * width + x) * 4;
      image.data[p] = COLOURS[value * 3]; image.data[p + 1] = COLOURS[value * 3 + 1]; image.data[p + 2] = COLOURS[value * 3 + 2]; image.data[p + 3] = 255;
    }
  }
  return image;
}

function median(values) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
