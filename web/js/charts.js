// Small canvas charts: heatmaps, line charts and probability bars.
import { COLOURS } from "./spectrogram.js";
import { colours, el, keyLabel } from "./common.js";

function prepare(canvas, height) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

const colourFor = (v) => {   // v in 0..1 -> spectrogram palette
  const i = Math.max(0, Math.min(255, Math.round(v * 255))) * 3;
  return `rgb(${COLOURS[i]},${COLOURS[i + 1]},${COLOURS[i + 2]})`;
};

// matrix[row][col]; rows drawn bottom-up when flipY. Values are normalised to the matrix's own range.
export function heatmap(canvas, matrix, { height = 200, flipY = false, xLabels, yLabels, marks = [], range, labelWidth = 0, labelHeight = 0, title } = {}) {
  const { ctx, width } = prepare(canvas, height);
  const c = colours();
  const rows = matrix.length, cols = matrix[0]?.length ?? 0;
  if (!rows || !cols) return;
  let [lo, hi] = range ?? [Infinity, -Infinity];
  if (!range) for (const r of matrix) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const cw = (width - labelWidth) / cols, ch = (height - labelHeight) / rows;
  for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
    ctx.fillStyle = colourFor(Math.max(0, Math.min(1, (matrix[r][k] - lo) / (hi - lo || 1))));
    const y = flipY ? (rows - 1 - r) * ch : r * ch;
    ctx.fillRect(labelWidth + k * cw, y, Math.ceil(cw), Math.ceil(ch));
  }
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = c.muted;
  if (yLabels) yLabels.forEach((label, r) => { if (label) ctx.fillText(label, 2, (flipY ? rows - 1 - r : r) * ch + ch / 2 + 3.5); });
  if (xLabels) {
    ctx.textAlign = "center";
    xLabels.forEach((label, k) => { if (label) ctx.fillText(label, labelWidth + k * cw + cw / 2, height - 3); });
    ctx.textAlign = "start";
  }
  for (const m of marks) {            // {row, col, colour}
    ctx.strokeStyle = m.colour; ctx.lineWidth = 2;
    const y = flipY ? (rows - 1 - m.row) * ch : m.row * ch;
    ctx.strokeRect(labelWidth + m.col * cw + 1, y + 1, cw - 2, ch - 2);
  }
  if (title) { ctx.fillStyle = c.text; ctx.font = "11px system-ui"; ctx.fillText(title, labelWidth + 4, 12); }
}

// series: [{values, colour, dashed}], x from xs (same length as values).
export function lineChart(canvas, xs, series, { height = 190, xLabel = "", yLabel = "", xTicks = [], logX = false } = {}) {
  const { ctx, width } = prepare(canvas, height);
  const c = colours();
  const left = 36, bottom = 22, top = 8, right = 8;
  const all = series.flatMap((s) => s.values.filter(Number.isFinite));
  const lo = Math.min(...all), hi = Math.max(...all);
  const fx = logX ? Math.log10 : (v) => v;
  const x0 = fx(xs[0] || 1), x1 = fx(xs[xs.length - 1]);
  const X = (v) => left + ((fx(v) - x0) / (x1 - x0)) * (width - left - right);
  const Y = (v) => top + (1 - (v - lo) / (hi - lo || 1)) * (height - top - bottom);
  ctx.font = "10px system-ui"; ctx.fillStyle = c.faint; ctx.strokeStyle = c.border;
  for (const t of xTicks) { const x = X(t); ctx.fillRect(x, height - bottom, 1, 4); ctx.fillText(t >= 1000 ? `${t / 1000}k` : `${t}`, x - 6, height - 6); }
  for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4; ctx.fillText(v.toFixed(hi - lo < 10 ? 1 : 0), 2, Y(v) + 3); ctx.fillRect(left, Y(v), width - left - right, 0.5); }
  ctx.fillText(xLabel, width - right - ctx.measureText(xLabel).width, height - 6);
  for (const s of series) {
    ctx.strokeStyle = s.colour; ctx.lineWidth = 1.5; ctx.setLineDash(s.dashed ? [4, 3] : []);
    ctx.beginPath();
    s.values.forEach((v, i) => { if (!Number.isFinite(v) || xs[i] <= 0 && logX) return; i ? ctx.lineTo(X(xs[i]), Y(v)) : ctx.moveTo(X(xs[i]), Y(v)); });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (yLabel) { ctx.fillStyle = c.faint; ctx.fillText(yLabel, left + 4, top + 10); }
}

// Probability bars: items [{key, probability}], truth = key code to highlight.
export function probabilityBars(items, truth, { compact = false } = {}) {
  return el("div", { class: "bars" }, items.flatMap(({ key, probability }) => {
    const isTruth = key === truth;
    const bar = el("div", { class: `bar${isTruth ? " truth" : ""}` }, el("div", { style: `width:${(probability * 100).toFixed(1)}%` }));
    return [el("span", { class: `k${isTruth ? " truth" : ""}` }, compact ? keyLabel(key) : `${keyLabel(key)}`),
      bar, el("span", { class: "num muted" }, `${(probability * 100).toFixed(probability < 0.1 ? 1 : 0)}%`)];
  }));
}
