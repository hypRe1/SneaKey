// Helpers shared by the three views.

export const $ = (id) => document.getElementById(id);

const KEY_SYMBOLS = {
  Space: "␣", Backspace: "⌫", Enter: "⏎", Tab: "⇥", CapsLock: "⇪", ShiftLeft: "⇧", ShiftRight: "⇧",
  ControlLeft: "ctl", MetaLeft: "◆", Comma: ",", Period: ".", Semicolon: ";", Quote: "'", Slash: "/", Backslash: "\\",
  IntlBackslash: "\\", Minus: "-", Equal: "=", BracketLeft: "[", BracketRight: "]", Backquote: "`",
};

// "KeyA" -> "a", "Digit1" -> "1", "Space" -> "␣"
export function keyLabel(code) {
  if (code == null) return "";
  if (code in KEY_SYMBOLS) return KEY_SYMBOLS[code];
  if (/^Key[A-Z]$/.test(code)) return code[3].toLowerCase();
  if (/^Digit\d$/.test(code)) return code[5];
  return code;
}

export const isLetter = (code) => /^Key[A-Z]$/.test(code);

export async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch { /* not JSON */ }
    throw new Error(message);
  }
  return response.headers.get("content-type")?.includes("json") ? response.json() : response;
}

// A message that later errors must not hide (e.g. "restart the server"); other errors are appended to it.
let stickyMessage = "";
export function setStickyError(message) {
  stickyMessage = message;
  showError("");
}

export function showError(message) {
  const box = $("error");
  const text = stickyMessage ? `${stickyMessage}${message ? ` (Last error: ${message})` : ""}` : message || "";
  box.textContent = text;
  box.classList.toggle("hidden", !text);
  if (message) box.scrollIntoView({ block: "nearest" });
}

export const pct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`);
export const secs = (ms) => `${(ms / 1000).toFixed(2)} s`;

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) if (child != null) node.append(child.nodeType ? child : String(child));
  return node;
}

export function roleBadge(role) {
  const titles = {
    unseen: "No model was trained or tuned on this recording: its scores measure generalisation.",
    test: "Held-out test recording (data/splits.json).",
    validation: "Used to choose settings (data/splits.json): scores are mildly optimistic.",
    trained: "The key classifier was trained on this recording: its scores are optimistic.",
    locked: "Reserved for a final evaluation (data/splits.json). Prediction is disabled.",
  };
  return el("span", { class: `badge ${role}`, title: titles[role] || "" }, role);
}

// Reads the CSS colour tokens so canvases follow the theme.
export function colours() {
  const style = getComputedStyle(document.documentElement);
  const get = (name) => style.getPropertyValue(name).trim();
  return { bg: get("--bg"), text: get("--text"), muted: get("--muted"), faint: get("--faint"), border: get("--border"),
    accent: get("--accent"), ok: get("--ok"), bad: get("--bad"), warn: get("--warn"), release: get("--release"),
    candidate: get("--candidate"), surface: get("--surface"), surface3: get("--surface-3") };
}

// Decodes a 16-bit mono WAV (as served by the API) into Float32 samples.
export async function fetchSamples(url) {
  const buffer = await (await api(url)).arrayBuffer();
  const view = new DataView(buffer);
  let offset = 12, rate = 48000, dataStart = 0, dataLength = 0;
  while (offset + 8 <= view.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buffer, offset, 4));
    const size = view.getUint32(offset + 4, true);
    if (id === "fmt ") rate = view.getUint32(offset + 12, true);
    if (id === "data") { dataStart = offset + 8; dataLength = Math.min(size, view.byteLength - dataStart); break; }
    offset += 8 + size + (size % 2);
  }
  const ints = new Int16Array(buffer, dataStart, dataLength >> 1);
  const samples = new Float32Array(ints.length);
  for (let i = 0; i < ints.length; i++) samples[i] = ints[i] / 32768;
  return { samples, sampleRate: rate };
}

// Character-level minimum-edit alignment, for highlighting differences between texts.
export function alignChars(a, b) {
  const n = a.length, m = b.length;
  const cost = new Uint16Array((n + 1) * (m + 1));
  const at = (i, j) => i * (m + 1) + j;
  for (let i = 0; i <= n; i++) cost[at(i, 0)] = i;
  for (let j = 0; j <= m; j++) cost[at(0, j)] = j;
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) {
    cost[at(i, j)] = Math.min(cost[at(i - 1, j)] + 1, cost[at(i, j - 1)] + 1, cost[at(i - 1, j - 1)] + (a[i - 1] !== b[j - 1]));
  }
  const ops = [];
  let i = n, j = m;
  while (i || j) {
    if (i && j && cost[at(i, j)] === cost[at(i - 1, j - 1)] + (a[i - 1] !== b[j - 1])) {
      ops.push([a[i - 1] === b[j - 1] ? "eq" : "sub", a[i - 1], b[j - 1]]); i--; j--;
    } else if (i && cost[at(i, j)] === cost[at(i - 1, j)] + 1) { ops.push(["del", a[i - 1], ""]); i--; }
    else { ops.push(["ins", "", b[j - 1]]); j--; }
  }
  return ops.reverse();
}
