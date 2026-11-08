// A small, dependency-free QR code encoder (byte mode), so the phone pairing link can be scanned instead of
// typed. Nothing is loaded from the network. Supports versions 1-10 at error-correction level M, which covers
// URLs up to ~250 bytes. Returns a boolean matrix (true = dark module).
//
// Verified against zbarimg / OpenCV in tests/qr_check.py.

// ---- Galois field GF(256) for Reed-Solomon ----
const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
(() => {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
})();
const mul = (a, b) => (a === 0 || b === 0 ? 0 : EXP[LOG[a] + LOG[b]]);

function rsGenerator(degree) {
  let poly = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array(poly.length + 1).fill(0);
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= mul(poly[j], EXP[i]);
      next[j + 1] ^= poly[j];
    }
    poly = next;
  }
  return poly.reverse();        // leading coefficient first, as rsEncode's synthetic division expects
}

function rsEncode(data, ecLength) {
  const gen = rsGenerator(ecLength);
  const result = new Array(ecLength).fill(0);
  for (const byte of data) {
    const factor = byte ^ result[0];
    result.shift();
    result.push(0);
    for (let j = 0; j < gen.length - 1; j++) result[j] ^= mul(gen[j + 1], factor);
  }
  return result;
}

// ---- version data (level M): total data codewords, EC per block, block layout ----
// [dataCodewords, ecPerBlock, group1Blocks, group1DataPerBlock, group2Blocks, group2DataPerBlock]
const VERSIONS_M = {
  1: [16, 10, 1, 16, 0, 0], 2: [28, 16, 1, 28, 0, 0], 3: [44, 26, 1, 44, 0, 0],
  4: [64, 18, 2, 32, 0, 0], 5: [86, 24, 2, 43, 0, 0], 6: [108, 16, 4, 27, 0, 0],
};
const ALIGN_POS = { 1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34] };

// Versions 1-6 only: single-byte character-count field, no version-information blocks, and verified to scan
// (tests/qr_check.py). Version 6 holds 84 bytes at level M, far more than any pairing URL (~40 bytes).
const MAX_VERSION = 6;

function chooseVersion(byteLength) {
  for (let v = 1; v <= MAX_VERSION; v++) {
    const needed = Math.ceil((4 + 8 + byteLength * 8) / 8);
    if (needed <= VERSIONS_M[v][0]) return v;
  }
  throw new Error("Text too long for this QR encoder (max 84 bytes).");
}

// ---- bit stream -> data codewords ----
function dataCodewords(text, version) {
  const bytes = new TextEncoder().encode(text);
  const [capacity] = VERSIONS_M[version];
  const bits = [];
  const push = (value, count) => { for (let i = count - 1; i >= 0; i--) bits.push((value >> i) & 1); };
  push(0b0100, 4);                                   // byte mode
  push(bytes.length, 8);
  for (const b of bytes) push(b, 8);
  for (let i = 0; i < 4 && bits.length < capacity * 8; i++) bits.push(0);   // terminator
  while (bits.length % 8) bits.push(0);
  const codewords = [];
  for (let i = 0; i < bits.length; i += 8) codewords.push(parseInt(bits.slice(i, i + 8).join(""), 2));
  for (let pad = 0; codewords.length < capacity; pad++) codewords.push(pad % 2 ? 0x11 : 0xec);
  return codewords;
}

// ---- interleave data + EC across blocks ----
function finalCodewords(data, version) {
  const [, ecLen, g1, d1, g2, d2] = VERSIONS_M[version];
  const blocks = [];
  let at = 0;
  for (let i = 0; i < g1; i++) { blocks.push(data.slice(at, at + d1)); at += d1; }
  for (let i = 0; i < g2; i++) { blocks.push(data.slice(at, at + d2)); at += d2; }
  const ec = blocks.map((b) => rsEncode(b, ecLen));
  const out = [];
  const maxData = Math.max(d1, d2);
  for (let i = 0; i < maxData; i++) for (const b of blocks) if (i < b.length) out.push(b[i]);
  for (let i = 0; i < ecLen; i++) for (const e of ec) out.push(e[i]);
  return out;
}

// ---- matrix construction ----
function build(version, codewords) {
  const size = version * 4 + 17;
  const m = Array.from({ length: size }, () => new Array(size).fill(null));
  const reserved = Array.from({ length: size }, () => new Array(size).fill(false));
  const set = (r, c, v) => { m[r][c] = v; reserved[r][c] = true; };

  const finder = (r, c) => {
    for (let i = -1; i <= 7; i++) for (let j = -1; j <= 7; j++) {
      const rr = r + i, cc = c + j;
      if (rr < 0 || cc < 0 || rr >= size || cc >= size) continue;
      const edge = i === -1 || i === 7 || j === -1 || j === 7;
      const dark = i >= 0 && i <= 6 && j >= 0 && j <= 6 && (i === 0 || i === 6 || j === 0 || j === 6 || (i >= 2 && i <= 4 && j >= 2 && j <= 4));
      set(rr, cc, edge ? false : dark);
    }
  };
  finder(0, 0); finder(0, size - 7); finder(size - 7, 0);

  for (let i = 8; i < size - 8; i++) {               // timing patterns
    set(6, i, i % 2 === 0);
    set(i, 6, i % 2 === 0);
  }
  const aligns = ALIGN_POS[version];                 // alignment patterns
  for (const r of aligns) for (const c of aligns) {
    if (reserved[r][c]) continue;
    for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++)
      set(r + i, c + j, Math.max(Math.abs(i), Math.abs(j)) !== 1);
  }
  set(size - 8, 8, true);                            // dark module
  for (let i = 0; i < 9; i++) { if (!reserved[8][i]) set(8, i, false); if (!reserved[i][8]) set(i, 8, false); }
  for (let i = 0; i < 8; i++) { if (!reserved[8][size - 1 - i]) set(8, size - 1 - i, false); if (!reserved[size - 1 - i][8]) set(size - 1 - i, 8, false); }

  // Place data bits (zigzag, upward/downward columns), skipping reserved modules.
  const bits = [];
  for (const cw of codewords) for (let i = 7; i >= 0; i--) bits.push((cw >> i) & 1);
  let bit = 0, upward = true;
  for (let col = size - 1; col > 0; col -= 2) {
    if (col === 6) col--;                            // skip the timing column
    for (let k = 0; k < size; k++) {
      const row = upward ? size - 1 - k : k;
      for (const c of [col, col - 1]) {
        if (reserved[row][c]) continue;
        m[row][c] = bit < bits.length ? bits[bit++] === 1 : false;
      }
    }
    upward = !upward;
  }
  return applyBestMask(m, reserved, version);
}

const MASKS = [
  (r, c) => (r + c) % 2 === 0,
  (r, c) => r % 2 === 0,
  (r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
];

function applyBestMask(m, reserved, version) {
  const size = m.length;
  let best = null;
  for (let mask = 0; mask < 8; mask++) {
    const trial = m.map((row) => row.slice());
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++)
      if (!reserved[r][c] && MASKS[mask](r, c)) trial[r][c] = !trial[r][c];
    writeFormat(trial, mask);
    const penalty = score(trial);
    if (best === null || penalty < best.penalty) best = { penalty, matrix: trial };
  }
  return best.matrix;
}

// The 15-bit format string is placed most-significant-bit first, into two copies at fixed positions (verified
// against the qrcode library and zbar in tests/qr_check.py).
const FORMAT_COPY1 = [[8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 7], [8, 8], [7, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8]];

function writeFormat(m, mask) {
  const size = m.length;
  const bits = (0b00 << 3) | mask;                    // level M = 0b00
  let rem = bits;
  for (let i = 0; i < 10; i++) { rem <<= 1; if (rem & 0x400) rem ^= 0x537; }
  const format = (((bits << 10) | rem) ^ 0b101010000010010) & 0x7fff;
  const copy2 = [];
  for (let i = 0; i < 7; i++) copy2.push([size - 1 - i, 8]);   // up column 8 (the dark module is not a format cell)
  for (let i = 0; i < 8; i++) copy2.push([8, size - 8 + i]);   // along row 8
  for (let j = 0; j < 15; j++) {
    const v = ((format >> (14 - j)) & 1) === 1;
    m[FORMAT_COPY1[j][0]][FORMAT_COPY1[j][1]] = v;
    m[copy2[j][0]][copy2[j][1]] = v;
  }
}


function score(m) {
  const size = m.length;
  let penalty = 0;
  // Rule 1: runs of 5+ same-colour modules in rows and columns.
  for (let r = 0; r < size; r++) for (const line of [m[r], m.map((row) => row[r])]) {
    let run = 1;
    for (let c = 1; c < size; c++) {
      if (line[c] === line[c - 1]) { run++; if (run === 5) penalty += 3; else if (run > 5) penalty++; }
      else run = 1;
    }
  }
  // Rule 2: 2x2 blocks of the same colour.
  for (let r = 0; r < size - 1; r++) for (let c = 0; c < size - 1; c++)
    if (m[r][c] === m[r][c + 1] && m[r][c] === m[r + 1][c] && m[r][c] === m[r + 1][c + 1]) penalty += 3;
  // Rule 3: finder-like 1:1:3:1:1 patterns.
  const pat1 = [true, false, true, true, true, false, true, false, false, false, false];
  const pat2 = [false, false, false, false, true, false, true, true, true, false, true];
  for (let r = 0; r < size; r++) for (let c = 0; c <= size - 11; c++) {
    const row = m[r].slice(c, c + 11), col = m.slice(r, r + 11).map((x) => x[c]);
    if (pat1.every((v, i) => v === row[i]) || pat2.every((v, i) => v === row[i])) penalty += 40;
    if (r <= size - 11 && (pat1.every((v, i) => v === col[i]) || pat2.every((v, i) => v === col[i]))) penalty += 40;
  }
  // Rule 4: deviation from 50% dark.
  let dark = 0;
  for (const row of m) for (const v of row) if (v) dark++;
  penalty += Math.floor(Math.abs((dark * 100) / (size * size) - 50) / 5) * 10;
  return penalty;
}

export function qrMatrix(text) {
  const bytes = new TextEncoder().encode(text).length;
  const version = chooseVersion(bytes);
  const data = dataCodewords(text, version);
  const codewords = finalCodewords(data, version);
  return build(version, codewords);
}

// Draws the matrix into a canvas with a quiet zone, sized to fit `pixels`.
export function drawQr(canvas, text, pixels = 240) {
  const m = qrMatrix(text);
  const quiet = 4;
  const size = m.length + quiet * 2;
  const scale = Math.max(1, Math.floor(pixels / size));
  const dim = size * scale;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = dim * ratio;
  canvas.height = dim * ratio;
  canvas.style.width = `${dim}px`;
  canvas.style.height = `${dim}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, dim, dim);
  ctx.fillStyle = "#000000";
  for (let r = 0; r < m.length; r++) for (let c = 0; c < m.length; c++)
    if (m[r][c]) ctx.fillRect((c + quiet) * scale, (r + quiet) * scale, scale, scale);
  return dim;
}
