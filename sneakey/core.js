// SneaKey core: pure, DOM-free logic shared by the browser app and the Node tests.
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- WAV (PCM16)

  // Float32 [-1,1] -> Int16, clamped (no normalisation or gain of any kind).
  function floatToInt16(f) {
    const out = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      let s = f[i];
      if (!(s === s)) s = 0; // NaN guard
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      out[i] = Math.round(s < 0 ? s * 32768 : s * 32767);
    }
    return out;
  }

  // chunks: array of Float32Array (mono). Returns Uint8Array containing a canonical 44-byte-header WAV.
  function encodeWav(chunks, sampleRate) {
    let frames = 0;
    for (const c of chunks) frames += c.length;
    const dataBytes = frames * 2;
    const buf = new ArrayBuffer(44 + dataBytes);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + dataBytes, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);              // PCM
    v.setUint16(22, 1, true);              // mono
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * 2, true); // byte rate
    v.setUint16(32, 2, true);              // block align
    v.setUint16(34, 16, true);             // bits per sample
    str(36, "data"); v.setUint32(40, dataBytes, true);
    let off = 44;
    for (const c of chunks) {
      const i16 = floatToInt16(c);
      for (let i = 0; i < i16.length; i++, off += 2) v.setInt16(off, i16[i], true);
    }
    return new Uint8Array(buf);
  }

  // ---------------------------------------------------------------- CSV

  const CSV_COLUMNS = [
    "event_index", "event_type", "key", "code", "location",
    "time_ms", "event_time_ms", "t_rel_ms", "sample_index_est",
    "repeat", "shift", "ctrl", "alt", "meta",
    "action", "target_char", "typed_char", "correct", "cursor_before", "cursor_after"
  ];

  function csvField(v) {
    if (v === null || v === undefined) return "";
    if (typeof v === "boolean") return v ? "1" : "0";
    const s = String(v);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function buildCsv(events) {
    const lines = [CSV_COLUMNS.join(",")];
    for (const e of events) lines.push(CSV_COLUMNS.map((c) => csvField(e[c])).join(","));
    return lines.join("\n") + "\n";
  }

  // ---------------------------------------------------------------- Clock sync

  // Ordinary least squares y = slope*x + intercept, computed on centred x for precision.
  function linearFit(points) {
    const n = points.length;
    if (n === 0) return null;
    let mx = 0, my = 0;
    for (const p of points) { mx += p.x; my += p.y; }
    mx /= n; my /= n;
    let sxx = 0, sxy = 0;
    for (const p of points) { sxx += (p.x - mx) ** 2; sxy += (p.x - mx) * (p.y - my); }
    const slope = sxx > 0 ? sxy / sxx : null;
    return { n, mx, my, slope };
  }

  // A slope fitted over a short span is dominated by timer jitter (a few ms of noise over ~5 s is hundreds of ppm,
  // far more than real audio-clock drift), so below this span only the offset is fitted.
  const MIN_FIT_SPAN_MS = 15000;
  // Real oscillators differ by well under ~100 ppm; a fitted slope further than 500 ppm from nominal is jitter, not drift.
  const MAX_PLAUSIBLE_DRIFT = 0.0005;

  // points: [{perf_ms, sample}] where `sample` is the audio sample index at that performance.now() instant.
  // Returns a model mapping performance.now() <-> sample index. If the points span too little time to give a
  // trustworthy slope, the nominal slope (sampleRate/1000 samples per ms) is used and only the offset is fitted.
  function buildSyncModel(points, sampleRate) {
    const nominal = sampleRate / 1000;
    if (!points || points.length === 0) return null;
    const pts = points.map((p) => ({ x: p.perf_ms, y: p.sample }));
    const xs = pts.map((p) => p.x);
    const span = Math.max(...xs) - Math.min(...xs);
    const f = linearFit(pts);
    let slope = f.slope, slopeSource = "fitted";
    if (slope === null || span < MIN_FIT_SPAN_MS || Math.abs(slope / nominal - 1) > MAX_PLAUSIBLE_DRIFT) {
      slope = nominal; slopeSource = "nominal";
    }
    const intercept = f.my - slope * f.mx;
    let sq = 0, maxAbs = 0;
    for (const p of pts) {
      const r = p.y - (slope * p.x + intercept);
      sq += r * r; if (Math.abs(r) > maxAbs) maxAbs = Math.abs(r);
    }
    return {
      n: pts.length,
      slope_samples_per_ms: slope,
      slope_source: slopeSource,
      effective_sample_rate_hz: slope * 1000,
      intercept_samples: intercept,
      residual_rms_samples: Math.sqrt(sq / pts.length),
      residual_max_samples: maxAbs,
      perf_span_ms: span,
      sampleAt: (perf_ms) => slope * perf_ms + intercept,
      perfAt: (sample) => (sample - intercept) / slope
    };
  }

  // Strip functions so a model can be serialised to JSON.
  function modelToJson(m) {
    if (!m) return null;
    const { sampleAt, perfAt, ...rest } = m; // eslint-disable-line no-unused-vars
    return rest;
  }

  // ---------------------------------------------------------------- ZIP (stored, no compression)

  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  // files: [{name: "dir/file", data: Uint8Array}] -> Uint8Array
  function buildZip(files, date) {
    const enc = new TextEncoder();
    const d = date || new Date();
    const dosTime = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
    const dosDate = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    const parts = [], central = [];
    let offset = 0;
    for (const f of files) {
      const name = enc.encode(f.name);
      const crc = crc32(f.data);
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true);
      lh.setUint16(8, 0, true); lh.setUint16(10, dosTime, true); lh.setUint16(12, dosDate, true);
      lh.setUint32(14, crc, true); lh.setUint32(18, f.data.length, true); lh.setUint32(22, f.data.length, true);
      lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
      parts.push(new Uint8Array(lh.buffer), name, f.data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true);
      ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true); ch.setUint16(12, dosTime, true);
      ch.setUint16(14, dosDate, true); ch.setUint32(16, crc, true);
      ch.setUint32(20, f.data.length, true); ch.setUint32(24, f.data.length, true);
      ch.setUint16(28, name.length, true); ch.setUint32(42, offset, true);
      central.push(new Uint8Array(ch.buffer), name);
      offset += 30 + name.length + f.data.length;
    }
    let cdSize = 0;
    for (const c of central) cdSize += c.length;
    const eocd = new DataView(new ArrayBuffer(22));
    eocd.setUint32(0, 0x06054b50, true);
    eocd.setUint16(8, files.length, true); eocd.setUint16(10, files.length, true);
    eocd.setUint32(12, cdSize, true); eocd.setUint32(16, offset, true);
    const all = [...parts, ...central, new Uint8Array(eocd.buffer)];
    let total = 0;
    for (const p of all) total += p.length;
    const out = new Uint8Array(total);
    let o = 0;
    for (const p of all) { out.set(p, o); o += p.length; }
    return out;
  }

  // ---------------------------------------------------------------- Text generation / stats

  function countWords(s) { return s.trim() === "" ? 0 : s.trim().split(/\s+/).length; }

  function charFrequency(str) {
    const f = {};
    for (const ch of str) f[ch] = (f[ch] || 0) + 1;
    return f;
  }

  // Build a ~150-200 word paragraph from the bundled sentences. Sentences containing characters that are
  // still under-sampled (per `counts`, the cumulative typed-character counts) are drawn more often.
  function generateParagraph(sentences, counts, rng, opts) {
    const o = Object.assign({ minWords: 150, maxWords: 200, targetPerChar: 30 }, opts || {});
    rng = rng || Math.random;
    counts = counts || {};
    const need = (c) => Math.max(0, 1 - (counts[c] || 0) / o.targetPerChar);
    let pool = sentences.map((s) => ({ s, w: 1 + [...new Set(s)].reduce((a, c) => a + need(c), 0), n: countWords(s) }));
    const picked = [];
    let words = 0;
    while (words < o.minWords && pool.length) {
      const fit = pool.filter((p) => words + p.n <= o.maxWords);
      if (!fit.length) break;
      let r = rng() * fit.reduce((a, p) => a + p.w, 0);
      let choice = fit[fit.length - 1];
      for (const p of fit) { r -= p.w; if (r <= 0) { choice = p; break; } }
      picked.push(choice.s);
      words += choice.n;
      pool = pool.filter((p) => p !== choice);
    }
    return picked.join(" ");
  }

  // Typing state machine. Decides what a keydown does to the typed text; it never discards anything,
  // it only reports the action so the caller can log every event.
  class TypingModel {
    constructor(target) {
      this.target = target;
      this.typed = [];
      this.keystrokes = { chars: 0, correct: 0, incorrect: 0, backspaces: 0 };
      this.keyCharFreq = {};   // every character keystroke, including ones later corrected
    }
    get done() { return this.typed.length >= this.target.length; }
    // ev: {key, ctrlKey, altKey, metaKey}
    keydown(ev) {
      const before = this.typed.length;
      const r = { action: "ignored", expected: this.target[before] ?? "", typed: "", correct: null, before, after: before };
      if (this.done) { r.action = "ignored_done"; return r; }
      if (ev.ctrlKey || ev.altKey || ev.metaKey) return r;
      if (ev.key === "Backspace") {
        if (before > 0) { this.typed.pop(); r.action = "backspace"; this.keystrokes.backspaces++; }
        else r.action = "backspace_empty";
        r.after = this.typed.length;
        return r;
      }
      if (ev.key.length === 1) {
        const ch = ev.key;
        this.typed.push(ch);
        const ok = ch === this.target[before];
        this.keystrokes.chars++;
        ok ? this.keystrokes.correct++ : this.keystrokes.incorrect++;
        this.keyCharFreq[ch] = (this.keyCharFreq[ch] || 0) + 1;
        r.action = "char"; r.typed = ch; r.correct = ok; r.after = this.typed.length;
      }
      return r;
    }
    get typedText() { return this.typed.join(""); }
    finalCounts() {
      let correct = 0, incorrect = 0;
      this.typed.forEach((c, i) => (c === this.target[i] ? correct++ : incorrect++));
      return { correct, incorrect, total: this.typed.length };
    }
  }

  function computeStats(model, elapsedMs) {
    const fc = model.finalCounts();
    const k = model.keystrokes;
    const minutes = Math.max(elapsedMs, 1) / 60000;
    return {
      wpm: +((fc.correct / 5) / minutes).toFixed(2),
      raw_wpm: +((k.chars / 5) / minutes).toFixed(2),
      accuracy_keystrokes_pct: k.chars ? +((k.correct / k.chars) * 100).toFixed(2) : 100,
      accuracy_final_text_pct: fc.total ? +((fc.correct / fc.total) * 100).toFixed(2) : 100,
      final_correct_chars: fc.correct,
      final_incorrect_chars: fc.incorrect,
      final_total_chars: fc.total,
      char_keystrokes: k.chars,
      correct_keystrokes: k.correct,
      incorrect_keystrokes: k.incorrect,
      backspace_keystrokes: k.backspaces,
      typing_elapsed_ms: Math.round(elapsedMs)
    };
  }

  const api = {
    floatToInt16, encodeWav, CSV_COLUMNS, buildCsv, csvField,
    linearFit, buildSyncModel, modelToJson, crc32, buildZip,
    countWords, charFrequency, generateParagraph, TypingModel, computeStats
  };
  root.SneaKeyCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
