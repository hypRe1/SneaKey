import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";

const require = createRequire(import.meta.url);
const C = require("../sneakey/core.js");
const SENTENCES = require("../sneakey/passages.js");
const tmp = mkdtempSync(join(tmpdir(), "sneakey-"));
const py = (code, ...args) => execFileSync("python3", ["-I", "-c", code, ...args], { encoding: "utf8" });

test("WAV is valid 16-bit mono PCM and round-trips samples", () => {
  const sr = 44100;
  const a = new Float32Array(1000).map((_, i) => Math.sin(i / 10) * 0.5);
  const b = new Float32Array(500).fill(1.5);          // clips to full scale
  const c = Float32Array.from([0, -1, 1, -1.2, NaN]);
  const wav = C.encodeWav([a, b, c], sr);
  const p = join(tmp, "t.wav"); writeFileSync(p, wav);
  const out = JSON.parse(py(`
import wave, json, sys, struct
w = wave.open(sys.argv[1]); n = w.getnframes()
d = struct.unpack('<%dh' % n, w.readframes(n))
print(json.dumps({'ch': w.getnchannels(), 'sw': w.getsampwidth(), 'sr': w.getframerate(), 'n': n, 'ct': w.getcomptype(),
  'first': d[:3], 'clip': d[1000], 'tail': d[1500:]}))`, p));
  assert.deepEqual([out.ch, out.sw, out.sr, out.n, out.ct], [1, 2, sr, 1505, "NONE"]);
  assert.equal(out.clip, 32767);
  assert.deepEqual(out.tail, [0, -32768, 32767, -32768, 0]);
  assert.equal(out.first[0], 0);
  assert.equal(wav.length, 44 + 1505 * 2);
  assert.equal(Math.round(a[1] * 32767), out.first[1]);   // no gain/normalisation applied
});

test("CSV escaping handles commas, quotes, newlines and parses back", () => {
  const ev = [
    { event_index: 0, event_type: "keydown", key: ",", code: "Comma", repeat: false, shift: true, correct: true },
    { event_index: 1, event_type: "keydown", key: '"', code: "Quote", repeat: true, correct: false },
    { event_index: 2, event_type: "keydown", key: "Enter", code: "Enter" }
  ];
  const p = join(tmp, "t.csv"); writeFileSync(p, C.buildCsv(ev));
  const rows = JSON.parse(py(`
import csv, json, sys
print(json.dumps(list(csv.DictReader(open(sys.argv[1], newline='')))))`, p));
  assert.equal(rows.length, 3);
  assert.equal(rows[0].key, ","); assert.equal(rows[1].key, '"');
  assert.equal(rows[0].shift, "1"); assert.equal(rows[1].repeat, "1"); assert.equal(rows[0].repeat, "0");
  assert.equal(Object.keys(rows[0]).length, C.CSV_COLUMNS.length);
});

test("sync model recovers offset and slope; falls back to nominal slope on short spans", () => {
  const sr = 48000, slope = sr / 1000 * 1.00002, off = -12345;
  const pts = Array.from({ length: 200 }, (_, i) => { const x = 1e5 + i * 250; return { perf_ms: x, sample: slope * x + off + (i % 2 ? 3 : -3) }; });
  const m = C.buildSyncModel(pts, sr);
  assert.equal(m.slope_source, "fitted");
  assert.ok(Math.abs(m.slope_samples_per_ms - slope) < 1e-4);
  assert.ok(Math.abs(m.sampleAt(1e5 + 7000) - (slope * (1e5 + 7000) + off)) < 3.5);
  assert.ok(Math.abs(m.perfAt(m.sampleAt(123456)) - 123456) < 1e-6);
  const short = C.buildSyncModel(pts.slice(0, 4), sr);
  assert.equal(short.slope_source, "nominal");
  assert.ok(Math.abs(short.sampleAt(pts[0].perf_ms) - pts[0].sample) < 4);
  assert.equal(C.buildSyncModel([], sr), null);
});

test("zip is readable by Python and contents match", () => {
  const files = [
    { name: "session_X/audio.wav", data: new Uint8Array([1, 2, 3, 250, 0, 9]) },
    { name: "session_X/keystrokes.csv", data: new TextEncoder().encode("a,b\n1,2\n") },
    { name: "session_X/metadata.json", data: new TextEncoder().encode("{}") }
  ];
  const p = join(tmp, "t.zip"); writeFileSync(p, C.buildZip(files));
  const out = py(`
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None
print('|'.join('%s:%d' % (i.filename, len(z.read(i))) for i in z.infolist()))`, p).trim();
  assert.equal(out, "session_X/audio.wav:6|session_X/keystrokes.csv:8|session_X/metadata.json:2");
});

test("paragraph generation: 150-200 words, ASCII, varied, favours under-sampled characters", () => {
  const seen = new Set();
  for (let i = 0; i < 50; i++) {
    const t = C.generateParagraph(SENTENCES, {}, Math.random);
    const w = C.countWords(t);
    assert.ok(w >= 150 && w <= 200, "words=" + w);
    assert.ok(/^[\x20-\x7e]+$/.test(t));
    seen.add(t);
  }
  assert.ok(seen.size > 40);
  const all = C.generateParagraph(SENTENCES, {}, Math.random);
  assert.ok(/[A-Z]/.test(all) && /[,.]/.test(all) && / /.test(all));
  const sat = {}; for (const ch of new Set(SENTENCES.join(""))) sat[ch] = 1000;
  delete sat["z"]; delete sat["Z"];
  let biased = 0, plain = 0;
  for (let i = 0; i < 200; i++) {
    if (/[zZ]/.test(C.generateParagraph(SENTENCES, sat, Math.random))) biased++;
    if (/[zZ]/.test(C.generateParagraph(SENTENCES, {}, Math.random))) plain++;
  }
  assert.ok(biased > plain, `biased=${biased} plain=${plain}`);
});

test("typing model preserves wrong keys and backspaces, ignores modifiers, ends at target length", () => {
  const m = new C.TypingModel("Hi!");
  const k = (key, o = {}) => m.keydown({ key, ctrlKey: false, altKey: false, metaKey: false, ...o });
  assert.equal(k("Shift").action, "ignored");
  assert.deepEqual([k("H").correct, k("x").correct], [true, false]);
  assert.equal(m.typedText, "Hx");
  assert.equal(k("Backspace").action, "backspace");
  assert.equal(k("c", { ctrlKey: true }).action, "ignored");
  assert.equal(k("i").correct, true);
  assert.equal(m.done, false);
  assert.equal(k("1").correct, false);          // wrong final char still ends the paragraph
  assert.equal(m.done, true);
  assert.equal(k("a").action, "ignored_done");
  assert.deepEqual(m.keystrokes, { chars: 4, correct: 2, incorrect: 2, backspaces: 1 });
  assert.deepEqual(m.keyCharFreq, { H: 1, x: 1, i: 1, "1": 1 });   // 'x' kept although corrected
  const e = new C.TypingModel("a"); assert.equal(e.keydown({ key: "Backspace" }).action, "backspace_empty");
  const s = C.computeStats(m, 60000);
  assert.equal(s.final_correct_chars, 2); assert.equal(s.final_incorrect_chars, 1);
  assert.equal(s.accuracy_keystrokes_pct, 50);
});
