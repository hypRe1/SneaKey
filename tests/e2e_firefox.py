"""End-to-end test: drives record_keystrokes.html in headless Firefox (fake microphone) through two sessions,
then validates the exported files independently in Python.  Run: python3 tests/e2e_firefox.py  (server on :8000)."""
import base64, csv, io, json, os, sys, wave, zipfile, struct, tempfile, re
sys.path.insert(0, os.path.dirname(__file__))
from marionette import Firefox

URL = "http://localhost:8000/record_keystrokes.html"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JS helper injected into the page: types text through real keydown/keyup events on the typing element.
JS_TYPER = r"""
const cb = arguments[arguments.length - 1];
const [mode, limit] = [arguments[0], arguments[1]];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const el = document.getElementById('typing');
const SH = {'!':'Digit1','?':'Slash','"':'Quote',':':'Semicolon'};
const PL = {',':'Comma','.':'Period',';':'Semicolon',"'":'Quote','-':'Minus','/':'Slash',' ':'Space','1':'Digit1','2':'Digit2','4':'Digit4','9':'Digit9','5':'Digit5','8':'Digit8','0':'Digit0'};
function codeOf(ch){ if(/[a-z]/i.test(ch)) return 'Key'+ch.toUpperCase(); if(SH[ch]) return SH[ch]; return PL[ch] || ('Unknown_'+ch); }
const sent = [];
function fire(type, key, code, o){
  o = o || {};
  const ev = new KeyboardEvent(type, {key, code, repeat: !!o.repeat, shiftKey: !!o.shift, bubbles: true, cancelable: true});
  sent.push({type, key, code, repeat: !!o.repeat, shift: !!o.shift});
  el.dispatchEvent(ev);
}
async function press(ch, o){
  o = o || {};
  const shifted = /[A-Z]/.test(ch) || '!?":'.includes(ch);
  if (shifted) { fire('keydown','Shift','ShiftLeft',{shift:true}); await sleep(3); }
  fire('keydown', ch, codeOf(ch), {shift: shifted}); await sleep(3);
  if (o.repeat) { fire('keydown', ch, codeOf(ch), {shift: shifted, repeat: true}); await sleep(3); }
  fire('keyup', ch, codeOf(ch), {shift: shifted}); await sleep(3);
  if (shifted) { fire('keyup','Shift','ShiftLeft',{shift:false}); await sleep(3); }
}
async function bksp(){ fire('keydown','Backspace','Backspace'); await sleep(3); fire('keyup','Backspace','Backspace'); await sleep(3); }
(async () => {
  document.getElementById('start').click();
  for (let i = 0; i < 100 && document.getElementById('start').disabled === false; i++) await sleep(50);
  for (let i = 0; i < 200 && !document.getElementById('rec').classList.contains('on'); i++) await sleep(50);
  if (!document.getElementById('rec').classList.contains('on')) return cb({error: 'recording did not start: ' + document.getElementById('status').textContent});
  el.focus();
  const status0 = document.getElementById('status').textContent;
  const target = Array.from(document.querySelectorAll('#text .c')).map(s => s.textContent).join('');
  await sleep(1200);                                  // some leading audio before the first key
  const n = Math.min(target.length, limit);
  for (let i = 0; i < n; i++) {
    const ch = target[i];
    if (i % 23 === 5 && i < target.length - 2) { await press(ch === 'e' ? 'a' : 'e'); await bksp(); }   // wrong char then correct
    if (i % 41 === 7 && i < target.length - 3) { await press(ch, {repeat: true}); await bksp(); continue; }  // held key (auto-repeat) then fix
    await press(ch);
    await sleep(8 + (i % 5) * 4);
  }
  if (n < target.length) { await sleep(600); document.getElementById('finish').click(); }
  for (let i = 0; i < 200 && !window.sneakeyLastSession; i++) await sleep(50);
  const S = window.sneakeyLastSession;
  if (!S) return cb({error: 'no export; status=' + document.getElementById('status').textContent});
  const b64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
  const links = Array.from(document.querySelectorAll('#downloads a')).map(a => ({href: a.href.slice(0, 5), download: a.download}));
  cb({sent, status0, target, wav: b64(S.wav), zip: b64(S.zip), csv: S.csv, metadata: S.metadata, links,
      summaryVisible: document.getElementById('summary').classList.contains('on'),
      coverage: JSON.parse(localStorage.getItem('sneakey.charCounts.v1') || '{}'),
      summaryText: document.getElementById('kv').innerText});
  window.sneakeyLastSession = null;
})().catch(e => cb({error: String(e) + ' ' + (e.stack || '')}));
"""

fails = []
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond: fails.append(msg)

def validate(label, r, outdir):
    print(f"\n=== {label} ===")
    if "error" in r:
        check(False, "session ran: " + r["error"]); return None
    md = r["metadata"]; sent = r["sent"]
    folder = md["folder"]
    d = os.path.join(outdir, folder); os.makedirs(d)
    wavb = base64.b64decode(r["wav"]); open(os.path.join(d, "audio.wav"), "wb").write(wavb)
    open(os.path.join(d, "keystrokes.csv"), "w", newline="").write(r["csv"])
    open(os.path.join(d, "metadata.json"), "w").write(json.dumps(md, indent=2))

    # 1. WAV validity -------------------------------------------------------------------
    w = wave.open(os.path.join(d, "audio.wav"))
    n = w.getnframes(); sr = w.getframerate()
    pcm = struct.unpack("<%dh" % n, w.readframes(n))
    check((w.getnchannels(), w.getsampwidth(), w.getcomptype()) == (1, 2, "NONE"), f"WAV is mono 16-bit PCM (sr={sr}, frames={n}, {n/sr:.2f}s)")
    check(wavb[:4] == b"RIFF" and wavb[8:12] == b"WAVE" and len(wavb) == 44 + 2 * n, "RIFF/WAVE header + exact byte length")
    check(md["audio"]["sample_rate_hz"] == sr and md["audio"]["frames"] == n and md["audio"]["channels"] == 1, "metadata sample rate / frames / channels match the WAV")
    check(max(abs(x) for x in pcm) > 100 and not md["audio"]["silent"], f"WAV is not silent (peak int16={max(abs(x) for x in pcm)})")
    check(sr in (44100, 48000), f"sample rate is the AudioContext/device rate ({sr}) - not a hardcoded value")
    c = md["capture"]
    check(c["echo_cancellation"] == c["noise_suppression"] == c["auto_gain_control"] == "disabled", "echo cancellation / noise suppression / AGC reported disabled by Firefox")

    # 2. key events labelled correctly --------------------------------------------------
    rows = list(csv.DictReader(io.StringIO(r["csv"], newline="")))
    check(len(rows) == len(sent) == md["events"]["total"], f"one CSV row per dispatched event ({len(rows)} rows == {len(sent)} sent)")
    mism = [i for i, (a, b) in enumerate(zip(rows, sent)) if (a["event_type"], a["key"], a["code"], a["repeat"] == "1", a["shift"] == "1") != (b["type"], b["key"], b["code"], b["repeat"], b["shift"])]
    check(not mism, f"event_type/key/code/repeat/shift of every row matches what was sent ({len(mism)} mismatches)")
    sh_sent = [(b["type"], b["shift"]) for b in sent if b["key"] == "Shift"]
    sh_rows = [(a["event_type"], a["shift"] == "1") for a in rows if a["key"] == "Shift" and a["code"] == "ShiftLeft"]
    check(len(sh_rows) > 0 and sh_rows == sh_sent, f"Shift recorded as its own keydown/keyup rows ({len(sh_rows)}), matching what was sent")
    caps = [a for a in rows if a["event_type"] == "keydown" and re.fullmatch(r"[A-Z]", a["key"])]
    check(caps and all(a["code"] == "Key" + a["key"] and a["shift"] == "1" for a in caps), f"capital letters: key='H' vs code='KeyH' distinguished, shift=1 ({len(caps)} samples)")

    # 3. repeats -------------------------------------------------------------------------
    rep = [a for a in rows if a["repeat"] == "1"]
    check(len(rep) == md["events"]["repeated_keydown"] == sum(1 for s in sent if s["repeat"]) and len(rep) > 0, f"repeated keydowns flagged repeat=1 and counted separately ({len(rep)})")
    check(all(a["event_type"] == "keydown" for a in rep) and all(a["repeat"] == "0" for a in rows if a not in rep), "all other events have repeat=0")

    # 4. incorrect characters + backspaces preserved; replay CSV == metadata typed text --
    typed = []; wrong = 0; bs = 0
    for a in rows:
        if a["event_type"] != "keydown": continue
        if a["action"] == "char":
            typed.append(a["typed_char"]); wrong += a["correct"] == "0"
            check_ok = a["typed_char"] == a["key"] and a["correct"] == ("1" if a["typed_char"] == a["target_char"] else "0")
            if not check_ok: fails.append("row labelled wrongly: " + str(a))
        elif a["action"] == "backspace": typed.pop(); bs += 1
    check("".join(typed) == md["typed_text"], "replaying keystrokes.csv reproduces metadata typed_text exactly")
    check(wrong > 0 and bs > 0 and wrong == md["stats"]["incorrect_keystrokes"] and bs == md["stats"]["backspace_keystrokes"], f"wrong characters ({wrong}) and backspaces ({bs}) preserved, counts match metadata")
    check(md["target_text"] == r["target"], "metadata target_text == displayed paragraph")
    corrected = sum(md["char_frequency"]["character_keystrokes_including_corrected"].values())
    check(corrected == md["stats"]["char_keystrokes"] == len(typed) + bs, f"char_keystroke frequency includes corrected characters ({corrected})")
    check(all(a["action"] == "" for a in rows if a["event_type"] == "keyup"), "keyup rows carry no action")

    # 5. timestamps / sync ---------------------------------------------------------------
    t = [float(a["time_ms"]) for a in rows]
    check(all(y >= x for x, y in zip(t, t[1:])), "time_ms (performance.now) is monotonic across events")
    ev_t = [float(a["event_time_ms"]) for a in rows]
    check(max(abs(a - b) for a, b in zip(t, ev_t)) < 50, f"performance.now() vs event.timeStamp agree (max diff {max(abs(a-b) for a,b in zip(t,ev_t)):.2f} ms)")
    start = md["session"]["audio_start_perf_ms_estimate"]
    rel = [float(a["t_rel_ms"]) for a in rows]
    check(max(abs((x - start) - y) for x, y in zip(t, rel)) < 0.01, "t_rel_ms == time_ms - estimated audio start")
    smp = [int(a["sample_index_est"]) for a in rows]
    sy0 = md["sync"]["models"][md["sync"]["primary_mapping"]]
    tol = abs(sy0["effective_sample_rate_hz"] / sr - 1) * max(rel) / 1000 * sr + 2
    check(max(abs(s - y * sr / 1000) for s, y in zip(smp, rel)) <= tol, f"sample_index_est == t_rel_ms * sample_rate within the fitted clock-rate error (tol {tol:.1f} samples)")
    check(all(y >= x for x, y in zip(smp, smp[1:])), "sample_index_est is non-decreasing")
    check(min(smp) > 0 and max(smp) <= n + sr * 0.5, f"all events fall inside the recording (samples {min(smp)}..{max(smp)} of {n})")
    sy = md["sync"]; M = sy["models"][sy["primary_mapping"]]
    check(M["n"] > 10, f"{M['n']} sync points in primary mapping '{sy['primary_mapping']}'")
    check(abs(M["effective_sample_rate_hz"] / sr - 1) < 0.005, f"fitted clock rate {M['effective_sample_rate_hz']:.1f} Hz vs nominal {sr}")
    print(f"        residual RMS {M['residual_rms_samples']/sr*1000:.3f} ms (max {M['residual_max_samples']/sr*1000:.2f} ms); slope source: {M['slope_source']}")
    print("        offset of other clock mappings vs primary (ms):", sy["model_offset_vs_primary_ms"])
    last_key = max(t) - start
    check(md["audio"]["duration_s"] * 1000 >= last_key - 50, f"audio ({md['audio']['duration_s']*1000:.0f} ms) covers last key event at {last_key:.0f} ms")
    check(md["audio"]["missing_input_frames_zero_filled"] == 0 and md["audio"]["clean_stop"], "no dropped/zero-filled input frames, clean worklet stop")
    first_down = next(float(a["t_rel_ms"]) for a in rows if a["event_type"] == "keydown")
    check(first_down > 1000, f"leading audio exists before first key (first keydown at {first_down:.0f} ms)")

    # 6. summary / export artefacts ------------------------------------------------------
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(r["zip"])))
    check(z.testzip() is None and sorted(z.namelist()) == [f"{folder}/audio.wav", f"{folder}/keystrokes.csv", f"{folder}/metadata.json"], "zip contains session_TIMESTAMP/{audio.wav,keystrokes.csv,metadata.json}")
    check(z.read(f"{folder}/audio.wav") == wavb and z.read(f"{folder}/keystrokes.csv").decode() == r["csv"], "zip contents identical to individual files")
    check(re.fullmatch(r"session_\d{8}_\d{6}", folder) is not None, f"folder name {folder}")
    check(r["summaryVisible"] and [l["download"] for l in r["links"]] == [f"{folder}_audio.wav", f"{folder}_keystrokes.csv", f"{folder}_metadata.json", f"{folder}.zip"], "summary shows 3 file downloads + zip, uniquely named")
    check(all(l["href"] == "blob:" for l in r["links"]), "download links are local blob: URLs")
    for key in ("target_text", "typed_text", "audio", "capture", "char_frequency", "stats", "session", "sync"):
        check(key in md, f"metadata has '{key}'")
    check(all(k in md["session"] for k in ("started_iso", "finished_iso")) and "wpm" in md["stats"] and "accuracy_keystrokes_pct" in md["stats"], "metadata has timestamps, WPM and accuracy")
    print("        summary:", r["summaryText"].replace("\n", " | ")[:300])
    return md

ff = Firefox(extra_prefs={"media.autoplay.default": 0, "media.autoplay.blocking_policy": 0})  # headless has no real user gesture
out = tempfile.mkdtemp(prefix="sneakey-out-")
try:
    ff.get(URL)
    reqs = ff.run("const cb=arguments[arguments.length-1]; cb(performance.getEntriesByType('resource').map(e=>e.name))")
    check(all(u.startswith("http://localhost:8000/") for u in reqs), f"page only loads local resources: {[u.rsplit('/',1)[-1] for u in reqs]}")
    r1 = ff.run(JS_TYPER, "full", 10**6)
    m1 = validate("session 1: complete paragraph", r1, out)
    check(m1 and m1["session"]["completed_paragraph"] and m1["stats"]["final_total_chars"] == m1["target_chars"], "session 1 ended automatically at the end of the paragraph")

    # second session: new paragraph, stop early with "Finish now"
    ff.run("const cb=arguments[arguments.length-1]; document.getElementById('newpara').click(); setTimeout(()=>cb(1),200);")
    r2 = ff.run(JS_TYPER, "partial", 160)
    m2 = validate("session 2: partial, finished manually", r2, out)
    if m1 and m2:
        print("\n=== independence of sessions ===")
        check(m1["session_id"] != m2["session_id"] and m1["folder"] != m2["folder"], f"distinct session ids/folders ({m1['folder']} vs {m2['folder']})")
        check(m1["target_text"] != m2["target_text"], "different paragraphs")
        check(not m2["session"]["completed_paragraph"] and m2["stats"]["final_total_chars"] < m2["target_chars"], "session 2 partial export reflects only what was typed")
        check(m1["events"]["total"] > m2["events"]["total"] and m2["audio"]["frames"] < m1["audio"]["frames"], "event counts and audio lengths are per-session")
        check(os.path.getsize(os.path.join(out, m2["folder"], "audio.wav")) < os.path.getsize(os.path.join(out, m1["folder"], "audio.wav")), "session 2's files are separate from session 1's")
        c2 = r2["coverage"]; c1 = r1["coverage"]
        exp = {k: c1.get(k, 0) + m2["char_frequency"]["character_keystrokes_including_corrected"].get(k, 0) for k in c2}
        check(c2 == exp and sum(c1.values()) == m1["stats"]["char_keystrokes"], "cumulative coverage = sum of both sessions' character keystrokes")
        print(f"\n(exported files kept in {out})")
finally:
    ff.close()
print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURE(S)")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
