"""
End-to-end check in headless Firefox against a real server and an empty temporary data folder:

  Record   fills in the participant form, records with Firefox's fake microphone (a tone, not keyboard audio),
           types the generated text with synthetic key events and saves. Checks the files on disk are in the
           recording format the training scripts read, with the generator metadata.
  Analyse  opens the new session and checks the timeline and inspector render without errors.
  Predict  runs the pipeline on it (no real key clicks, so few or no presses) and checks the results render.

    python tests/browser_check.py
"""
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from marionette import Firefox  # noqa: E402

WAIT = """
const [check, timeoutMs] = [arguments[0], arguments[1]];
const done = arguments[arguments.length - 1];
const started = Date.now();
(function poll() {
  let value; try { value = new Function(check)(); } catch (e) { value = null; }
  if (value) return done(value);
  if (Date.now() - started > timeoutMs) return done({ timedOut: true, error: document.getElementById("error")?.textContent });
  setTimeout(poll, 100);
})();
"""
TRACK_ERRORS = """window.__errors = [];
window.addEventListener('error', (e) => __errors.push(String(e.message)));
window.addEventListener('unhandledrejection', (e) => __errors.push(String(e.reason)));
arguments[arguments.length - 1](true);"""

TYPE_TEXT = """
const done = arguments[arguments.length - 1];
const area = document.getElementById('typing');
const target = [...document.querySelectorAll('#text .c')].map((c) => c.textContent).join('');
const codeFor = (ch) => ch === ' ' ? 'Space' : ch === ',' ? 'Comma' : ch === '.' ? 'Period' : ch === '?' ? 'Slash'
  : /[a-z]/i.test(ch) ? 'Key' + ch.toUpperCase() : 'Unidentified';
let i = 0;
(function next() {
  if (i >= target.length) return done(target.length);
  const ch = target[i++];
  const shift = ch !== ch.toLowerCase() || ch === '?';
  const opts = { key: ch, code: codeFor(ch), shiftKey: shift, bubbles: true };
  if (i === 5) {   // one typo and its correction, so the log has a wrong character and a backspace
    area.dispatchEvent(new KeyboardEvent('keydown', { key: 'q', code: 'KeyQ', bubbles: true }));
    area.dispatchEvent(new KeyboardEvent('keyup', { key: 'q', code: 'KeyQ', bubbles: true }));
    area.dispatchEvent(new KeyboardEvent('keydown', { key: 'Backspace', code: 'Backspace', bubbles: true }));
    area.dispatchEvent(new KeyboardEvent('keyup', { key: 'Backspace', code: 'Backspace', bubbles: true }));
  }
  area.dispatchEvent(new KeyboardEvent('keydown', opts));
  setTimeout(() => { area.dispatchEvent(new KeyboardEvent('keyup', opts)); setTimeout(next, 25); }, 30);
})();
"""


class Check:
    failures = 0

    def __call__(self, condition, message):
        print(("PASS  " if condition else "FAIL  ") + message, flush=True)
        Check.failures += not condition


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    check = Check()
    data = Path(tempfile.mkdtemp(prefix="sneakey-e2e-"))
    port = free_port()
    server = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--port", str(port)], cwd=ROOT,
                              env=dict(os.environ, SNEAKEY_DATA=str(data)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}/"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "api/health", timeout=2)
            break
        except OSError:
            time.sleep(0.3)
    # Synthetic clicks aren't user gestures, so allow audio to start without one (real use needs no change).
    browser = Firefox(port=free_port(), extra_prefs={"media.autoplay.default": 0, "media.autoplay.blocking_policy": 0})
    try:
        wait = lambda js, seconds=60: browser.run(WAIT, js, seconds * 1000)

        # ---- record ----
        browser.get(base + "#record")
        browser.run(TRACK_ERRORS)
        check(wait("return document.querySelectorAll('#text .c').length > 50") is True, "a test text is generated")
        browser.run("""const done = arguments[arguments.length - 1];
            document.getElementById('start').click();
            setTimeout(() => done(document.getElementById('error').textContent), 300);""")
        check("consent" in browser.run("arguments[0](document.getElementById('error').textContent)"), "recording refuses to start without consent")
        browser.run("""const done = arguments[arguments.length - 1];
            for (const [id, v] of [['participant','P-test'],['keyboard','test kb'],['microphone','fake'],['environment','headless']]) document.getElementById(id).value = v;
            document.getElementById('consent').checked = true;
            document.getElementById('length').value = '180';
            document.getElementById('length').dispatchEvent(new Event('change'));
            setTimeout(() => { document.getElementById('start').click(); done(1); }, 800);""")
        check(wait("return document.getElementById('recState').classList.contains('on')", 20) is True, "microphone starts")
        time.sleep(0.6)
        typed = browser.run(TYPE_TEXT)
        check(wait("return !document.getElementById('saved').classList.contains('hidden')", 30) is True, f"session saves after typing {typed} characters")
        sessions = list((data / "recordings").glob("session_*"))
        check(len(sessions) == 1, "one session folder on disk")
        folder = sessions[0]
        with wave.open(str(folder / "audio.wav")) as w:
            check((w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 48000), "audio is mono 16-bit 48 kHz")
            seconds = w.getnframes() / 48000
        rows = list(csv.DictReader(open(folder / "keystrokes.csv")))
        meta = json.loads((folder / "metadata.json").read_text())
        check(rows and list(rows[0])[:9] == ["event_index", "event_type", "key", "code", "location", "time_ms", "event_time_ms", "t_rel_ms", "sample_index_est"],
              "keystrokes.csv has the original column layout")
        downs = [r for r in rows if r["event_type"] == "keydown"]
        check(any(r["action"] == "backspace" for r in downs) and any(r["correct"] == "0" for r in downs), "typo and backspace are logged")
        positions = [int(r["sample_index_est"]) for r in downs]
        check(positions == sorted(positions) and 0 < positions[0] and positions[-1] < seconds * 48000, "event positions lie inside the audio, in order")
        check(meta["typed_text"] == meta["target_text"], "typed text equals the target")
        check(meta["participant"]["id"] == "P-test" and meta["conditions"]["keyboard"] == "test kb", "participant and setup are saved")
        check(meta["generator"]["seed"] is not None and len(meta["generator"]["sentences"]) >= 2, "generator seed and sentences are saved")
        errors = browser.run("arguments[0](window.__errors)")
        check(not errors, f"no script errors on Record {errors or ''}")

        # ---- analyse ----
        session_id = folder.name
        browser.get(base + f"#analyse/{session_id}")
        browser.run(TRACK_ERRORS)
        check(isinstance(wait("return document.querySelector('#analyseTimeline canvas.main')"), dict), "analyse timeline renders")
        time.sleep(1)
        browser.run("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight'})); arguments[0](1)")
        check(isinstance(wait("return document.querySelector('#keystroke .big-key')"), dict), "keystroke inspector shows the first key")
        errors = browser.run("arguments[0](window.__errors)")
        check(not errors, f"no script errors on Analyse {errors or ''}")

        # ---- predict ----
        browser.get(base + f"#predict/{session_id}")
        browser.run(TRACK_ERRORS)
        wait("return document.getElementById('runPredict')")
        browser.run("document.getElementById('runPredict').click(); arguments[0](1)")
        check(wait("return !document.getElementById('predictResults').classList.contains('hidden') && document.getElementById('metricCards').children.length", 120) == 5,
              "prediction runs and shows its metrics")
        result = json.loads(urllib.request.urlopen(urllib.request.Request(base + f"api/sessions/{session_id}/predict", method="POST")).read())
        check(result["role"] == "unseen" and result["evaluation"]["detection"]["typed_presses"] > 0, "new session is unseen and evaluated against its typed presses")
        errors = browser.run("arguments[0](window.__errors)")
        check(not errors, f"no script errors on Predict {errors or ''}")
    finally:
        browser.close()
        server.terminate()
        server.wait(10)
        shutil.rmtree(data, ignore_errors=True)
    print(f"\n{Check.failures} failure(s)")
    sys.exit(1 if Check.failures else 0)


if __name__ == "__main__":
    main()
