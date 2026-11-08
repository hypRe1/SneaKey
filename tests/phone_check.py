"""
End-to-end check of phone recording with two headless Firefox instances against `run.py --phone` and an empty
temporary data folder:

  HTTPS   the phone port serves the phone page, its script and endpoints over TLS, and refuses everything else.
  Phone   opens the phone page and starts its (fake) microphone. The browser loads it from http://127.0.0.1 (a
          secure context for the microphone, like the real HTTPS link) because headless Firefox in this harness
          won't accept the self-signed certificate; the page, script and endpoints are identical.
  Laptop  selects "A phone" on the Record page, pairs, types the generated text with synthetic key events, saves.
  Checks  the session's audio came from the phone, is 48 kHz and as long as the typing, every key event maps to a
          position inside it, the timing details are stored, and the phone port serves nothing but the phone page.

Firefox's fake microphone plays a tone, not keyboard sounds, so this tests the recording path, not accuracy.

    python tests/phone_check.py
"""
import csv
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from marionette import Firefox  # noqa: E402
from browser_check import TRACK_ERRORS, TYPE_TEXT, WAIT, Check, free_port  # noqa: E402

AUDIO_PREFS = {"media.autoplay.default": 0, "media.autoplay.blocking_policy": 0}


def main():
    check = Check()
    data = Path(tempfile.mkdtemp(prefix="sneakey-phone-e2e-"))
    port, phone_port = free_port(), free_port()
    server = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--port", str(port), "--phone", "--phone-port", str(phone_port)],
                              cwd=ROOT, env=dict(os.environ, SNEAKEY_DATA=str(data)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}/"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "api/health", timeout=2)
            break
        except OSError:
            time.sleep(0.3)
    insecure = ssl.create_default_context()
    insecure.check_hostname, insecure.verify_mode = False, ssl.CERT_NONE
    phone_base = f"https://127.0.0.1:{phone_port}"

    def phone_port_status(path):
        try:
            return urllib.request.urlopen(phone_base + path, context=insecure, timeout=5).status
        except urllib.error.HTTPError as error:
            return error.code

    laptop = Firefox(port=free_port(), extra_prefs=AUDIO_PREFS)
    phone = Firefox(port=free_port(), extra_prefs=AUDIO_PREFS)
    try:
        wait = lambda browser, js, seconds=60: browser.run(WAIT, js, seconds * 1000)

        # The phone port only serves the phone page and its endpoints.
        check(phone_port_status("/") == 404 and phone_port_status("/api/sessions") == 404 and phone_port_status("/api/phone/status") == 404,
              "phone port refuses the app, the session API and the laptop-side phone API")
        check(phone_port_status("/api/phone/d/WRONGCODE/poll") == 409, "phone endpoints need the pairing code")

        # Laptop: choose the phone and pair.
        laptop.get(base + "#record")
        laptop.run(TRACK_ERRORS)
        wait(laptop, "return document.querySelectorAll('#text .c').length > 50")
        laptop.run("""const s = document.getElementById('micSource'); s.value = 'phone'; s.dispatchEvent(new Event('change'));
                      document.getElementById('pairPhone').click(); arguments[0](1)""")
        links = wait(laptop, "const l = [...document.querySelectorAll('#phoneLinks .phone-link')].map(e => e.textContent); return l.length && l")
        check(isinstance(links, list) and links[0].startswith("https://") and "/p/" in links[0], f"pairing link shown: {links}")
        code = links[0].rsplit("/", 1)[1]

        check(phone_port_status(f"/p/{code}") == 200 and phone_port_status("/js/phone.js") == 200
              and phone_port_status(f"/api/phone/d/{code}/clock") == 200, "phone page, script and endpoints served over HTTPS")

        # Phone: open the page and start the microphone.
        phone.get(f"{base}p/{code}")
        phone.run(TRACK_ERRORS)
        wait(phone, "return document.getElementById('enable')")
        phone.run("document.getElementById('enable').click(); arguments[0](1)")
        check(wait(phone, "return document.getElementById('state').textContent.includes('waiting')", 20) is True, "phone microphone on, waiting")
        check(wait(laptop, "return document.getElementById('phoneStatus').textContent.includes('connected')", 20) is True, "laptop sees the phone connected")

        # Laptop: record a short test with synthetic typing.
        laptop.run("""for (const [id, v] of [['participant','P-phone'],['keyboard','test kb'],['microphone','phone'],['environment','headless']]) document.getElementById(id).value = v;
            document.getElementById('consent').checked = true;
            document.getElementById('length').value = '180'; document.getElementById('length').dispatchEvent(new Event('change'));
            setTimeout(() => { document.getElementById('start').click(); arguments[0](1); }, 800);""")
        check(wait(laptop, "return document.getElementById('recState').textContent.startsWith('Recording on the phone')", 20) is True, "recording on the phone")
        check(wait(phone, "return document.getElementById('state').className.includes('recording')", 10) is True, "phone shows it is recording")
        typed = laptop.run(TYPE_TEXT)
        check(wait(laptop, "return !document.getElementById('saved').classList.contains('hidden')", 60) is True, f"session saved after typing {typed} characters")
        check(wait(phone, "return document.getElementById('state').textContent.startsWith('Done')", 20) is True, "phone finished uploading")

        folders = list((data / "recordings").glob("session_*"))
        check(len(folders) == 1, "one session on disk")
        folder = folders[0]
        meta = json.loads((folder / "metadata.json").read_text())
        rows = list(csv.DictReader(open(folder / "keystrokes.csv")))
        with wave.open(str(folder / "audio.wav")) as w:
            frames, rate = w.getnframes(), w.getframerate()
        downs = [int(r["sample_index_est"]) for r in rows if r["event_type"] == "keydown"]
        check(meta["capture"]["device"] == "phone" and rate == 48000, "audio from the phone, stored at 48 kHz")
        check(0 < downs[0] and downs[-1] < frames and downs == sorted(downs), "every key event maps inside the phone audio, in order")
        typing_s = (downs[-1] - downs[0]) / 48000
        check(abs(frames / 48000 - typing_s) < 6, f"audio length {frames / 48000:.1f} s matches the typing ({typing_s:.1f} s)")
        sync = meta["sync"]
        check(sync["laptop_clock"]["pings"] > 3 and sync["phone_clock"]["pings"] > 3 and sync["audio_gaps_ms"] < 100,
              f"timing measured: laptop {sync['laptop_clock']['min_rtt_ms']} ms, phone {sync['phone_clock']['min_rtt_ms']} ms round trip, gaps {sync['audio_gaps_ms']} ms")
        listed = json.loads(urllib.request.urlopen(base + "api/sessions").read())
        check(listed[0]["microphone"] == "phone", "session listed as a phone recording")
        for name, browser in (("laptop", laptop), ("phone", phone)):
            errors = browser.run("arguments[0](window.__errors)")
            check(not errors, f"no script errors on the {name} {errors or ''}")
    finally:
        laptop.close()
        phone.close()
        server.terminate()
        server.wait(10)
        shutil.rmtree(data, ignore_errors=True)
    print(f"\n{Check.failures} failure(s)")
    sys.exit(1 if Check.failures else 0)


if __name__ == "__main__":
    main()
