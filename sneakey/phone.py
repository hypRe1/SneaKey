"""Recording with a phone's microphone while the keyboard is logged on the laptop.

Used for demos where the audio must come from a separate device (proving the prediction is audio-only), and for
training phone models. The phone opens https://<laptop LAN address>:8443/p/<code> (see run.py --phone), keeps its
microphone open, and polls for commands; the laptop's Record page starts and stops the recording.

Timing. The keyboard events are timed on the laptop's page clock, the audio on the phone's audio clock. Both are
tied to the server's clock:
  * each device repeatedly asks the server for its time (round-trip "pings"); the ping with the shortest round
    trip in each 5 s window gives that device's clock offset to the server, to within half its round-trip time,
    and a straight line through those offsets also absorbs any rate difference between the clocks;
  * the phone page records pairs (page time, audio frame) every 250 ms, giving audio frame <-> phone time.
Chaining these maps each key event to an estimated sample in the phone audio (`sample_index_est`, as the laptop
recorder writes). What remains unknown is the phone's microphone delay and any audio-clock drift; the ground-truth
labelling measures both from the audio (groundtruth.py, wide mode for phone sessions).

All state is in memory (one active pairing), so phone recordings only work while the server keeps running.
"""
import secrets
import threading
import time

import numpy as np
from scipy import signal

from .config import SAMPLE_RATE

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"     # no 0/O/1/I/L: easy to type on a phone
CONNECTED_WITHIN_MS = 3000
MAX_RECORDING_S = 15 * 60
PING_WINDOW_MS = 5000


def server_ms():
    """The server's clock, shared by both devices' timing: milliseconds, monotonic."""
    return time.perf_counter_ns() / 1e6


class NotPaired(Exception):
    pass


class Recording:
    def __init__(self):
        self.id = secrets.token_hex(4)
        self.chunks = {}              # start_frame -> int16 samples
        self.state = "starting"       # starting -> recording (first audio) -> stopping -> done
        self.sync = None              # the phone's timing data, sent when it finishes
        self.started_ms = server_ms()


class Pairing:
    def __init__(self):
        self.code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        self.created_ms = server_ms()
        self.phone_seen_ms = None
        self.phone = {}               # user agent, sample rate, microphone settings
        self.level = 0.0              # recent microphone peak, for positioning the phone
        self.recording = None
        self.lock = threading.Lock()

    def connected(self):
        return self.phone_seen_ms is not None and server_ms() - self.phone_seen_ms < CONNECTED_WITHIN_MS


_pairing = None
_lock = threading.Lock()


def new_pairing():
    global _pairing
    with _lock:
        _pairing = Pairing()
    return _pairing


def current():
    return _pairing


def _pairing_for(code):
    pairing = _pairing
    if pairing is None or not secrets.compare_digest(code.upper(), pairing.code):
        raise NotPaired("Unknown or expired pairing code. Open the link shown on the laptop again.")
    return pairing


# ---- phone side ----

def phone_hello(code, info):
    pairing = _pairing_for(code)
    pairing.phone = {k: info.get(k) for k in ("user_agent", "sample_rate", "track_settings")}
    pairing.phone_seen_ms = server_ms()
    return {"ok": True}


def phone_poll(code, level=None):
    """The phone asks what to do; also how it reports it is alive and its microphone level."""
    pairing = _pairing_for(code)
    pairing.phone_seen_ms = server_ms()
    if level is not None:
        pairing.level = float(level)
    rec = pairing.recording
    if rec is None or rec.state == "done":
        return {"command": "wait"}
    return {"command": "stop" if rec.state == "stopping" else "record", "recording_id": rec.id}


def phone_audio(code, recording_id, start_frame, pcm):
    pairing = _pairing_for(code)
    rec = pairing.recording
    if rec is None or rec.id != recording_id or rec.state == "done":
        raise ValueError("No such recording in progress.")
    samples = np.frombuffer(pcm, dtype="<i2")
    with pairing.lock:
        rec.chunks[int(start_frame)] = samples.copy()
        if rec.state == "starting":
            rec.state = "recording"
        total = sum(len(c) for c in rec.chunks.values())
    rate = pairing.phone.get("sample_rate") or SAMPLE_RATE
    if total > MAX_RECORDING_S * rate:
        raise ValueError("Recording too long.")
    pairing.phone_seen_ms = server_ms()
    return {"ok": True}


def phone_finish(code, recording_id, sync):
    """The phone has sent all its audio. sync: {sample_rate, clock_pairs: [{perf, frame}], pings: [{t0, t1, server}]}"""
    pairing = _pairing_for(code)
    rec = pairing.recording
    if rec is None or rec.id != recording_id:
        raise ValueError("No such recording.")
    rec.sync = sync
    rec.state = "done"
    return {"ok": True}


# ---- laptop side ----

def start():
    pairing = _pairing
    if pairing is None or not pairing.connected():
        raise NotPaired("The phone isn't connected. Open the pairing link on the phone first.")
    pairing.recording = Recording()
    return {"recording_id": pairing.recording.id}


def stop():
    pairing = _pairing
    if pairing is None or pairing.recording is None:
        raise NotPaired("No phone recording in progress.")
    if pairing.recording.state != "done":
        pairing.recording.state = "stopping"
    return {"ok": True}


def status():
    pairing = _pairing
    if pairing is None:
        return {"paired": False}
    rec = pairing.recording
    rate = pairing.phone.get("sample_rate") or SAMPLE_RATE
    return {
        "paired": True, "code": pairing.code, "connected": pairing.connected(),
        "phone": pairing.phone, "level": round(pairing.level, 4),
        "recording": None if rec is None else {
            "id": rec.id, "state": rec.state,
            "seconds": round(sum(len(c) for c in rec.chunks.values()) / rate, 1),
        },
    }


def take_finished(recording_id):
    """The finished recording, for turning into a session. Raises if the phone hasn't finished uploading."""
    pairing = _pairing
    rec = pairing.recording if pairing else None
    if rec is None or rec.id != recording_id:
        raise NotPaired("No such phone recording.")
    if rec.state != "done":
        raise ValueError("The phone is still sending its audio.")
    return rec, dict(pairing.phone)


# ---- timing ----

def clock_offset(pings):
    """Offset from a device's page clock to the server clock: server = t + offset(t).
    pings: [{t0, t1, server}] (device times before/after, server time). Uses the fastest round trip in each
    5 s window and a straight line through those, so a rate difference between the clocks is absorbed.
    Returns (offset function, details)."""
    rows = sorted(((p["t0"] + p["t1"]) / 2, p["server"] - (p["t0"] + p["t1"]) / 2, p["t1"] - p["t0"])
                  for p in pings if p["t1"] >= p["t0"])
    if not rows:
        raise ValueError("No clock measurements.")
    best = {}
    for mid, offset, rtt in rows:
        window = int(mid // PING_WINDOW_MS)
        if window not in best or rtt < best[window][2]:
            best[window] = (mid, offset, rtt)
    points = np.array(sorted(best.values()))
    mids, offsets, rtts = points[:, 0], points[:, 1], points[:, 2]
    if len(points) >= 3 and mids[-1] - mids[0] > 10000:
        slope, intercept = np.polyfit(mids, offsets, 1)
        if abs(slope) > 0.001:                  # over 1000 ppm: not a clock-rate difference, ignore it
            slope, intercept = 0.0, float(np.median(offsets))
    else:
        slope, intercept = 0.0, float(offsets[np.argmin(rtts)])
    residual = offsets - (intercept + slope * mids)
    details = {"pings": len(rows), "windows": len(points), "min_rtt_ms": round(float(rtts.min()), 2),
               "median_rtt_ms": round(float(np.median(rtts)), 2), "rate_ppm": round(float(slope) * 1e6, 1),
               "residual_ms": round(float(np.abs(residual).max()), 2)}
    return (lambda t: intercept + slope * np.asarray(t, dtype=float)), details


def frame_clock(clock_pairs, sample_rate):
    """Audio frame as a function of the phone's page time, from (perf, frame) pairs. The fitted rate is kept only
    if plausible (within 0.5% of nominal); otherwise the nominal rate is used."""
    perf = np.array([p["perf"] for p in clock_pairs], dtype=float)
    frame = np.array([p["frame"] for p in clock_pairs], dtype=float)
    nominal = sample_rate / 1000
    slope = nominal
    if len(perf) >= 3 and perf.max() - perf.min() > 5000:
        fitted = np.polyfit(perf, frame, 1)[0]
        if abs(fitted / nominal - 1) < 0.005:
            slope = fitted
    intercept = float(np.mean(frame - slope * perf))
    return (lambda t: intercept + slope * np.asarray(t, dtype=float)), {"frames_per_ms": slope, "pairs": len(perf)}


def assemble(rec, laptop_pings, event_times):
    """Phone audio as one 48 kHz int16 array, and each laptop event time (page ms) as an estimated sample index.
    Returns (audio, samples, sync details)."""
    sync = rec.sync or {}
    rate = int(sync.get("sample_rate") or SAMPLE_RATE)
    if not rec.chunks:
        raise ValueError("The phone sent no audio.")
    first = min(rec.chunks)
    length = max(start + len(c) for start, c in rec.chunks.items()) - first
    audio = np.zeros(length, dtype=np.int16)
    covered = 0
    for start, chunk in rec.chunks.items():
        audio[start - first:start - first + len(chunk)] = chunk
        covered += len(chunk)

    laptop_offset, laptop_details = clock_offset(laptop_pings)
    phone_offset, phone_details = clock_offset(sync.get("pings", []))
    frame_at, frame_details = frame_clock(sync.get("clock_pairs", []), rate)
    server = np.asarray(event_times, dtype=float) + laptop_offset(event_times)
    # Invert server = t + offset(t) for the phone (offset is linear: intercept + slope * t).
    o0, o1 = float(phone_offset(0.0)), float(phone_offset(1.0)) - float(phone_offset(0.0))
    phone_perf = (server - o0) / (1 + o1)
    samples = frame_at(phone_perf) - first

    if rate != SAMPLE_RATE:                 # the models need 48 kHz: resample the audio and rescale positions
        from math import gcd
        g = gcd(SAMPLE_RATE, rate)
        audio = np.clip(np.round(signal.resample_poly(audio.astype(float), SAMPLE_RATE // g, rate // g)), -32768, 32767).astype(np.int16)
        samples = samples * SAMPLE_RATE / rate
    details = {
        "method": "phone audio + laptop key log, both mapped to the server clock by round-trip pings; "
                  "microphone delay and drift are measured from the audio when labelling",
        "phone_sample_rate": rate, "audio_gaps_ms": round((length - covered) / rate * 1000, 1),
        "laptop_clock": laptop_details, "phone_clock": phone_details, "phone_audio_clock": frame_details,
        "uncertainty_ms": round((laptop_details["min_rtt_ms"] + phone_details["min_rtt_ms"]) / 2, 1),
    }
    return audio, np.rint(samples).astype(int), details
