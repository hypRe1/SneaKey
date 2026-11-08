"""Ground truth: where each logged key event is heard in the audio.

The recorder logs each key event's estimated audio sample (`sample_index_est`), which cannot include the
microphone's input delay (30-60 ms, different every time the microphone opens). v6's labelling (frontend.load,
ported here unchanged) measures that delay and snaps each event onto its sound:

  1. Click-onset strength: the 1.2 kHz high-passed energy per millisecond minus its 15 ms running mean, clipped at 0
     and at its 99th percentile.
  2. The press lag is the shift (0-100 ms) at which the logged keydowns line up with the most onset strength.
  3. Each keydown is moved by that lag and then to the loudest millisecond within +/-10 ms.
  4. Key releases get their own lag, searched from 5 ms before to 30 ms after the press lag.

These positions are the labels the models were trained and evaluated with.

Phone recordings (wide=True). The audio comes from a different device than the keyboard log, so the event
positions come from a clock mapping across the network (sneakey/phone.py) and the unknown delay is larger and
less certain (phone microphone + Bluetooth/USB buffers: often 100-300 ms), and the two clocks drift apart:
  * the press lag is searched over -400 .. +1000 ms instead of 0 .. 100 ms;
  * drift: the lag is re-measured in 15 s windows (+/-60 ms around the global lag) and a robust straight line
    (Theil-Sen) through those gives each event its own lag, so a phone whose audio clock runs a few hundred ppm
    fast or slow is still labelled to the millisecond;
  * then steps 3-4 as above, with each event's own lag.
Laptop recordings use the original method unchanged.
"""
import numpy as np
from scipy import ndimage

from .config import SAMPLES_PER_MS
from .detection import click_envelope


def onset_strength(envelope):
    novelty = np.maximum(envelope - ndimage.uniform_filter1d(envelope, 15), 0)
    return np.minimum(novelty, np.percentile(novelty, 99))


PHONE_LAG_RANGE_MS = (-400, 1000)
DRIFT_WINDOW_MS = 15000
DRIFT_SEARCH_MS = 60
MAX_DRIFT_PPM = 2000


def _best_lag(novelty, times_ms, lags):
    scores = [novelty[np.clip(np.rint(times_ms + lag).astype(int), 0, len(novelty) - 1)].mean() for lag in lags]
    return int(lags[int(np.argmax(scores))])


def _lag_scores(novelty, times_ms, lags):
    index = np.clip(np.rint(times_ms[None, :] + lags[:, None]).astype(int), 0, len(novelty) - 1)
    return novelty[index].mean(axis=1)


def _drift_model(novelty, down_ms, global_lag):
    """Per-time lag from windowed lag measurements: returns (lag_at(times_ms), drift_ppm, windows)."""
    windows = []
    start = down_ms.min()
    while start < down_ms.max():
        inside = down_ms[(down_ms >= start) & (down_ms < start + DRIFT_WINDOW_MS)]
        if len(inside) >= 25:
            lags = np.arange(global_lag - DRIFT_SEARCH_MS, global_lag + DRIFT_SEARCH_MS + 1)
            scores = _lag_scores(novelty, inside, lags)
            windows.append((float(inside.mean()), int(lags[scores.argmax()])))
        start += DRIFT_WINDOW_MS / 2
    if len(windows) < 3:
        return (lambda t: np.full(np.shape(t), float(global_lag))), 0.0, windows
    t, lag = np.array(windows).T
    slopes = [(lag[j] - lag[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)) if t[j] > t[i]]
    slope = float(np.median(slopes))
    if abs(slope) * 1e6 > MAX_DRIFT_PPM:          # implausible: keep a constant lag
        slope = 0.0
    intercept = float(np.median(lag - slope * t))
    return (lambda times: intercept + slope * np.asarray(times, dtype=float)), slope * 1e6, windows


def label_events(samples, events, wide=False):
    """Acoustic positions for logged events.

    events: dicts with "type" ("keydown"/"keyup") and "sample" (the logged sample_index_est).
    wide: phone recordings (see the module docstring).
    Returns {"press_lag_ms", "release_lag_ms", "centers": sample index of each event's sound (None for other
    event types)}."""
    if wide:
        return _label_events_wide(samples, events)
    envelope = click_envelope(samples)
    novelty = onset_strength(envelope)
    estimated_ms = np.array([e["sample"] for e in events], dtype=float) / SAMPLES_PER_MS
    is_down = np.array([e["type"] == "keydown" for e in events], dtype=bool)
    is_up = np.array([e["type"] == "keyup" for e in events], dtype=bool)
    result = {"press_lag_ms": None, "release_lag_ms": None, "centers": [None] * len(events)}
    if not is_down.any():
        return result

    press_lag = _best_lag(novelty, estimated_ms[is_down], np.arange(0, 101))
    result["press_lag_ms"] = press_lag
    lags = {"keydown": press_lag}
    if is_up.any():
        lags["keyup"] = _best_lag(novelty, estimated_ms[is_up], np.arange(press_lag - 5, press_lag + 31))
        result["release_lag_ms"] = lags["keyup"]

    for i, event in enumerate(events):
        if event["type"] not in lags:
            continue
        t = estimated_ms[i] + lags[event["type"]]
        a, b = max(0, int(t) - 10), min(len(envelope), int(t) + 11)
        if a < b:
            result["centers"][i] = int((a + envelope[a:b].argmax()) * SAMPLES_PER_MS)
    return result


def _label_events_wide(samples, events):
    envelope = click_envelope(samples)
    novelty = onset_strength(envelope)
    estimated_ms = np.array([e["sample"] for e in events], dtype=float) / SAMPLES_PER_MS
    is_down = np.array([e["type"] == "keydown" for e in events], dtype=bool)
    is_up = np.array([e["type"] == "keyup" for e in events], dtype=bool)
    result = {"press_lag_ms": None, "release_lag_ms": None, "centers": [None] * len(events), "drift_ppm": None}
    if not is_down.any():
        return result
    lags = np.arange(PHONE_LAG_RANGE_MS[0], PHONE_LAG_RANGE_MS[1] + 1)
    global_lag = int(lags[_lag_scores(novelty, estimated_ms[is_down], lags).argmax()])
    lag_at, drift_ppm, windows = _drift_model(novelty, estimated_ms[is_down], global_lag)
    result.update(press_lag_ms=global_lag, drift_ppm=round(drift_ppm, 1), lag_windows=windows)

    release_offset = 0
    if is_up.any():
        up_ms = estimated_ms[is_up]
        offsets = np.arange(-5, 31)
        scores = [novelty[np.clip(np.rint(up_ms + lag_at(up_ms) + o).astype(int), 0, len(novelty) - 1)].mean() for o in offsets]
        release_offset = int(offsets[int(np.argmax(scores))])
        result["release_lag_ms"] = global_lag + release_offset

    for i, event in enumerate(events):
        if event["type"] not in ("keydown", "keyup"):
            continue
        t = estimated_ms[i] + float(lag_at(estimated_ms[i])) + (release_offset if event["type"] == "keyup" else 0)
        a, b = max(0, int(t) - 10), min(len(envelope), int(t) + 11)
        if a < b:
            result["centers"][i] = int((a + envelope[a:b].argmax()) * SAMPLES_PER_MS)
    return result


def is_phone_session(session):
    return session.get("microphone") == "phone"


def true_presses(session, samples):
    """The session's key presses as heard: [{"event", "code", "key", "sample", "repeat"}] in time order.
    `event` indexes session["events"]."""
    labels = label_events(samples, session["events"], wide=is_phone_session(session))
    presses = [{"event": i, "code": e["code"], "key": e["key"], "sample": labels["centers"][i], "repeat": e["repeat"]}
               for i, e in enumerate(session["events"]) if e["type"] == "keydown" and labels["centers"][i] is not None]
    presses.sort(key=lambda p: p["sample"])
    return presses, labels
