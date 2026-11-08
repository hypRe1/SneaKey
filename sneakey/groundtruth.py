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
"""
import numpy as np
from scipy import ndimage

from .config import SAMPLES_PER_MS
from .detection import click_envelope


def onset_strength(envelope):
    novelty = np.maximum(envelope - ndimage.uniform_filter1d(envelope, 15), 0)
    return np.minimum(novelty, np.percentile(novelty, 99))


def _best_lag(novelty, times_ms, lags):
    scores = [novelty[np.clip(np.rint(times_ms + lag).astype(int), 0, len(novelty) - 1)].mean() for lag in lags]
    return int(lags[int(np.argmax(scores))])


def label_events(samples, events):
    """Acoustic positions for logged events.

    events: dicts with "type" ("keydown"/"keyup") and "sample" (the logged sample_index_est).
    Returns {"press_lag_ms", "release_lag_ms", "centers": sample index of each event's sound (None for other
    event types)}."""
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


def true_presses(session, samples):
    """The session's key presses as heard: [{"event", "code", "key", "sample", "repeat"}] in time order.
    `event` indexes session["events"]."""
    labels = label_events(samples, session["events"])
    presses = [{"event": i, "code": e["code"], "key": e["key"], "sample": labels["centers"][i], "repeat": e["repeat"]}
               for i, e in enumerate(session["events"]) if e["type"] == "keydown" and labels["centers"][i] is not None]
    presses.sort(key=lambda p: p["sample"])
    return presses, labels
