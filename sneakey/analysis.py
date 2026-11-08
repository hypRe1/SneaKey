"""Signals for inspecting a recording: what each preprocessing stage does to the audio.

Nothing here changes the pipeline. It exposes its intermediate signals:
  * filtered audio exactly as each stage hears it (raw, 180 Hz detector input, 700 Hz classifier input,
    1.2 kHz labelling envelope),
  * the detector's novelty curve (spectral flux / its running median) with its threshold, the candidates and
    their press probabilities,
  * the ground-truth positions of each logged key event,
  * per-keystroke detail: the exact log-mel patch the CNN sees, its output, and the press classifier's features.
"""
import numpy as np

from .audio import highpass
from .config import SAMPLE_RATE, SAMPLES_PER_MS
from .detection import FEATURE_NAMES, event_features, spectral_flux_novelty
from .groundtruth import onset_strength, true_presses
from .detection import click_envelope

FILTERS = {
    "raw": None,
    "detector": 180,      # spectral-flux candidate detection
    "model": 700,         # press classifier and CNN key classifier input
    "click": 1200,        # click envelope used for peak refinement and ground-truth labelling
}


def filtered(samples, variant):
    if variant not in FILTERS:
        raise ValueError(f"Unknown filter {variant!r}; choose from {sorted(FILTERS)}.")
    return samples if FILTERS[variant] is None else highpass(samples, FILTERS[variant])


def session_signals(session, samples, pipeline):
    novelty = spectral_flux_novelty(samples)
    centers = pipeline.detector.candidates(samples, novelty)
    press_probability = pipeline.detector.press_probabilities(samples, centers)
    presses, labels = true_presses(session, samples)

    envelope = click_envelope(samples)
    envelope_db = 10 * np.log10(envelope + 1e-14)
    floor_db = float(np.median(envelope_db))
    events = []
    for i, event in enumerate(session["events"]):
        center = labels["centers"][i]
        level = None
        if center is not None:
            ms = center // SAMPLES_PER_MS
            level = round(float(envelope_db[max(0, ms - 2):ms + 3].max() - floor_db), 1)
        events.append({**{k: event[k] for k in ("type", "key", "code", "repeat", "action", "correct", "typed_char", "target_char")},
                       "logged_ms": event["sample"] / SAMPLES_PER_MS,
                       "time_ms": center / SAMPLES_PER_MS if center is not None else None,
                       "snr_db": level})
    return {
        "duration_ms": len(samples) / SAMPLES_PER_MS,
        "press_lag_ms": labels["press_lag_ms"],
        "release_lag_ms": labels["release_lag_ms"],
        "events": events,
        "novelty": np.round(novelty[:int(len(samples) / SAMPLES_PER_MS) + 1], 2).tolist(),   # one per ms
        "onset_strength_max": float(onset_strength(envelope).max()),
        "flux_threshold": pipeline.detector.flux_threshold,
        "press_threshold": pipeline.detector.press_threshold,
        "candidates": [{"time_ms": float(c / SAMPLES_PER_MS), "press_probability": round(float(p), 4)}
                       for c, p in zip(centers, press_probability)],
        "background_db": round(floor_db, 1),
        "filters": FILTERS,
    }


def keystroke_detail(samples, sample, pipeline, top=8):
    """Everything the models compute for one press position."""
    view = pipeline.classifier.input_view(samples, sample)
    probabilities = pipeline.classify(samples, [sample])[0]
    order = np.argsort(-probabilities)[:top]
    features = event_features(samples, [sample])[0]
    press_probability = float(pipeline.detector.classifier.predict_proba(features[None])[0, pipeline.detector.press_column])
    half = 60 * SAMPLES_PER_MS
    lo, hi = max(0, sample - half), min(len(samples), sample + half)
    return {
        "time_ms": sample / SAMPLES_PER_MS,
        "input_view": view,                                       # matrix (time steps, bands)
        "classifier": {"id": pipeline.classifier.id, "name": pipeline.classifier.name, "kind": pipeline.classifier.kind},
        "top": [{"key": pipeline.key_names[i], "probability": round(float(probabilities[i]), 4)} for i in order],
        "press_probability": round(press_probability, 4),
        "features": {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, features)},
        "waveform": {"start_ms": lo / SAMPLES_PER_MS,
                     "raw": np.round(samples[lo:hi:4], 5).tolist(),
                     "model": np.round(highpass(samples[max(0, lo - 12000):hi + 12000], 700)[lo - max(0, lo - 12000):][:hi - lo:4], 5).tolist(),
                     "step_samples": 4},
    }
