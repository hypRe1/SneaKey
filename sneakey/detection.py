"""Finding key presses in audio: keyboard_timing_v6's detector and press classifier.

Ported from vendor/keyboard_timing_v6 (detector.py, train_events.py) with the numerics unchanged; the parity test
in tests/test_parity.py checks the output against the original code. Two stages:

  1. Candidates. The 180 Hz high-passed audio's spectral flux (summed positive change in STFT magnitude, one frame
     per millisecond) is divided by its own 251 ms running median, so steady background noise maps to ~1 and
     transients stand out. Peaks above `flux_threshold` are candidates, then moved to the loudest millisecond of the
     1.2 kHz high-passed energy within +/-3 ms.
  2. Presses. A gradient-boosting classifier, trained on log-labelled events, scores each candidate as a key press
     versus a key release or background noise, from 13 loudness/shape/spectrum/spacing features.
"""
import json
import sys

import joblib
import numpy as np
from scipy import fft, ndimage, signal

from .audio import highpass, highpass_sos
from .config import MODELS_DIR, SAMPLE_RATE, SAMPLES_PER_MS

FEATURE_NAMES = ["log_peak", "log_attack", "log_before", "log_after", "log_tail", "log_attack_before",
                 "log_attack_after", "log_crest", "low_band", "mid_band", "high_band", "log_prev_gap", "log_next_gap"]


def spectral_flux_novelty(samples):
    """The detector's novelty curve: spectral flux relative to its running median, one value per millisecond."""
    y = highpass(samples, 180)
    _, _, z = signal.stft(y, SAMPLE_RATE, nperseg=384, noverlap=336, boundary="zeros", padded=True)
    flux = np.r_[0, np.maximum(np.diff(np.abs(z), axis=1), 0).sum(axis=0)]
    floor = ndimage.median_filter(flux, 251)
    return flux / np.maximum(floor, np.percentile(flux, 15) + 1e-8)


def click_envelope(samples):
    """Mean power of the 1.2 kHz high-passed audio over 1 ms, one value per millisecond."""
    y = highpass(samples, 1200)
    return ndimage.uniform_filter1d(y * y, SAMPLES_PER_MS)[::SAMPLES_PER_MS]


def find_candidates(samples, threshold, novelty=None):
    """Sample indices of candidate key sounds (v6 detector.detect)."""
    relative = spectral_flux_novelty(samples) if novelty is None else novelty
    peaks, _ = signal.find_peaks(relative, height=threshold, prominence=0.45 * threshold, distance=18)
    envelope = click_envelope(samples)
    refined = np.array([max(0, p - 3) + envelope[max(0, p - 3):min(len(envelope), p + 4)].argmax()
                        for p in peaks if p + 4 < len(envelope)])
    return (np.unique(refined) * SAMPLES_PER_MS).astype(int)


def event_features(samples, centers):
    """v6's 13 press/release/background features for each candidate (train_events.event_features)."""
    y = signal.sosfiltfilt(highpass_sos(700), samples)
    p = np.asarray(centers, dtype=int)
    freq = np.fft.rfftfreq(2048, 1 / SAMPLE_RATE)
    band_masks = [(freq >= a) & (freq < b) for a, b in [(0, 2000), (2000, 8000), (8000, 24000)]]
    features = []
    for i, c in enumerate(p):
        def power(a, b):
            z = y[max(0, c + a * 48):min(len(y), c + b * 48)]
            return float(np.mean(z * z)) + 1e-12 if len(z) else 1e-12

        before, attack, after, tail = power(-25, -5), power(-3, 4), power(4, 12), power(12, 30)
        clip = y[max(0, c - 10 * 48):min(len(y), c + 22 * 48)]
        peak = float(np.max(clip * clip)) + 1e-12
        spec = np.abs(fft.rfft(clip * np.hanning(len(clip)), n=2048)) ** 2
        total = spec.sum() + 1e-12
        bands = [float(spec[mask].sum() / total) for mask in band_masks]
        previous_gap = (c - p[i - 1]) / 48 if i else 1000
        next_gap = (p[i + 1] - c) / 48 if i + 1 < len(p) else 1000
        features.append([np.log(peak), np.log(attack), np.log(before), np.log(after), np.log(tail),
                         np.log(attack / before), np.log(attack / after), np.log(peak / attack), *bands,
                         np.log1p(previous_gap), np.log1p(next_gap)])
    return np.array(features).reshape(len(p), len(FEATURE_NAMES))


def match(truth_ms, predicted_ms, tolerance=20):
    """One-to-one matching of predicted to true times within `tolerance` ms, minimising total distance
    (v6 detector.match). Returns, for each true time, the index of its predicted time or -1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    truth_ms, predicted_ms = np.asarray(truth_ms, float), np.asarray(predicted_ms, float)
    n, m = len(truth_ms), len(predicted_ms)
    if n == 0:
        return np.zeros(0, int)
    rows, cols, cost = [], [], []
    for i, t in enumerate(truth_ms):
        for j in np.flatnonzero(np.abs(predicted_ms - t) < tolerance):
            rows.append(i)
            cols.append(j)
            cost.append(0.001 + abs(predicted_ms[j] - t) / tolerance)
        rows.append(i)
        cols.append(m + i)
        cost.append(n + 1.0)
    r, c = min_weight_full_bipartite_matching(coo_matrix((cost, (rows, cols)), shape=(n, n + m)).tocsr())
    result = np.full(n, -1, int)
    good = c < m
    result[r[good]] = c[good]
    return result


def _load_event_classifier(path):
    # Saved with scikit-learn 1.8, which pickled one class under the module name "_loss".
    import sklearn._loss._loss
    sys.modules.setdefault("_loss", sklearn._loss._loss)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # version-mismatch warning: the estimator is unchanged and tested
        return joblib.load(path)


class PressDetector:
    def __init__(self, models_dir=MODELS_DIR):
        self.settings = json.loads((models_dir / "detector.json").read_text())
        self.classifier = _load_event_classifier(models_dir / "event_classifier.joblib")
        self.press_column = list(self.classifier.classes_).index("keydown")
        self.flux_threshold = self.settings["flux_threshold"]
        self.press_threshold = self.settings["press_threshold"]

    def candidates(self, samples, novelty=None):
        return find_candidates(samples, self.flux_threshold, novelty)

    def press_probabilities(self, samples, centers):
        if len(centers) == 0:
            return np.zeros(0)
        return self.classifier.predict_proba(event_features(samples, centers))[:, self.press_column]

    def detect(self, samples):
        """All candidates, each one's press probability, and which are kept as presses."""
        centers = self.candidates(samples)
        probability = self.press_probabilities(samples, centers)
        return centers, probability, probability >= self.press_threshold
