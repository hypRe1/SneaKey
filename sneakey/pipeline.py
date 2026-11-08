"""The prediction pipeline: audio in, key presses, per-key probabilities, text and words out.

    detection (v6)  ->  key classifier (CNN)  ->  raw text  +  dictionary word decoder

Loaded once per server process. Nothing here reads the keystroke log: predictions are audio-only.
"""
import numpy as np

from . import models
from .config import SAMPLES_PER_MS
from .decoder import WordDecoder, load_tuned_settings
from .detection import PressDetector

PRINTABLE = {"Space": " ", "Comma": ",", "Period": ".", "Semicolon": ";", "Quote": "'", "Slash": "/", "Minus": "-",
             "Equal": "=", "Backslash": "\\", "IntlBackslash": "\\"}


def render_text(codes):
    """Key codes to text the way v6 does (infer_audio.render): lowercase letters, Backspace deletes, no Shift."""
    text = ""
    for code in codes:
        if code == "Backspace":
            text = text[:-1]
        elif code.startswith("Key"):
            text += code[-1].lower()
        elif code.startswith("Digit"):
            text += code[-1]
        elif code in PRINTABLE:
            text += PRINTABLE[code]
    return text


class Pipeline:
    def __init__(self, classifier=None):
        """classifier: a models.load() result; default the active model (data/active_model.json)."""
        self.classifier = classifier or models.load(models.active_id())
        # A model trained for another microphone may carry its own press detector (training.train_detector).
        self.detector = getattr(self.classifier, "detector", None) or PressDetector()
        self.key_names = self.classifier.key_names
        self.decoder = WordDecoder(self.key_names, load_tuned_settings())

    @property
    def version(self):
        """Changes whenever a model file changes; used to key cached predictions."""
        return self.classifier.version

    def classify(self, samples, centers):
        """Key probabilities (rows over self.key_names) for presses at the given sample indices."""
        return self.classifier.classify(samples, np.asarray(centers, dtype=int))

    def predict(self, samples):
        centers, press_probability, is_press = self.detector.detect(samples)
        presses = centers[is_press]
        probabilities = self.classify(samples, presses) if len(presses) else np.zeros((0, len(self.key_names)))
        codes = [self.key_names[i] for i in probabilities.argmax(axis=1)]
        words = self.decoder.decode(np.log(probabilities + 1e-9)) if len(presses) else {"text": "", "words": []}
        return {
            "model_version": self.version,
            "key_names": self.key_names,
            "presses": [{"sample": int(c), "time_ms": float(c / SAMPLES_PER_MS), "key": code,
                         "press_probability": round(float(p), 4), "confidence": round(float(row.max()), 4)}
                        for c, code, p, row in zip(presses, codes, press_probability[is_press], probabilities)],
            "probabilities": probabilities,
            "candidates": {"time_ms": (centers / SAMPLES_PER_MS).tolist(),
                           "press_probability": np.round(press_probability, 4).tolist()},
            "text": render_text(codes),
            "words": words,
        }
