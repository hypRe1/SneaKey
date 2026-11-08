"""
Turns per-press key probabilities into English words using a dictionary.

The audio model gives, for every detected key press, a score for each of the 46 keys. Read on its own, the most
likely key is right only about half the time. But the letters of a word constrain each other: "tje" is not a
word, "the" is very common. So instead of taking the top key per press, we look for the sequence of dictionary
words that best explains the presses.

  1. Presses that are probably Space or punctuation split the recording into runs of letter presses.
  2. Within a run, every candidate word is aligned to the presses. The alignment allows for a key the detector
     missed (a letter of the word with no press) and for a sound that wasn't a key (a press with no letter);
     each costs a fixed penalty. The score is the log probability of the word's letters at the aligned presses,
     minus penalties, plus the word's log probability in English (common words cost less).
  3. A run may hold more than one word, in case a Space press was missed, at a cost per extra word.
  4. If no word fits a stretch of presses well, its raw letters are kept, so names and unusual words survive.

The weights are tuned on validation recordings by training/tune_decoder.py.

Each decoded piece also records which presses it came from ("presses": indices into the input rows) and, for
words, the decoder's scores for its best options ("scores"), so the interface can show per-word probability
distributions and a confidence. These additions do not change the decoded text.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MODELS_DIR

WORDLIST = MODELS_DIR / "wordlist.json"
TUNED_SETTINGS = MODELS_DIR / "decoder_settings.json"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
PUNCTUATION = {"Comma": ",", "Period": ".", "Semicolon": ";", "Quote": "'", "Slash": "/", "Minus": "-"}
SILENT = {"ShiftLeft", "ShiftRight", "CapsLock"}       # keys that don't produce a character
MAX_WORD_LENGTH = 14


@dataclass
class DecoderSettings:
    temperature: float = 4.0        # scales SVM margins before softmax; higher = less confident
    space_threshold: float = 0.5    # a press is a word break if P(Space) is at least this
    frequency_weight: float = 1.0   # weight of the word's log probability in English
    missed_key_cost: float = 4.0    # a letter of the word with no press
    extra_press_cost: float = 3.0   # a press that isn't part of any word
    split_cost: float = 2.0         # an extra word inside a run, i.e. a missed Space
    raw_word_cost: float = 20.0     # keeping raw letters instead of a dictionary word
    min_zipf: float = 2.5           # ignore dictionary words rarer than this


def load_tuned_settings():
    """Settings chosen by analysis/tune_word_decoder.py, or the defaults if it hasn't been run."""
    if TUNED_SETTINGS.exists():
        return DecoderSettings(**json.loads(TUNED_SETTINGS.read_text()))
    return DecoderSettings()


class WordDecoder:
    def __init__(self, key_names, settings=None):
        self.key_names = list(key_names)
        # A model trained on few recordings may lack some keys: those letters get probability 0.
        present = [f"Key{c.upper()}" in self.key_names for c in LETTERS]
        self.letter_columns = [self.key_names.index(f"Key{c.upper()}") if ok else 0 for c, ok in zip(LETTERS, present)]
        self.letter_present = np.array(present)
        self.space_column = self.key_names.index("Space") if "Space" in self.key_names else None
        self.frequencies = json.loads(WORDLIST.read_text())["words"]
        self.settings = settings or DecoderSettings()

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, settings):
        self._settings = settings
        self._build_dictionary(settings.min_zipf)

    def _build_dictionary(self, min_zipf):
        """For each word length: the words, their letters as indices 0-25, and their Zipf frequencies."""
        if getattr(self, "_dictionary_zipf", None) == min_zipf:
            return
        by_length = {}
        for word, zipf in self.frequencies.items():
            if zipf >= min_zipf and word.isascii() and word.isalpha() and len(word) <= MAX_WORD_LENGTH:
                by_length.setdefault(len(word), []).append((word.lower(), zipf))
        self.dictionary = {}
        for length, entries in by_length.items():
            words = [word for word, _ in entries]
            letters = np.array([[LETTERS.index(c) for c in word] for word in words])
            self.dictionary[length] = (words, letters, np.array([zipf for _, zipf in entries]))
        self._dictionary_zipf = min_zipf

    # ---- public ----

    def decode(self, key_scores):
        """key_scores: (presses, keys) SVM margins in time order.
        Returns {"text", "words": [...]}. Each word has the decoded "text", the "raw" top keys it came from,
        whether it came from the dictionary, the next-best "alternatives", and "separator" for spaces/punctuation."""
        pieces = []
        for kind, value, indices in self._split_into_runs(self._probabilities(key_scores)):
            if kind == "run":
                pieces += self._decode_run(value, indices)
            else:
                pieces.append({"text": value, "raw": value, "alternatives": [], "separator": True, "presses": indices})
        return {"text": _join(pieces), "words": pieces}

    # ---- step 1: runs of letter presses ----

    def _probabilities(self, key_scores):
        margins = np.asarray(key_scores, dtype=float) / self.settings.temperature
        weights = np.exp(margins - margins.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    def _split_into_runs(self, probabilities):
        """Yields ("run", letter log-probabilities with one row per press, press indices) and
        ("separator", text, [press index]) in time order."""
        run, indices = [], []
        for index, row in enumerate(probabilities):
            key = self.key_names[int(row.argmax())]
            p_space = row[self.space_column] if self.space_column is not None else 0.0
            if p_space >= self.settings.space_threshold or key == "Space":
                yield from _finish(run, indices)
                run, indices = [], []
                yield "separator", " ", [index]
            elif key in PUNCTUATION:
                yield from _finish(run, indices)
                run, indices = [], []
                yield "separator", PUNCTUATION[key], [index]
            elif key == "Backspace":
                if run:
                    run.pop()
                    indices.pop()
            elif key not in SILENT:
                letters = np.where(self.letter_present, row[self.letter_columns], 0.0)
                run.append(np.log(letters / max(letters.sum(), 1e-12) + 1e-6))
                indices.append(index)
        yield from _finish(run, indices)

    # ---- steps 2-4: words within a run ----

    def _decode_run(self, presses, indices):
        """Best split of a run of presses into words. best[k] is the best score for the first k presses."""
        s = self.settings
        spans = self._span_candidates(presses)
        count = len(presses)
        best = [0.0] + [-np.inf] * count
        choice = [None] * (count + 1)
        for end in range(1, count + 1):
            for start in range(end):
                if (start, end) not in spans:
                    continue
                score = best[start] + spans[(start, end)][0][0] - (s.split_cost if start > 0 else 0)
                if score > best[end]:
                    best[end], choice[end] = score, start

        pieces = []
        end = count
        while end > 0:
            start = choice[end]
            ranked = spans[(start, end)]
            raw = "".join(LETTERS[i] for i in presses[start:end].argmax(axis=1))
            pieces.append({"text": ranked[0][1], "raw": raw, "from_dictionary": ranked[0][2],
                           "alternatives": [option[1] for option in ranked[1:4]], "separator": False,
                           "presses": list(indices[start:end]),
                           "scores": [round(float(option[0]), 3) for option in ranked[:4]]})
            end = start
        pieces.reverse()
        # Words found inside one run had a missed Space between them.
        spaced = []
        for i, piece in enumerate(pieces):
            if i:
                spaced.append({"text": " ", "raw": "", "alternatives": [], "separator": True, "presses": []})
            spaced.append(piece)
        return spaced

    def _span_candidates(self, presses):
        """For every stretch presses[start:end], the best few (score, word, from_dictionary) options, best first."""
        s = self.settings
        count = len(presses)
        spans = {}
        for start in range(count):
            remaining = presses[start:]
            longest_span = min(len(remaining), MAX_WORD_LENGTH + 2)
            raw_scores = np.cumsum(remaining.max(axis=1))
            options = {end: [(raw_scores[end - start - 1] - s.raw_word_cost,
                              "".join(LETTERS[i] for i in presses[start:end].argmax(axis=1)), False)]
                       for end in range(start + 1, start + longest_span + 1)}
            for length, (words, letters, zipf) in self.dictionary.items():
                if length > longest_span + 2:
                    continue
                # Alignment scores after each press: rows[i] scores presses start .. start+i against each word.
                for used, scores in self._alignment_scores(remaining[:longest_span], letters):
                    if abs(used - length) > 2:
                        continue
                    total = scores + s.frequency_weight * log_probability(zipf)
                    for index in np.argpartition(-total, min(3, len(total) - 1))[:3]:
                        options[start + used].append((float(total[index]), words[index], True))
            for end, ranked in options.items():
                ranked.sort(key=lambda option: -option[0])
                unique = []
                for option in ranked:
                    if option[1] not in (existing[1] for existing in unique):
                        unique.append(option)
                spans[(start, end)] = unique[:4]
        return spans

    def _alignment_scores(self, presses, word_letters):
        """Aligns presses to every word in word_letters (words, length) at once. Yields (presses used, scores)
        after each press. Cell [j] of a row is the best score with the first j letters of each word."""
        s = self.settings
        length = word_letters.shape[1]
        count = len(word_letters)
        row = np.array([-s.missed_key_cost * j for j in range(length + 1)], dtype=float)[:, None].repeat(count, 1)
        for i, press in enumerate(presses, start=1):
            previous, row = row, np.empty_like(row)
            row[0] = previous[0] - s.extra_press_cost
            for j in range(1, length + 1):
                letter_match = previous[j - 1] + press[word_letters[:, j - 1]]
                row[j] = np.maximum(letter_match, np.maximum(previous[j] - s.extra_press_cost, row[j - 1] - s.missed_key_cost))
            yield i, row[length]


def log_probability(zipf):
    """Natural-log probability of a word from its Zipf frequency (Zipf 9 = once per word, 6 = once per 1000)."""
    return (zipf - 9) * np.log(10)


def _finish(run, indices):
    if run:
        yield "run", np.array(run), list(indices)


def _join(pieces):
    text = "".join(piece["text"] for piece in pieces)
    return " ".join(text.split())
