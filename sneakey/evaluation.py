"""Scoring a prediction against what was actually typed.

Two protocols from the original project are kept exactly:

  * Audio-only (v6's evaluation): detected presses are matched one-to-one to the labelled presses within 20 ms.
    Reports detection precision/recall, key accuracy over matched presses, and lowercase character/word error
    rates of the predicted text against the typed text (Levenshtein distance / typed length).
  * Logged timing (the CNN's headline metric): the classifier is run at the true press positions, auto-repeats
    excluded, measuring the classifier on its own, separate from the detector. Accuracy, balanced accuracy
    (mean per-key recall) and top-3.

On top of these, for the interface: word-level alignment of typed and decoded words, a per-key breakdown and a
confusion matrix.
"""
import numpy as np

from .config import MATCH_TOLERANCE_MS, SAMPLES_PER_MS
from .detection import match


# ---- edit distance ----

def edit_distance(a, b):
    """Levenshtein distance between two sequences (v6 evaluate.edit)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def align(a, b):
    """Minimum-edit alignment of two sequences as a list of (op, i, j) with op in equal/substitute/delete/insert;
    i indexes a (None for insert), j indexes b (None for delete)."""
    n, m = len(a), len(b)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1, cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    ops = []
    i, j = n, m
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("equal" if a[i - 1] == b[j - 1] else "substitute", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            ops.append(("delete", i - 1, None))
            i -= 1
        else:
            ops.append(("insert", None, j - 1))
            j -= 1
    return ops[::-1]


def text_error_rates(typed_text, predicted_text):
    """Lowercase character and word error rates, as v6 reports them."""
    typed = typed_text.lower()
    return {
        "cer": edit_distance(typed, predicted_text) / max(len(typed), 1),
        "wer": edit_distance(typed.split(), predicted_text.split()) / max(len(typed.split()), 1),
    }


# ---- key metrics ----

def key_metrics(predicted, truth, probabilities=None, key_names=None):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if not len(truth):
        return {"count": 0, "accuracy": None, "balanced": None, "top3": None}
    hit = predicted == truth
    result = {"count": int(len(truth)), "accuracy": float(hit.mean()),
              "balanced": float(np.mean([hit[truth == key].mean() for key in np.unique(truth)]))}
    if probabilities is not None:
        names = np.asarray(key_names)
        top3 = names[np.argsort(-probabilities, axis=1)[:, :3]]
        result["top3"] = float(np.mean([t in row for t, row in zip(truth, top3)]))
    return result


def per_key(predicted, truth):
    """Support, recall and the most common confusion for each true key."""
    rows = []
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    for key in np.unique(truth):
        mask = truth == key
        wrong = predicted[mask & (predicted != truth)]
        values, counts = np.unique(wrong, return_counts=True)
        rows.append({"key": str(key), "support": int(mask.sum()), "recall": float((predicted[mask] == key).mean()),
                     "confused_with": str(values[counts.argmax()]) if len(values) else None})
    return sorted(rows, key=lambda r: -r["support"])


def confusion(predicted, truth):
    keys = sorted(set(truth) | set(predicted))
    index = {k: i for i, k in enumerate(keys)}
    matrix = np.zeros((len(keys), len(keys)), dtype=int)
    for t, p in zip(truth, predicted):
        matrix[index[t], index[p]] += 1
    return {"keys": keys, "matrix": matrix.tolist()}


# ---- whole evaluation ----

def evaluate(prediction, true_presses, typed_text, logged_probabilities, key_names):
    """
    prediction: Pipeline.predict output. true_presses: groundtruth.true_presses. logged_probabilities: the
    classifier's probabilities at each true press (rows aligned with true_presses).
    """
    true_ms = np.array([p["sample"] for p in true_presses], dtype=float) / SAMPLES_PER_MS
    true_codes = np.array([p["code"] for p in true_presses])
    predicted_ms = np.array([p["time_ms"] for p in prediction["presses"]], dtype=float)
    predicted_codes = np.array([p["key"] for p in prediction["presses"]])

    # Audio-only: v6's matching and metrics, over every labelled keydown.
    matches = match(true_ms, predicted_ms, MATCH_TOLERANCE_MS)
    matched = matches >= 0
    correct = np.zeros(len(true_ms), dtype=bool)
    correct[matched] = predicted_codes[matches[matched]] == true_codes[matched]
    n_matched = int(matched.sum())
    precision = n_matched / max(len(predicted_ms), 1)
    recall = n_matched / max(len(true_ms), 1)
    detection = {
        "typed_presses": int(len(true_ms)), "detected_presses": int(len(predicted_ms)), "matched": n_matched,
        "missed": int(len(true_ms) - n_matched), "extra": int(len(predicted_ms) - n_matched),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "timing_error_ms": float(np.median(np.abs(predicted_ms[matches[matched]] - true_ms[matched]))) if n_matched else None,
    }
    audio_only = {
        "key_accuracy_matched": float(correct[matched].mean()) if n_matched else None,
        "correct_presses": int(correct.sum()),
        "correct_of_typed": float(correct.mean()) if len(correct) else None,
    }

    # For each predicted press, the true key it was matched to (None for an extra press).
    truth_for_prediction = [None] * len(predicted_ms)
    for i in np.flatnonzero(matched):
        truth_for_prediction[matches[i]] = {"code": str(true_codes[i]), "index": int(i)}

    # Logged timing: the classifier alone at the true press positions, auto-repeats excluded.
    not_repeat = np.array([not p["repeat"] for p in true_presses], dtype=bool)
    names = np.asarray(key_names)
    logged_codes = names[logged_probabilities.argmax(axis=1)] if len(logged_probabilities) else np.array([])
    logged = key_metrics(logged_codes[not_repeat], true_codes[not_repeat], logged_probabilities[not_repeat], key_names) \
        if len(logged_probabilities) else key_metrics([], [])

    text = {"typed": typed_text, "raw": prediction["text"], "dictionary": prediction["words"]["text"],
            "raw_rates": text_error_rates(typed_text, prediction["text"]) if typed_text else None,
            "dictionary_rates": text_error_rates(typed_text, prediction["words"]["text"]) if typed_text else None}

    return {
        "detection": detection,
        "audio_only": audio_only,
        "logged_timing": logged,
        "per_key": per_key(logged_codes[not_repeat], true_codes[not_repeat]) if len(logged_codes) else [],
        "confusion": confusion(list(logged_codes[not_repeat]), list(true_codes[not_repeat])) if len(logged_codes) else None,
        "text": text,
        "word_alignment": word_alignment(typed_text, prediction["words"]),
        "truth_for_prediction": truth_for_prediction,
        "true_presses": [{"time_ms": float(t), "code": str(c), "matched": int(matches[i]), "correct": bool(correct[i])}
                         for i, (t, c) in enumerate(zip(true_ms, true_codes))],
        "logged_predictions": [str(c) for c in logged_codes],
    }


def word_alignment(typed_text, decoded):
    """Align typed words with decoded words. Returns rows {op, typed, predicted, piece} where piece indexes
    decoded["words"] (the decoder's non-separator pieces carry the per-press detail)."""
    typed_words = [_clean(w) for w in typed_text.lower().split()]
    pieces = [i for i, piece in enumerate(decoded["words"]) if not piece["separator"]]
    predicted_words = [decoded["words"][i]["text"] for i in pieces]
    rows = []
    for op, i, j in align(typed_words, predicted_words):
        rows.append({"op": op, "typed": typed_words[i] if i is not None else None,
                     "predicted": predicted_words[j] if j is not None else None,
                     "piece": pieces[j] if j is not None else None})
    return rows


def words_correct(typed_text, predicted_text):
    """How many typed words appear exactly in the predicted text, after minimum-edit word alignment."""
    typed = [_clean(w) for w in typed_text.lower().split()]
    predicted = [_clean(w) for w in predicted_text.lower().split()]
    return sum(op == "equal" for op, _, _ in align(typed, predicted))


def _clean(word):
    """Typed words lose punctuation for word comparison: the decoder emits punctuation as separate pieces."""
    return "".join(c for c in word if c.isalnum()) or word
