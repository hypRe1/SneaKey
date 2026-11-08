"""EXPERIMENT (not used by the app): decoding key probabilities with a remote LLM (DeepSeek) instead of the local
dictionary decoder.

Result on held-out session 182727 (947 detected presses, same CNN probabilities for every decoder):
    keys only         CER  38.7%   WER  92.9%
    dictionary        CER  29.5%   WER  67.9%   <- the app's decoder
    DeepSeek (this)   CER 129.7%   WER 204.2%   it wrote pangrams ("the quick brown fox...") for the first chunk,
                                                then mostly echoed the top letters
    DeepSeek, word-slot prompt variant: CER 889% (generated long invented, repeating text); not kept.
    DeepSeek, candidate letters (style="candidates": top-3 letters per press, grouped into word slots, plus the
      most-likely-letter string; reply length capped): CER 55.2%, WER 82.7%, 53/168 words exactly right
      (dictionary: 68/168). It recovers some phrases but rewrites others into plausible, wrong sentences.
    DeepSeek, ensemble (style="ensemble": per press the averaged candidates of the CNN + an SVM + an LDA, none
      trained on this session, plus the dictionary decoder's guess per word slot):
        with guesses from the dictionary decoder on the averaged probabilities: CER 36.9%, WER 72.0%, 61/168 words
        with guesses from the CNN dictionary decoder (the best local decoder):  CER 35.3%, WER 73.2%, 70/168 words
      It mostly copies the guesses; of its own changes some fix words ("hat" -> "that") and more invent them
      ("h" -> "people"). Equal-weight averaging also hurts: the dictionary decoder on the CNN+SVM+LDA average
      scores CER 32.2% vs 29.5% on the CNN alone, because the SVM/LDA are weaker.
The LLM either invents fluent text or copies the noise; it does not do the alignment the dictionary decoder does.
Kept as a reproducible, opt-in experiment: POST /api/sessions/{id}/llm-decode.

THIS SENDS DATA OFF THE LAPTOP. For each detected key press, the classifier's most likely keys and their
probabilities are sent to api.deepseek.com. The audio and the typed text are never sent, but the typed text can
largely be reconstructed from what is sent, which is the point of the experiment. Only use it on recordings whose
participants agreed to that. The key is read from data/deepseek.env (git-ignored) or the DEEPSEEK_API_KEY
environment variable, never from the code.

How it works:
  * Each detected press becomes one line: its top keys with probabilities, e.g. "t .62 r .15 y .08".
    "_" is the space bar, "<" is backspace.
  * Presses are sent in chunks of about 120 (split at a likely space), each with the text decoded so far as
    context, so long recordings fit in the model's context.
  * The model returns only its reconstruction, which is scored with the same lowercase CER/WER as the other
    decoders, against the typed text it never sees.
"""
import hashlib
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np

from . import config

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
PROMPT_VERSION = 2
CHUNK_PRESSES = 120
TOP_K = 4
SYMBOLS = {"Space": "_", "Backspace": "<", "Comma": ",", "Period": ".", "Slash": "?", "Quote": "'", "Semicolon": ";",
           "Minus": "-"}

SYSTEM_PROMPT = """You decode keyboard acoustic side-channel output into the text that was typed.

Each input line is one detected key press, in time order: the classifier's most likely keys with probabilities.
"_" is the space bar, "<" is backspace (it deletes the previous character), letters are lowercase.
The evidence is noisy:
- About 1 in 5 typed keys was not detected at all (missing presses), and about 1 in 7 lines is a spurious press.
- The most likely key is right about 80% of the time; when it is wrong, the true key is usually listed second or
  third, and is often a physical neighbour on a QWERTY keyboard.
- Spaces are the most reliable key.
The text is English sentences, possibly unusual or nonsensical ones, so follow the evidence rather than
guessing a familiar phrase. Do not add words for which there is no evidence.

Reply with ONLY the reconstructed text for the presses given, in lowercase, with spaces and basic punctuation
(. , ?). No quotes, explanations or notes."""


CANDIDATES_SYSTEM_PROMPT = """You reconstruct typed English text from a keyboard acoustic side-channel.

For each word that was typed you get a word slot: a sequence of brackets, one per detected key press, in typing
order. Each bracket lists the 3 most likely letters for that press, most likely first, e.g. [t/r/y].
To reconstruct a word, pick one letter from each bracket. The first letter is right about 80% of the time; when
it is wrong, the right letter is usually second or third. Detection is imperfect, so you may also:
- skip a bracket (a spurious press, about 1 in 7), or
- add a letter that has no bracket (a missed press, about 1 in 5 letters).
Do this only when no word fits the brackets as they are. A slot is usually one word, occasionally two (a missed
space) or part of one (a spurious space). "<" in a bracket is backspace.
The sentences are English but may be unusual or nonsensical: follow the brackets, never substitute a familiar
phrase, and output roughly one word per slot.

Reply with ONLY the reconstructed text, lowercase, with spaces and basic punctuation (. , ?)."""

ENSEMBLE_SYSTEM_PROMPT = """You reconstruct typed English text from a keyboard acoustic side-channel.

Several independent key classifiers listened to each key press; their probabilities are averaged. The text is
given as word slots, one per line, split where the space bar was most likely pressed. Each line has:
- dict="...": the best guess of a separate dictionary-matching decoder for that slot. It is often right, but can
  be a wrong real word, or raw letters when no word fitted.
- one bracket per detected key press, in typing order, with the likely letters and their averaged probability,
  e.g. [t .62/r .15/y .08]. "<" is backspace.
The top letter is right about 80% of the time; otherwise the right one is usually listed second or third.
About 1 in 5 typed letters has no bracket (missed press) and about 1 in 7 brackets is spurious, so a word may
need one letter added or one bracket skipped. A slot is usually one word, occasionally two (missed space).
Combine the brackets and the dictionary guess to infer the words actually typed. The sentences are English but
may be unusual or nonsensical: follow the evidence, never substitute a familiar phrase, output about one word
per slot.

Reply with ONLY the reconstructed text, lowercase, with spaces and basic punctuation (. , ?)."""

STYLES = ("probabilities", "candidates", "ensemble")


class NotConfigured(Exception):
    pass


def api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    path = config.DATA_DIR / "deepseek.env"
    if not key and path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise NotConfigured("No DeepSeek API key: put DEEPSEEK_API_KEY=... in data/deepseek.env.")
    return key


def is_configured():
    try:
        api_key()
        return True
    except NotConfigured:
        return False


def symbol(code):
    if code.startswith("Key") and len(code) == 4:
        return code[3].lower()
    if code.startswith("Digit"):
        return code[5]
    return SYMBOLS.get(code)


def press_lines(probabilities, key_names):
    """One text line per press: top keys (that have a symbol) with probabilities."""
    symbols = [symbol(k) for k in key_names]
    lines = []
    for row in np.asarray(probabilities, dtype=float):
        options = [(symbols[i], row[i]) for i in np.argsort(-row) if symbols[i] is not None][:TOP_K]
        lines.append(" ".join(f"{s} {p:.2f}".replace(" 0.", " .") for s, p in options if p >= 0.01) or "?")
    return lines


def chunks(lines, size=CHUNK_PRESSES):
    """Split into chunks of about `size` presses, ending each at a press whose top key is a space if possible."""
    out, start = [], 0
    while start < len(lines):
        end = min(len(lines), start + size)
        if end < len(lines):
            for i in range(end, max(start + size // 2, start + 1), -1):
                if lines[i - 1].startswith("_ "):
                    end = i
                    break
        out.append((start, end))
        start = end
    return out


def _call(messages, key, max_tokens=1024):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}).encode()
    request = urllib.request.Request(API_URL, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read()[:300].decode(errors="replace")
        if error.code == 401:
            raise NotConfigured("DeepSeek rejected the API key (401): it may have been revoked.") from None
        raise RuntimeError(f"DeepSeek API error {error.code}: {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Couldn't reach DeepSeek: {error.reason}") from None
    return data["choices"][0]["message"]["content"], data.get("model", MODEL), data.get("usage", {})


def _clean(text):
    text = re.sub(r"^```\w*|```$", "", text.strip()).strip().strip('"').strip()
    return " ".join(text.lower().split())


def candidate_slots(probabilities, key_names, top=3):
    """Word slots of candidate letters: a press whose most likely key is Space ends a slot; every other press
    becomes "[a/b/c]", its top letters (or other typed symbols), most likely first. Returns (slots, argmax text)."""
    symbols = [symbol(k) for k in key_names]
    slots, current, argmax = [], [], []
    for row in np.asarray(probabilities, dtype=float):
        ranked = [symbols[i] for i in np.argsort(-row) if symbols[i] is not None]
        argmax.append(ranked[0] if ranked else "?")
        if ranked and ranked[0] == "_":
            if current:
                slots.append(current)
            current = []
            continue
        current.append("[" + "/".join([c for c in ranked if c != "_"][:top]) + "]")
    if current:
        slots.append(current)
    return slots, "".join(argmax).replace("_", " ")


def decode(probabilities, key_names, style="probabilities", progress=lambda done, total: None):
    """Returns {"text", "chunks", "model", "usage", "prompt_example", "style"}."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if style == "candidates":
        return _decode_candidates(probabilities, key_names, progress)
    if style == "ensemble":
        raise ValueError("Use decode_ensemble() for the ensemble style.")
    key = api_key()
    lines = press_lines(probabilities, key_names)
    parts, decoded, usage = [], "", {"prompt_tokens": 0, "completion_tokens": 0}
    model = MODEL
    spans = chunks(lines)
    first_prompt = None
    for n, (start, end) in enumerate(spans):
        context = f"Text decoded so far (for context only, do not repeat it):\n{decoded[-300:]}\n\n" if decoded else ""
        prompt = f"{context}Presses {start + 1}-{end} of {len(lines)}:\n" + "\n".join(lines[start:end])
        first_prompt = first_prompt or prompt
        reply, model, used = _call([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], key)
        text = _clean(reply)
        parts.append({"presses": [start, end], "text": text})
        decoded = f"{decoded} {text}".strip()
        for k in usage:
            usage[k] += int(used.get(k, 0))
        progress(n + 1, len(spans))
    return {"text": decoded, "chunks": parts, "model": model, "usage": usage, "presses_sent": len(lines),
            "system_prompt": SYSTEM_PROMPT, "prompt_example": first_prompt or "", "style": "probabilities"}


def _decode_candidates(probabilities, key_names, progress, slots_per_chunk=25):
    key = api_key()
    slots, argmax = candidate_slots(probabilities, key_names)
    argmax_words = argmax.split()
    parts, decoded, usage = [], "", {"prompt_tokens": 0, "completion_tokens": 0}
    model, first_prompt = MODEL, None
    spans = [(i, min(len(slots), i + slots_per_chunk)) for i in range(0, len(slots), slots_per_chunk)]
    for n, (start, end) in enumerate(spans):
        context = f"Text decoded so far (for context only, do not repeat it):\n{decoded[-300:]}\n\n" if decoded else ""
        prompt = (f"{context}Most likely letter per press: '{' '.join(argmax_words[start:end])}'\n"
                  f"Word slots {start + 1}-{end} of {len(slots)}:\n"
                  + "\n".join(f"{start + i + 1}: {' '.join(slot)}" for i, slot in enumerate(slots[start:end])))
        first_prompt = first_prompt or prompt
        # Cap the reply at what the slots can hold (about one word each): one chunk once ran away into 600+ words.
        reply, model, used = _call([{"role": "system", "content": CANDIDATES_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                                   key, max_tokens=6 * (end - start) + 20)
        text = _clean(reply)
        parts.append({"slots": [start, end], "text": text})
        decoded = f"{decoded} {text}".strip()
        for k in usage:
            usage[k] += int(used.get(k, 0))
        progress(n + 1, len(spans))
    return {"text": decoded, "chunks": parts, "model": model, "usage": usage, "presses_sent": len(probabilities),
            "system_prompt": CANDIDATES_SYSTEM_PROMPT, "prompt_example": first_prompt or "", "style": "candidates"}


def average_probabilities(model_outputs):
    """model_outputs: [(probabilities (presses, keys), key_names)] for the same presses. Returns (mean probabilities
    over the union of keys, union key names); a key a model lacks counts as probability 0 for that model."""
    keys = sorted({k for _, names in model_outputs for k in names})
    index = {k: i for i, k in enumerate(keys)}
    total = np.zeros((len(model_outputs[0][0]), len(keys)))
    for probabilities, names in model_outputs:
        total[:, [index[k] for k in names]] += np.asarray(probabilities, dtype=float)
    return total / len(model_outputs), keys


def ensemble_slots(probabilities, key_names, words, top=3):
    """Word slots for the ensemble prompt. probabilities: averaged (presses, keys); words: the dictionary decoder's
    output on the same presses (pieces with "presses"). A press whose top key is Space ends a slot."""
    symbols = [symbol(k) for k in key_names]
    piece_of = {}
    for n, piece in enumerate(words["words"]):
        if not piece["separator"]:
            for i in piece["presses"]:
                piece_of[i] = n
    slots, current = [], []
    for i, row in enumerate(np.asarray(probabilities, dtype=float)):
        ranked = [(symbols[j], row[j]) for j in np.argsort(-row) if symbols[j] is not None]
        if ranked and ranked[0][0] == "_":
            if current:
                slots.append(current)
            current = []
            continue
        options = [(c, p) for c, p in ranked if c != "_"][:top]
        current.append((i, "[" + "/".join(f"{c} {p:.2f}".replace(" 0.", " .") for c, p in options if p >= 0.02 or c == options[0][0]) + "]"))
    if current:
        slots.append(current)
    lines = []
    for slot in slots:
        pieces = sorted({piece_of[i] for i, _ in slot if i in piece_of})
        guess = " ".join(words["words"][n]["text"] for n in pieces)
        lines.append(f'dict="{guess}" | ' + " ".join(b for _, b in slot))
    return lines


def decode_ensemble(probabilities, key_names, words, progress=lambda done, total: None, slots_per_chunk=25):
    """Ensemble style: averaged candidates from several classifiers plus the dictionary decoder's word guesses."""
    key = api_key()
    lines = ensemble_slots(probabilities, key_names, words)
    parts, decoded, usage = [], "", {"prompt_tokens": 0, "completion_tokens": 0}
    model, first_prompt = MODEL, None
    spans = [(i, min(len(lines), i + slots_per_chunk)) for i in range(0, len(lines), slots_per_chunk)]
    for n, (start, end) in enumerate(spans):
        context = f"Text decoded so far (for context only, do not repeat it):\n{decoded[-300:]}\n\n" if decoded else ""
        prompt = f"{context}Word slots {start + 1}-{end} of {len(lines)}:\n" + "\n".join(
            f"{start + i + 1}: {line}" for i, line in enumerate(lines[start:end]))
        first_prompt = first_prompt or prompt
        reply, model, used = _call([{"role": "system", "content": ENSEMBLE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                                   key, max_tokens=6 * (end - start) + 20)
        text = _clean(reply)
        parts.append({"slots": [start, end], "text": text})
        decoded = f"{decoded} {text}".strip()
        for k in usage:
            usage[k] += int(used.get(k, 0))
        progress(n + 1, len(spans))
    return {"text": decoded, "chunks": parts, "model": model, "usage": usage, "presses_sent": len(probabilities),
            "system_prompt": ENSEMBLE_SYSTEM_PROMPT, "prompt_example": first_prompt or "", "style": "ensemble"}


def cache_key(probabilities, key_names, style="probabilities"):
    digest = hashlib.sha256(json.dumps([key_names, np.round(np.asarray(probabilities), 4).tolist()]).encode()).hexdigest()
    return f"{digest[:16]}.p{PROMPT_VERSION}.{style}"
