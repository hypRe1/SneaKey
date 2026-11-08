"""Famous quotes for typing tests, an alternative to the sentence grammar in sentences.py.

THIS FETCHES FROM THE INTERNET, ONCE. The first time a quotes test is requested, the full list (about 1,450 quotes)
is downloaded from dummyjson.com and stored in data/quotes.json. It is a plain GET: nothing about the laptop, the
participants or the recordings is sent. After that the local copy is used, so quotes work offline and a seed
always regenerates the same text.

Quotes are normalised to plain typeable ASCII (curly quotes and dashes straightened). Quotes that would still need
unusual characters, or that are written In Title Case Like This, are skipped. Within a test, quotes already used in
earlier recordings are avoided while unused ones remain; in "coverage" mode, quotes containing under-recorded
letters are preferred, as in the sentence generator.
"""
import json
import random
import re
import threading
import urllib.error
import urllib.request

from . import config
from .sentences import _letter_weight, _trigrams

SOURCE_URL = "https://dummyjson.com/quotes?limit=0"
CACHE_FILE = config.DATA_DIR / "quotes.json"
GENERATOR_VERSION = 1

TYPEABLE = re.compile(r"^[A-Za-z0-9 .,'!?;:\"()-]+$")
REPLACEMENTS = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ",
                "…": "...", " ": " "}

_quotes = None
_lock = threading.Lock()


class Unavailable(RuntimeError):
    """The quote list isn't cached and couldn't be downloaded."""


def normalise(text):
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def usable(text):
    if not TYPEABLE.match(text) or not 20 <= len(text) <= 300:
        return False
    words = re.findall(r"[A-Za-z]+", text)
    capitalised = sum(w[0].isupper() for w in words[1:])
    return len(words) < 5 or capitalised / (len(words) - 1) < 0.5     # skip "Title Case Quotes"


def _download():
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "SneaKey", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = json.load(response)["quotes"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
        raise Unavailable(f"Couldn't download the quote list from dummyjson.com ({error}). "
                          "Check the internet connection, or use random sentences.") from error
    quotes = [{"id": q["id"], "text": normalise(q["quote"]), "author": normalise(q.get("author", ""))} for q in raw]
    quotes = [q for q in quotes if usable(q["text"])]
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"source": SOURCE_URL, "quotes": quotes}, indent=0))
    return quotes


def load_quotes():
    global _quotes
    with _lock:
        if _quotes is None:
            _quotes = json.loads(CACHE_FILE.read_text())["quotes"] if CACHE_FILE.exists() else _download()
    return _quotes


def generate_test(target_chars=350, seed=None, mode="natural", previous_texts=(), key_counts=None, quotes=None):
    """Quotes totalling about target_chars characters, in the same shape as sentences.generate_test.

    Each entry of "sentences" is {"text", "frame": "quote", "author", "id"}.
    """
    quotes = load_quotes() if quotes is None else quotes
    seed = random.SystemRandom().randrange(2 ** 32) if seed is None else int(seed)
    rng = random.Random(seed)
    previous = " ".join(previous_texts)
    unused = [q for q in quotes if q["text"] not in previous]
    pool = unused if sum(len(q["text"]) + 1 for q in unused) >= 3 * target_chars else list(quotes)
    if mode == "coverage":
        word_weight = _letter_weight(key_counts or {})
        weights = [sum(word_weight(w) for w in q["text"].split()) / len(q["text"].split()) for q in pool]
    else:
        weights = [1.0] * len(pool)

    chosen, total = [], 0
    while total < target_chars and pool:
        index = rng.choices(range(len(pool)), weights)[0]
        quote = pool.pop(index)
        weights.pop(index)
        chosen.append({"text": quote["text"], "frame": "quote", "author": quote["author"], "id": quote["id"]})
        total += len(quote["text"]) + 1

    full = " ".join(q["text"] for q in chosen)
    seen = set()
    for text in previous_texts:
        seen |= _trigrams(text)
    return {"text": full, "sentences": chosen, "seed": seed, "mode": mode, "source": "quotes",
            "version": GENERATOR_VERSION, "repeated_trigrams": len(_trigrams(full) & seen)}
