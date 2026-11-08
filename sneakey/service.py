"""Application logic behind the API: one loaded pipeline, cached results and session roles.

Kept separate from api.py so it can be used from scripts and tests without a web server.
"""
import json
import threading
from collections import Counter
from functools import lru_cache

import numpy as np

from . import analysis, config, llm_decoder, models, phone, sessions, training
from .evaluation import text_error_rates, words_correct
from .audio import read_wav
from .evaluation import evaluate
from .groundtruth import true_presses
from .pipeline import Pipeline
from . import quotes
from .sentences import generate_test


class Locked(Exception):
    """The session is in the locked evaluation set."""


_pipeline = None
_pipeline_lock = threading.Lock()


def pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = Pipeline()
    return _pipeline


# ---- models ----

def list_models():
    active = models.active_id()
    return [{**m, "active": m["id"] == active} for m in models.list_models()]


def activate_model(model_id):
    """Switch the pipeline to another key classifier. Cached predictions are per model version, so nothing stale
    is shown; signals and keystroke views are recomputed with the new model."""
    global _pipeline
    try:
        classifier = models.load(model_id)
    except KeyError:
        raise sessions.NotFound(model_id)
    with _pipeline_lock:
        models.set_active(model_id)
        _pipeline = Pipeline(classifier)
    _signals.cache_clear()


def delete_model(model_id):
    try:
        models.delete(model_id)
    except KeyError:
        raise sessions.NotFound(model_id)


def start_training(request):
    return training.start(request, pipeline().classifier).status()


def training_status():
    job = training.current_job()
    return job.status() if job else {"state": "idle"}


@lru_cache(maxsize=4)
def _samples(session_id, modified):
    return read_wav(sessions.audio_path(session_id))


def samples(session_id):
    path = sessions.audio_path(session_id)
    return _samples(session_id, path.stat().st_mtime_ns)


# ---- session roles ----

def role(session_id):
    """How a session relates to the models, for honest evaluation:
    locked     reserved for a final test; Predict refuses it
    test       held-out test set (never trained or tuned on)
    validation used to choose settings / models
    trained    the key classifier was trained on it: its scores are optimistic
    unseen     never used by any model"""
    splits = config.load_splits()
    for name in ("locked", "test"):
        if session_id in splits.get(name, []):
            return name
    classifier = pipeline().classifier
    if session_id in classifier.trained_on:
        return "trained"
    if session_id in splits.get("validation", []) or session_id in classifier.validation_sessions:
        return "validation"     # used to choose this model
    return "unseen"


def list_sessions():
    return [{**s, "role": role(s["id"]), "has_prediction": _cache_path(s["id"]).exists()}
            for s in sessions.list_sessions()]


def session_detail(session_id):
    data = sessions.load_session(session_id)
    data["role"] = role(session_id)
    return data


# ---- recording support ----

def key_counts(session_ids=None):
    """Recorded keydowns (not auto-repeats) per key, over all sessions or the given ones."""
    counts = Counter()
    ids = [s["id"] for s in sessions.list_sessions()] if session_ids is None else session_ids
    for session_id in ids:
        events = sessions.load_session(session_id)["events"]
        counts.update(e["code"] for e in events if e["type"] == "keydown" and not e["repeat"])
    return dict(counts)


def new_test(target_chars=350, mode="natural", seed=None, source="sentences"):
    previous = [sessions.load_session(s["id"])["target_text"] for s in sessions.list_sessions()]
    generate = quotes.generate_test if source == "quotes" else generate_test
    test = generate(target_chars, seed=seed, mode=mode, previous_texts=previous,
                    key_counts=key_counts() if mode == "coverage" else None)
    return {"source": source, **test}


# ---- analysis ----

@lru_cache(maxsize=8)
def _signals(session_id, modified):
    return analysis.session_signals(sessions.load_session(session_id), samples(session_id), pipeline())


def signals(session_id):
    return _signals(session_id, sessions.audio_path(session_id).stat().st_mtime_ns)


def keystroke(session_id, time_ms):
    audio = samples(session_id)
    sample = int(round(float(time_ms) * config.SAMPLES_PER_MS))
    if not 0 <= sample < len(audio):
        raise ValueError("Time is outside the recording.")
    return analysis.keystroke_detail(audio, sample, pipeline())


# ---- prediction ----

def _cache_path(session_id):
    return config.CACHE_DIR / f"{session_id}.{pipeline().version}.json"


def predict(session_id, refresh=False):
    """Run the pipeline on a session and evaluate it against the keystroke log. Cached per model version."""
    session_role = role(session_id)
    if session_role == "locked":
        raise Locked(f"{session_id} is in the locked evaluation set (data/splits.json).")
    path = _cache_path(session_id)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    session = sessions.load_session(session_id)
    audio = samples(session_id)
    p = pipeline()
    prediction = p.predict(audio)
    presses, labels = true_presses(session, audio)
    logged = p.classify(audio, [q["sample"] for q in presses]) if presses else np.zeros((0, len(p.key_names)))
    result = {
        **{k: v for k, v in prediction.items() if k != "probabilities"},
        "session_id": session_id,
        "role": session_role,
        "probabilities": np.round(prediction["probabilities"], 4).tolist(),
        "press_lag_ms": labels["press_lag_ms"],
        "evaluation": evaluate(prediction, presses, session["typed_text"], logged, p.key_names),
        "model": {"id": p.classifier.id, "name": p.classifier.name, "kind": p.classifier.kind},
    }
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def ensemble_inputs(session_id, model_ids=None):
    """The active model plus other models' probabilities for the same detected presses, averaged, and the local
    dictionary decoder's output on that average. Models trained on this session are skipped (they'd leak)."""
    from .decoder import WordDecoder, load_tuned_settings
    result = predict(session_id)
    active = pipeline().classifier
    audio = samples(session_id)
    centers = [p["sample"] for p in result["presses"]]
    outputs = [(np.array(result["probabilities"]), result["key_names"])]
    used, skipped = [active.id], []
    others = model_ids if model_ids is not None else [m["id"] for m in models.list_models() if m["id"] != active.id]
    for model_id in others:
        model = models.load(model_id)
        if session_id in model.trained_on:
            skipped.append(model_id)
            continue
        outputs.append((model.classify(audio, centers), model.key_names))
        used.append(model_id)
    probabilities, keys = llm_decoder.average_probabilities(outputs)
    words = WordDecoder(keys, load_tuned_settings()).decode(np.log(probabilities + 1e-9))
    return result, probabilities, keys, words, used, skipped


def llm_decode(session_id, refresh=False, style="ensemble", model_ids=None, guesses="active", cached_only=False):
    """Opt-in: decode this session with DeepSeek (SENDS DATA OFF THE LAPTOP; see llm_decoder.py) and score it like
    the other decoders. Cached per session, models, style and prompt version.

    style="ensemble": per press the averaged candidates of the active model and every other model not trained on
    this session, plus a dictionary word guess per slot: from the active model's dictionary decoder
    (guesses="active", the best variant in tests) or from the decoder run on the averaged probabilities
    (guesses="ensemble"). cached_only=True returns a stored result or raises NotFound, and never sends anything."""
    if style not in llm_decoder.STYLES:
        raise ValueError(f"style must be one of {llm_decoder.STYLES}")
    if guesses not in ("active", "ensemble"):
        raise ValueError("guesses must be active or ensemble")
    typed = sessions.load_session(session_id)["typed_text"]
    extra = {}
    if style == "ensemble":
        result, probabilities, keys, ensemble_words, used, skipped = ensemble_inputs(session_id, model_ids)
        words = result["words"] if guesses == "active" else ensemble_words
        key = f"{llm_decoder.cache_key(probabilities, keys, style)}.{guesses}"
        extra = {"models_used": used, "models_skipped_trained_on_session": skipped, "guesses": guesses,
                 "ensemble_dictionary": {"text": ensemble_words["text"],
                                         "rates": text_error_rates(typed, ensemble_words["text"]) if typed else None}}
    else:
        result = predict(session_id)                  # refuses locked sessions; cached
        key = llm_decoder.cache_key(result["probabilities"], result["key_names"], style)
    path = config.CACHE_DIR / f"{session_id}.{result['model_version']}.llm.{key}.json"
    if path.exists() and not refresh:
        return _with_word_counts(json.loads(path.read_text()))
    if cached_only:
        raise sessions.NotFound(session_id)
    if style == "ensemble":
        decoded = llm_decoder.decode_ensemble(probabilities, keys, words)
    else:
        decoded = llm_decoder.decode(result["probabilities"], result["key_names"], style)
    text = result["evaluation"]["text"]
    out = {**decoded, **extra, "session_id": session_id, "typed": typed,
           "rates": text_error_rates(typed, decoded["text"]) if typed else None,
           "dictionary": {"text": text["dictionary"], "rates": text["dictionary_rates"]},
           "raw": {"text": text["raw"], "rates": text["raw_rates"]}}
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return _with_word_counts(out)


def _with_word_counts(out):
    """Words exactly right (after minimum-edit alignment with the typed words) for each decoder's text."""
    if out.get("typed"):
        out["words_right"] = {name: words_correct(out["typed"], text) for name, text in
                              [("llm", out["text"]), ("dictionary", out["dictionary"]["text"]), ("raw", out["raw"]["text"])]}
        out["typed_words"] = len(out["typed"].split())
    return out


def create_phone_session(body):
    """A session from a finished phone recording and the laptop's key log. body: {recording_id, metadata,
    events (rows with event_time, the laptop page time), laptop_pings}."""
    rec, phone_info = phone.take_finished(body["recording_id"])
    events = body["events"]
    times = [float(e["event_time"]) for e in events]
    audio, samples, sync = phone.assemble(rec, body["laptop_pings"], times)
    rows = [{**{k: v for k, v in e.items() if k != "event_time"}, "event_time_ms": round(t),
             "t_rel_ms": f"{s / config.SAMPLES_PER_MS:.3f}", "sample_index_est": int(s)}
            for e, t, s in zip(events, times, samples)]
    metadata = dict(body["metadata"])
    metadata["capture"] = {**metadata.get("capture", {}), "device": "phone", "phone": phone_info}
    metadata["sync"] = sync
    metadata["audio"] = {"sample_rate_hz": config.SAMPLE_RATE, "processing": "phone audio; resampled to 48 kHz if recorded at another rate"}
    session_id = sessions.create_session(metadata, rows)
    sessions.save_audio(session_id, audio.astype("<i2").tobytes(), config.SAMPLE_RATE)
    return {"id": session_id, "sync": sync, "seconds": round(len(audio) / config.SAMPLE_RATE, 1)}


def phone_links(code):
    return [f"https://{address}:{config.PHONE_PORT}/p/{code}" for address in config.PHONE_ADDRESSES]


def summary():
    """Metrics pooled over every session with a cached prediction, grouped by role. Presses and characters are
    pooled (not session means), so long sessions count for more, as in the original evaluation."""
    groups = {}
    for s in sessions.list_sessions():
        path = _cache_path(s["id"])
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        e = r["evaluation"]
        g = groups.setdefault(r["role"], Counter())
        g["sessions"] += 1
        g["typed"] += e["detection"]["typed_presses"]
        g["detected"] += e["detection"]["detected_presses"]
        g["matched"] += e["detection"]["matched"]
        g["correct"] += e["audio_only"]["correct_presses"]
        g["logged_count"] += e["logged_timing"]["count"]
        g["logged_correct"] += round((e["logged_timing"]["accuracy"] or 0) * e["logged_timing"]["count"])
        g["logged_top3"] += round((e["logged_timing"].get("top3") or 0) * e["logged_timing"]["count"])
        typed_len = len(e["text"]["typed"])
        if e["text"]["raw_rates"]:
            g["chars"] += typed_len
            g["raw_char_errors"] += e["text"]["raw_rates"]["cer"] * typed_len
            g["dict_char_errors"] += e["text"]["dictionary_rates"]["cer"] * typed_len
    out = {}
    for name, g in groups.items():
        out[name] = {
            "sessions": g["sessions"], "typed_presses": g["typed"],
            "precision": g["matched"] / max(g["detected"], 1), "recall": g["matched"] / max(g["typed"], 1),
            "key_accuracy_matched": g["correct"] / max(g["matched"], 1),
            "logged_accuracy": g["logged_correct"] / max(g["logged_count"], 1),
            "logged_top3": g["logged_top3"] / max(g["logged_count"], 1),
            "raw_cer": g["raw_char_errors"] / max(g["chars"], 1),
            "dictionary_cer": g["dict_char_errors"] / max(g["chars"], 1),
        }
    return out
