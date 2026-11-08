"""Training key classifiers from recordings, as a background job driven by the Train page.

A run:
  1. Labels every keydown (auto-repeats excluded) of the chosen sessions with the ground-truth method and cuts a
     100 ms slice of 700 Hz high-passed audio around it (classifier.slices_around: the same code inference uses).
     Slices are cached per session.
  2. Trains one model kind on the training sessions:
       svm  RBF SVM on band64+mel features, Platt-calibrated probabilities (sklearn SVC(probability=True))
       lda  shrinkage LDA on the same features: trains in seconds, a strong simple baseline
       cnn  the same network and recipe as the shipped CNN (training/train_cnn.py)
     Keys with fewer than `min_presses` training examples are left out (the model can't learn them).
  3. Scores the new model AND the currently active model on the same validation presses: key accuracy, balanced
     accuracy and top-3 at the true press times; optionally also end to end (detector + classifier + dictionary)
     against the typed text.
  4. Saves the model with a manifest naming its training and validation sessions. It is not activated
     automatically.

Validation sessions are only used to compare models; test and locked sessions (data/splits.json) are refused.
"""
import json
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np

from . import config, models, sessions
from .audio import read_wav
from .classifier import MAX_SHIFT_MS, TEST_SHIFTS_MS, KeyNet, crop, normalise, predict_probabilities, slices_around, spectrograms
from .evaluation import key_metrics, per_key, text_error_rates
from .features import band_mel_features
from .groundtruth import true_presses

KINDS = {
    "svm": {"label": "SVM (RBF)", "params": {"C": 10.0}},
    "lda": {"label": "LDA (shrinkage)", "params": {}},
    "cnn": {"label": "CNN", "params": {"epochs": 30}},
}
# CNN training is switched off for now: it takes much longer than the SVM/LDA (minutes to tens of minutes on a
# laptop CPU) and, with the small datasets new recordings give, it hasn't shown an advantage yet. The code below is
# kept; add "cnn" back here (and its button in web/index.html) to re-enable it. The shipped CNN is still available
# for prediction.
ENABLED_KINDS = ("svm", "lda")
CACHE_VERSION = 1


class Cancelled(Exception):
    pass


# ---- data ----

def session_presses(session_id):
    """(slices, codes) for one session's non-repeat keydowns, cached per audio file."""
    audio_path = sessions.audio_path(session_id)
    cache = config.CACHE_DIR / "presses" / f"{session_id}.{audio_path.stat().st_mtime_ns}.v{CACHE_VERSION}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["slices"], data["codes"]
    audio = read_wav(audio_path)
    presses = [p for p in true_presses(sessions.load_session(session_id), audio)[0] if not p["repeat"]]
    slices = slices_around(audio, [p["sample"] for p in presses])
    codes = np.array([p["code"] for p in presses], dtype=str)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, slices=slices, codes=codes)
    return slices, codes


def dataset(session_ids, report=lambda *a: None, check=lambda: None):
    slices, codes = [], []
    for i, session_id in enumerate(session_ids):
        check()
        s, c = session_presses(session_id)
        slices.append(s)
        codes.append(c)
        report(f"{session_id}: {len(c)} presses", (i + 1) / len(session_ids))
    if not slices:
        return np.zeros((0, 4800), np.float32), np.zeros(0, str)
    return np.concatenate(slices), np.concatenate(codes)


def validate_selection(train_ids, validation_ids):
    splits = config.load_splits()
    available = {s["id"] for s in sessions.list_sessions() if s["has_audio"]}
    for session_id in train_ids + validation_ids:
        if session_id not in available:
            raise ValueError(f"{session_id} doesn't exist or has no audio.")
        if session_id in splits["locked"] or session_id in splits["test"]:
            raise ValueError(f"{session_id} is a test/locked session (data/splits.json) and can't be used here.")
    if set(train_ids) & set(validation_ids):
        raise ValueError("A session can't be in both the training and validation sets.")
    if not train_ids:
        raise ValueError("Choose at least one training session.")


# ---- models ----

def train_feature_model(kind, slices, codes, params, report):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    report("Computing band-energy and mel features…", 0.05)
    features = band_mel_features(slices)
    if kind == "svm":
        report(f"Fitting the SVM (C={params['C']}, probability calibration with 5-fold CV)… this step can't be "
               "interrupted and takes longest", 0.15)
        model = SVC(kernel="rbf", C=float(params["C"]), gamma="scale", probability=True, cache_size=1000, random_state=0)
    else:
        report("Fitting shrinkage LDA…", 0.15)
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    estimator = make_pipeline(StandardScaler(), model).fit(features, codes)
    return models.FeatureModel(estimator), estimator


def train_cnn(slices, codes, params, report, check):
    """The shipped CNN's recipe (training/train_cnn.py), with progress and cancellation."""
    import torch
    from torch import nn
    report("Computing log-mel spectrograms…", 0.02)
    spectra = spectrograms(slices)
    keys = np.array(sorted(set(codes)))
    labels = np.searchsorted(keys, codes)
    epochs = int(params["epochs"])
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    network = KeyNet(len(keys))
    optimiser = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(spectra) / 128))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=3e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(epochs):
        network.train()
        order = rng.permutation(len(spectra))
        total = 0.0
        for start in range(0, len(order), 128):
            check()
            batch = order[start:start + 128]
            shift = int(rng.integers(-MAX_SHIFT_MS, MAX_SHIFT_MS + 1))
            loss = loss_fn(network(torch.from_numpy(normalise(crop(spectra[batch], shift)))), torch.from_numpy(labels[batch]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(batch)
        report(f"epoch {epoch + 1}/{epochs}: loss {total / len(order):.3f}", 0.05 + 0.95 * (epoch + 1) / epochs)
    network.eval()

    class Trained:
        key_names = [str(k) for k in keys]

        def classify(self, audio, centers):
            if len(centers) == 0:
                return np.zeros((0, len(keys)))
            return predict_probabilities(network, spectrograms(slices_around(audio, centers)), TEST_SHIFTS_MS)
    return Trained(), network, keys


# ---- evaluation ----

def logged_timing(classify_slices, key_names, slices, codes):
    """Classifier alone at the true press times, on pre-cut slices."""
    probabilities = classify_slices(slices)
    names = np.asarray(key_names)
    predicted = names[probabilities.argmax(axis=1)]
    return {**key_metrics(predicted, codes, probabilities, key_names), "per_key": per_key(predicted, codes)}


def slice_classifier(model):
    """Classify pre-cut slices directly (avoids re-reading audio) for each model kind."""
    if isinstance(model, models.FeatureModel):
        return lambda slices: model.estimator.predict_proba(band_mel_features(slices))
    if hasattr(model, "network"):                       # KeyCNN (shipped or loaded)
        return lambda slices: predict_probabilities(model.network, spectrograms(slices), TEST_SHIFTS_MS)
    return lambda slices: model.classify_slices(slices)


def end_to_end(classifier, session_ids, report, check):
    """Detector + classifier + dictionary decoder on each validation session, pooled like the summary."""
    from .evaluation import match
    from .pipeline import Pipeline
    pipeline = Pipeline(classifier)
    typed_chars = raw_err = dict_err = matched = correct = typed = detected = 0
    for session_id in session_ids:
        check()
        session = sessions.load_session(session_id)
        audio = read_wav(sessions.audio_path(session_id))
        prediction = pipeline.predict(audio)
        presses, _ = true_presses(session, audio)
        truth_ms = np.array([p["sample"] for p in presses]) / config.SAMPLES_PER_MS
        predicted_ms = np.array([p["time_ms"] for p in prediction["presses"]])
        m = match(truth_ms, predicted_ms, config.MATCH_TOLERANCE_MS)
        ok = m >= 0
        predicted_codes = np.array([p["key"] for p in prediction["presses"]])
        truth_codes = np.array([p["code"] for p in presses])
        correct += int((predicted_codes[m[ok]] == truth_codes[ok]).sum()) if ok.any() else 0
        matched += int(ok.sum())
        typed += len(presses)
        detected += len(predicted_ms)
        if session["typed_text"]:
            n = len(session["typed_text"])
            typed_chars += n
            raw_err += text_error_rates(session["typed_text"], prediction["text"])["cer"] * n
            dict_err += text_error_rates(session["typed_text"], prediction["words"]["text"])["cer"] * n
        report(f"end to end: {session_id}", None)
    return {"key_accuracy_matched": correct / max(matched, 1), "recall": matched / max(typed, 1),
            "raw_cer": raw_err / typed_chars if typed_chars else None,
            "dictionary_cer": dict_err / typed_chars if typed_chars else None}


# ---- the job ----

class Job:
    def __init__(self, request):
        self.id = uuid.uuid4().hex[:8]
        self.request = request
        self.state = "running"          # running | done | failed | cancelled
        self.stage = "Starting"
        self.progress = 0.0
        self.log = []
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None
        self.cancel_requested = False

    def report(self, message, progress=None):
        self.log.append(f"[{time.time() - self.started:6.1f}s] {message}")
        self.log = self.log[-400:]
        if progress is not None:
            self.progress = float(progress)

    def check(self):
        if self.cancel_requested:
            raise Cancelled()

    def status(self):
        return {"id": self.id, "state": self.state, "stage": self.stage, "progress": round(self.progress, 3),
                "log": self.log[-80:], "result": self.result, "error": self.error, "request": self.request,
                "elapsed_s": round((self.finished or time.time()) - self.started, 1)}


_job = None
_job_lock = threading.Lock()


def current_job():
    return _job


def start(request, active_classifier):
    """request: {kind, params, train: [...], validation: [...], name, end_to_end}. Raises ValueError if invalid."""
    global _job
    kind = request.get("kind")
    if kind == "cnn" and kind not in ENABLED_KINDS:
        raise ValueError("CNN training is disabled for now; train an SVM or LDA instead.")
    if kind not in ENABLED_KINDS:
        raise ValueError(f"kind must be one of {sorted(ENABLED_KINDS)}")
    request = {"kind": kind, "name": (request.get("name") or "").strip()[:60],
               "params": {**KINDS[kind]["params"], **{k: v for k, v in (request.get("params") or {}).items() if k in KINDS[kind]["params"]}},
               "train": list(dict.fromkeys(request.get("train") or [])),
               "validation": list(dict.fromkeys(request.get("validation") or [])),
               "end_to_end": bool(request.get("end_to_end", True)),
               "min_presses": max(1, int(request.get("min_presses", 5)))}
    if kind == "svm" and not 0.01 <= float(request["params"]["C"]) <= 1000:
        raise ValueError("C must be between 0.01 and 1000.")
    if kind == "cnn" and not 1 <= int(request["params"]["epochs"]) <= 100:
        raise ValueError("epochs must be between 1 and 100.")
    validate_selection(request["train"], request["validation"])
    with _job_lock:
        if _job and _job.state == "running":
            raise ValueError("A training run is already in progress.")
        _job = Job(request)
        threading.Thread(target=_run, args=(_job, active_classifier), daemon=True).start()
    return _job


def cancel():
    if _job and _job.state == "running":
        _job.cancel_requested = True


def _stage(job, name, low, high):
    """Report progress within [low, high] of the overall bar."""
    job.stage = name

    def report(message, fraction=None):
        job.report(message, None if fraction is None else low + (high - low) * fraction)
    return report


def _run(job, active):
    r = job.request
    try:
        slices, codes = dataset(r["train"], _stage(job, "Preparing training data", 0.0, 0.15), job.check)
        counts = {k: int(n) for k, n in zip(*np.unique(codes, return_counts=True))}
        dropped = sorted(k for k, n in counts.items() if n < r["min_presses"])
        keep = ~np.isin(codes, dropped)
        slices, codes = slices[keep], codes[keep]
        if len(set(codes)) < 2:
            raise ValueError("Too few labelled presses to train on.")
        job.report(f"{len(codes)} training presses over {len(set(codes))} keys"
                   + (f"; left out (fewer than {r['min_presses']} examples): {', '.join(dropped)}" if dropped else ""))

        report = _stage(job, f"Training {KINDS[r['kind']]['label']}", 0.15, 0.75)
        fit_started = time.time()
        network = estimator = keys = None
        if r["kind"] == "cnn":
            model, network, keys = train_cnn(slices, codes, r["params"], report, job.check)
            classify_slices = lambda s: predict_probabilities(network, spectrograms(s), TEST_SHIFTS_MS)
        else:
            model, estimator = train_feature_model(r["kind"], slices, codes, r["params"], report)
            job.check()
            classify_slices = slice_classifier(model)
        fit_seconds = time.time() - fit_started
        job.report(f"trained in {fit_seconds:.0f} s", None)

        evaluation = {"candidate": None, "active": None, "presses": 0}
        if r["validation"]:
            report = _stage(job, "Scoring on validation sessions", 0.75, 0.85)
            v_slices, v_codes = dataset(r["validation"], report, job.check)
            evaluation["presses"] = int(len(v_codes))
            evaluation["candidate"] = logged_timing(classify_slices, model.key_names, v_slices, v_codes)
            evaluation["active"] = {**logged_timing(slice_classifier(active), active.key_names, v_slices, v_codes),
                                    "model": {"id": active.id, "name": active.name}}
            overlap = sorted(set(r["validation"]) & active.trained_on)
            if overlap:
                evaluation["active"]["warning"] = (f"The active model was trained on {len(overlap)} of these validation "
                                                   "sessions, so its scores here are optimistic.")
            job.report(f"validation key accuracy: new {evaluation['candidate']['accuracy']:.1%}, "
                       f"active {evaluation['active']['accuracy']:.1%}", None)
            if r["end_to_end"]:
                report = _stage(job, "Scoring end to end (detector + classifier + dictionary)", 0.85, 0.99)
                candidate_view = model
                candidate_view.version = f"candidate-{job.id}"
                evaluation["candidate"]["end_to_end"] = end_to_end(candidate_view, r["validation"], report, job.check)
                evaluation["active"]["end_to_end"] = end_to_end(active, r["validation"], report, job.check)

        job.check()
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + r["kind"]
        manifest = {
            "id": model_id, "kind": r["kind"],
            "name": r["name"] or f"{KINDS[r['kind']]['label']} · {len(r['train'])} sessions",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_on": r["train"], "validation_sessions": r["validation"],
            "params": r["params"], "training_presses": int(len(codes)), "keys": sorted(set(codes.tolist())),
            "left_out_keys": dropped, "fit_seconds": round(fit_seconds, 1),
            "validation": _summary(evaluation["candidate"]),
            "compared_with": {"model": evaluation["active"]["model"], **_summary(evaluation["active"])} if evaluation["active"] else None,
        }
        models.save(manifest, r["kind"], network=network, key_names=keys, estimator=estimator)
        job.result = {"model": manifest, "evaluation": evaluation}
        job.state, job.stage, job.progress = "done", "Finished", 1.0
        job.report(f"saved model {model_id}")
    except Cancelled:
        job.state, job.stage = "cancelled", "Cancelled"
        job.report("cancelled")
    except Exception as error:      # reported to the page; the server keeps running
        job.state, job.stage, job.error = "failed", "Failed", str(error)
        job.report("failed: " + "".join(traceback.format_exception_only(error)).strip())
    finally:
        job.finished = time.time()


def _summary(evaluation):
    if not evaluation:
        return None
    return {k: evaluation.get(k) for k in ("count", "accuracy", "balanced", "top3", "end_to_end")}
