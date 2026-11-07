#!/usr/bin/env python3
"""
SneaKey classifier experiments.

Protocol (fixed before running anything):
  * TEST sessions are locked away: no experiment, feature choice, model choice or hyper-parameter ever sees them.
    They are scored exactly once, by `final`, on the single configuration chosen from the development results.
  * DEV sessions: every comparison is leave-one-session-out (LOSO) over the dev sessions.
  * Label set: keys with >= 20 clean 30 ms press clips in the DEV sessions (test sessions do not influence it).
  * Metrics on held-out sessions: balanced accuracy + top-3 on clean clips (primary), plus accuracy on all
    confidently aligned presses and on every press of a known key (what live typing actually looks like).

    .venv/bin/python analysis/experiments.py cache        # wide int16 slices around every press anchor
    .venv/bin/python analysis/experiments.py features     # stage A: feature sets (fixed model: shrinkage LDA)
    .venv/bin/python analysis/experiments.py models       # stage B: model families on the best feature set
    .venv/bin/python analysis/experiments.py data         # stage C: train on clean vs confident vs all
    .venv/bin/python analysis/experiments.py cnn [cfg]    # stage D: CNN on log-mel with augmentation
    .venv/bin/python analysis/experiments.py final        # train chosen config on all dev, score locked test ONCE
"""
import csv, glob, json, os, sys, time, wave
from collections import Counter
import numpy as np
from scipy import fft as sfft, signal

OUT = "analysis_output/experiments"
TEST_SESSIONS = ["session_20261107_184609", "session_20261107_184948", "session_20261107_185605"]   # locked
SR = 48000
PRE_CACHE_MS, POST_CACHE_MS = 30, 80
C0 = PRE_CACHE_MS * SR // 1000               # index of the anchor (onset) inside each cached slice
SEED = 0
ms = lambda v: int(round(v * SR / 1000))


# ====================================================================================================== cache
def build_cache():
    os.makedirs(OUT, exist_ok=True)
    X, meta = [], []
    for sdir in sorted(glob.glob("datasets/session_*")):
        if not os.path.isdir(sdir): continue
        s = os.path.basename(sdir)
        with wave.open(f"recordings/{s}/audio.wav") as w:
            assert w.getframerate() == SR
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        for r in csv.DictReader(open(f"{sdir}/manifest.csv", newline="")):
            if r["dataset_class"] != "press" or r["window_ms"] != "30" or r["anchor_sample"] == "": continue
            a = int(r["anchor_sample"]); lo, hi = a - ms(PRE_CACHE_MS), a + ms(POST_CACHE_MS)
            if lo < 0 or hi > len(pcm): continue
            X.append(pcm[lo:hi].copy())
            meta.append(dict(session=s, event_index=int(r["event_index"]), label=r["label"], key=r["key"], clean30=r["clean"] == "1",
                             alignment=r["alignment"], category=r["category"], anchor=a, exclusion=r["exclusion_reasons"]))
    # clean status at 50/80 ms windows too
    m50 = {}
    for sdir in sorted(glob.glob("datasets/session_*")):
        if not os.path.isdir(sdir): continue
        s = os.path.basename(sdir)
        for r in csv.DictReader(open(f"{sdir}/manifest.csv", newline="")):
            if r["dataset_class"] == "press" and r["window_ms"] in ("50", "80"): m50[(s, int(r["event_index"]), r["window_ms"])] = r["clean"] == "1"
    for m in meta:
        m["clean50"] = m50.get((m["session"], m["event_index"], "50"), False); m["clean80"] = m50.get((m["session"], m["event_index"], "80"), False)
    np.save(f"{OUT}/cache_x.npy", np.stack(X))
    json.dump(meta, open(f"{OUT}/cache_meta.json", "w"))
    print(f"cached {len(X)} press slices ({PRE_CACHE_MS} ms pre .. {POST_CACHE_MS} ms post) from {len(set(m['session'] for m in meta))} sessions")


def load_cache():
    X = np.load(f"{OUT}/cache_x.npy").astype(np.float32) / 32768.0
    meta = json.load(open(f"{OUT}/cache_meta.json"))
    return X, meta


def split(meta):
    sess = np.array([m["session"] for m in meta])
    dev = ~np.isin(sess, TEST_SESSIONS)
    y = np.array([m["label"] for m in meta])
    clean = np.array([m["clean30"] for m in meta])
    keys = sorted(k for k, n in Counter(y[dev & clean]).items() if n >= 20)
    inset = np.isin(y, keys)
    conf = np.array([m["alignment"] in ("detected", "detected_resolved") for m in meta])
    return dict(sess=sess, dev=dev, y=y, clean=clean, conf=conf, inset=inset, keys=keys,
                dev_sessions=sorted(set(sess[dev])), test_sessions=sorted(set(sess[~dev])))


# ====================================================================================================== features (batched)
def window_slice(X, pre_ms, post_ms, shift=0):
    return X[:, C0 - ms(pre_ms) + shift: C0 + ms(post_ms) + shift]


def mel_fb(n_fft, n_mels, fmin=300.0, fmax=20000.0):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700.0); mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(min(fmax, SR / 2)), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


def logmel(seg, n_fft=256, hop=64, n_mels=32):
    """(N, L) -> (N, n_mels, frames) natural-log mel power."""
    fr = np.lib.stride_tricks.sliding_window_view(seg, n_fft, axis=1)[:, ::hop, :]
    P = np.abs(np.fft.rfft(fr * np.hanning(n_fft), axis=2)) ** 2
    return np.log(np.maximum(np.einsum("ntf,mf->nmt", P, mel_fb(n_fft, n_mels)), 1e-12))


def pool_t(S, t_pool):
    e = np.linspace(0, S.shape[2], t_pool + 1).astype(int)
    return np.concatenate([S[:, :, a:max(b, a + 1)].mean(2) for a, b in zip(e[:-1], e[1:])], axis=1)


def bands(seg, nb=32, fmin=500.0, fmax=20000.0, nfft=4096):
    L = seg.shape[1]
    w = seg * signal.windows.tukey(L, alpha=min(1.0, 2 * 0.001 * SR / L))
    P = np.abs(np.fft.rfft(w, nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / SR); e = np.geomspace(fmin, min(fmax, SR / 2 * 0.98), nb + 1)
    W = np.stack([((f >= lo) & (f < hi)) / max(1, ((f >= lo) & (f < hi)).sum()) for lo, hi in zip(e[:-1], e[1:])])
    return 10 * np.log10(P @ W.T + 1e-20)


def feat(name, X, shift=0):
    """Named feature sets. All on the cached slices; window defaults to 5 ms pre / 25 ms post (30 ms)."""
    spec = FEATS[name]
    seg = window_slice(X, spec.get("pre", 5), spec.get("post", 25), shift)
    parts = []
    for p in spec["parts"]:
        kind = p[0]
        if kind == "band": parts.append(bands(seg, p[1]))
        elif kind == "mel":       # ("mel", n_mels, t_pool, n_fft, hop)
            parts.append(pool_t(logmel(seg, p[3], p[4], p[1]), p[2]))
        elif kind == "mfcc":      # ("mfcc", n_mels, n_coef, t_pool, n_fft, hop, deltas)
            S = logmel(seg, p[4], p[5], p[1]); C = sfft.dct(S, type=2, norm="ortho", axis=1)[:, :p[2], :]
            parts.append(pool_t(C, p[3]))
            if p[6]: parts.append(pool_t(np.diff(C, axis=2), max(1, p[3] - 1)))
    F = np.concatenate(parts, axis=1)
    if spec.get("shape_only"):   # remove per-clip overall level from the log-spectral parts
        F = F - F.mean(1, keepdims=True)
    return F.astype(np.float64)


FEATS = {
    "base224": {"parts": [("band", 32), ("mel", 32, 6, 256, 64)]},
    "base224_w20": {"pre": 5, "post": 15, "parts": [("band", 32), ("mel", 32, 6, 256, 64)]},
    "base224_w40": {"pre": 5, "post": 35, "parts": [("band", 32), ("mel", 32, 6, 256, 64)]},
    "base224_w50": {"pre": 5, "post": 45, "parts": [("band", 32), ("mel", 32, 6, 256, 64)]},
    "band64": {"parts": [("band", 64)]},
    "mel40x10": {"parts": [("mel", 40, 10, 256, 48)]},
    "mel64x10": {"parts": [("mel", 64, 10, 256, 48)]},
    "mel40x15_512": {"parts": [("mel", 40, 15, 512, 48)]},
    "band64+mel40x10": {"parts": [("band", 64), ("mel", 40, 10, 256, 48)]},
    "mfcc20x10+d": {"parts": [("mfcc", 40, 20, 10, 256, 48, True)]},
    "band64+mfcc20x10+d": {"parts": [("band", 64), ("mfcc", 40, 20, 10, 256, 48, True)]},
    "band64+mel40x10_w40": {"pre": 5, "post": 35, "parts": [("band", 64), ("mel", 40, 10, 256, 48)]},
    "band64+mel40x10_shape": {"parts": [("band", 64), ("mel", 40, 10, 256, 48)], "shape_only": True},
}


# ====================================================================================================== evaluation
def topk(P, y, classes, k):
    idx = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean([y[i] in classes[idx[i]] for i in range(len(y))]))


def evaluate(pred, P, classes, y, mask_eval_sets):
    from sklearn.metrics import balanced_accuracy_score
    out = {}
    for name, m in mask_eval_sets.items():
        if m.sum() == 0: continue
        out[name] = {"n": int(m.sum()), "acc": float(np.mean(pred[m] == y[m])), "bal_acc": float(balanced_accuracy_score(y[m], pred[m])),
                     "top3": topk(P[m], y[m], classes, 3)}
    return out


def loso(make_model, F, S, train_mask, eval_masks, sessions, fit_kwargs=None):
    """Train on train_mask rows of the other sessions, predict every row of the held-out session. Returns metrics on eval_masks."""
    from sklearn.base import clone
    y = S["y"]; n = len(y)
    keys = np.array(S["keys"])
    pred = np.empty(n, dtype=object); P = np.zeros((n, len(keys)))
    done = np.zeros(n, bool)
    for s in sessions:
        te = (S["sess"] == s) & S["inset"]
        tr = (S["sess"] != s) & train_mask & S["dev"] & S["inset"]
        m = make_model().fit(F[tr], y[tr])
        pr = m.predict_proba(F[te]) if hasattr(m, "predict_proba") else softmax(m.decision_function(F[te]))
        cls = list(m.classes_)
        Pfull = np.zeros((te.sum(), len(keys)))
        for j, k in enumerate(keys):
            if k in cls: Pfull[:, j] = pr[:, cls.index(k)]
        P[te] = Pfull; pred[te] = keys[Pfull.argmax(1)]; done |= te
    masks = {k: v & done for k, v in eval_masks.items()}
    return evaluate(pred, P, keys, y, masks), P, pred


def softmax(D):
    D = D - D.max(1, keepdims=True); E = np.exp(D); return E / E.sum(1, keepdims=True)


def eval_masks(S, sessions):
    ins = np.isin(S["sess"], sessions) & S["inset"]
    return {"clean": ins & S["clean"], "confident": ins & S["conf"], "all": ins}


def log(stage, name, res, extra=None):
    os.makedirs(OUT, exist_ok=True)
    rec = {"stage": stage, "name": name, "time": time.strftime("%H:%M:%S"), **(extra or {}), "results": res}
    with open(f"{OUT}/results.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    c, a = res.get("clean", {}), res.get("all", {})
    print(f"[{stage}] {name:32s} clean bal {c.get('bal_acc', float('nan')):.3f} acc {c.get('acc', float('nan')):.3f} top3 {c.get('top3', float('nan')):.3f}"
          f" | conf acc {res.get('confident', {}).get('acc', float('nan')):.3f} | all acc {a.get('acc', float('nan')):.3f} top3 {a.get('top3', float('nan')):.3f}", flush=True)


# ====================================================================================================== classical models
def model_zoo():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    return {
        "lda_shrink": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "logreg_C0.1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
        "logreg_C1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")),
        "svm_rbf_C10": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        "svm_rbf_C100": lambda: make_pipeline(StandardScaler(), SVC(C=100, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        "lda_then_svm": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=None), SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        "pca_whiten_svm": lambda: make_pipeline(StandardScaler(), PCA(n_components=0.99, whiten=True, random_state=SEED), SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        "hist_gb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEED),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED),
        "knn_on_lda": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"), KNeighborsClassifier(n_neighbors=15, weights="distance")),
    }


def stage_features():
    X, meta = load_cache(); S = split(meta)
    print(f"{len(S['keys'])} keys; dev sessions {len(S['dev_sessions'])}; clean dev clips in label set {int((S['dev'] & S['clean'] & S['inset']).sum())}")
    make = model_zoo()["lda_shrink"]
    for name in FEATS:
        F = feat(name, X)
        res, _, _ = loso(make, F, S, S["clean"], eval_masks(S, S["dev_sessions"]), S["dev_sessions"])
        log("A_features", name, res, {"dims": F.shape[1], "model": "lda_shrink"})


def stage_models(feature_name):
    X, meta = load_cache(); S = split(meta); F = feat(feature_name, X)
    for name, make in model_zoo().items():
        t = time.time()
        res, _, _ = loso(make, F, S, S["clean"], eval_masks(S, S["dev_sessions"]), S["dev_sessions"])
        log("B_models", name, res, {"features": feature_name, "seconds": round(time.time() - t, 1)})


def stage_data(feature_name, model_name):
    X, meta = load_cache(); S = split(meta); F = feat(feature_name, X); make = model_zoo()[model_name]
    for tname, tm in (("train_clean", S["clean"]), ("train_confident", S["conf"]), ("train_all", np.ones(len(S["y"]), bool))):
        res, _, _ = loso(make, F, S, tm, eval_masks(S, S["dev_sessions"]), S["dev_sessions"])
        log("C_data", tname, res, {"features": feature_name, "model": model_name})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cache"
    if cmd == "cache": build_cache()
    elif cmd == "features": stage_features()
    elif cmd == "models": stage_models(sys.argv[2])
    elif cmd == "data": stage_data(sys.argv[2], sys.argv[3])
    else: sys.exit(f"unknown command {cmd}")
