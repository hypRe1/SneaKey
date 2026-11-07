#!/usr/bin/env python3
"""
SneaKey baseline keystroke classifier (classical ML, no neural network).

    .venv/bin/python analysis/classify_baseline.py            # datasets/session_*/ + recordings/session_*/ -> analysis_output/classifier_baseline/

Data: clean 30 ms PRESS clips (manifest clean==1, window 30) from every session in datasets/, label = event.code.
Only keys with >= --min-per-key clean clips in total AND present in every session are classified.

Evaluation (all hyper-parameters fixed in advance; nothing is tuned on test data):
  * leave-one-session-out (LOSO): train on all other sessions, test on the held-out one  <- the honest number
  * pooled random stratified 5-fold: what a careless split would report (optimistic reference)
  * controls: identical pipeline on the 20 ms of audio BEFORE each keystroke, and with shuffled training labels
  * learning curve: LOSO with the training set capped at k clips per key
  * coarse task: Space vs Backspace vs everything else
  * confusion geometry: are mistakes physically close keys?
Optional per-session feature normalisation uses only unlabelled audio of that session (no test labels).
"""
import argparse, csv, glob, json, os, sys, wave
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_keystroke_segments import features as band_features   # identical to the separability analysis

SEED = 0
# UK/ANSI letter-block geometry in key widths (row stagger 0, 0.25, 0.75), used only to measure confusion distances
ROWS = {0: ("qwertyuiop", 0.0), 1: ("asdfghjkl", 0.25), 2: ("zxcvbnm,.", 0.75)}
POS = {f"Key{c.upper()}" if c.isalpha() else {",": "Comma", ".": "Period"}[c]: (x + i, float(r)) for r, (s, x) in ROWS.items() for i, c in enumerate(s)}


# ====================================================================================================== data
def read_wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0, w.getframerate()


def mel_filterbank(sr, n_fft, n_mels=32, fmin=300.0, fmax=20000.0):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700.0); mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(min(fmax, sr / 2)), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


def logmel_tf(seg, sr, n_fft=256, hop=64, n_mels=32, t_pool=6):
    """log-mel spectrogram of the clip, pooled into t_pool time slices (keeps the click's time structure)."""
    fb = mel_filterbank(sr, n_fft, n_mels)
    win = np.hanning(n_fft)
    frames = [seg[i:i + n_fft] * win for i in range(0, len(seg) - n_fft + 1, hop)]
    S = np.log(np.maximum(fb @ (np.abs(np.fft.rfft(frames, axis=1)) ** 2).T, 1e-12))   # (mels, frames)
    edges = np.linspace(0, S.shape[1], t_pool + 1).astype(int)
    return np.concatenate([S[:, a:max(b, a + 1)].mean(1) for a, b in zip(edges[:-1], edges[1:])])


def load(datasets_root, recordings_root, window, min_per_key):
    rows, src = [], {}
    for man in sorted(glob.glob(os.path.join(datasets_root, "session_*", "manifest.csv"))):
        sdir = os.path.dirname(man); sess = os.path.basename(sdir)
        for r in csv.DictReader(open(man, newline="")):
            if r["dataset_class"] == "press" and r["window_ms"] == str(window) and r["clean"] == "1":
                r["_session"] = sess; r["_clip"] = os.path.join(sdir, r["clip_path"]); rows.append(r)
        src[sess] = read_wav(os.path.join(recordings_root, sess, "audio.wav"))
    sessions = sorted(src)
    tot = Counter(r["label"] for r in rows)
    per = defaultdict(Counter)
    for r in rows: per[r["label"]][r["_session"]] += 1
    keys = sorted(k for k, n in tot.items() if n >= min_per_key and all(per[k][s] > 0 for s in sessions))
    rows = [r for r in rows if r["label"] in keys]
    return rows, src, sessions, keys, tot


def featurise(rows, src):
    clipF, preF = [], []
    for r in rows:
        x, sr = read_wav(r["_clip"])
        clipF.append(np.concatenate([band_features(x, sr), logmel_tf(x, sr)]))
        a = int(r["anchor_sample"]); full, srs = src[r["_session"]]
        pre = full[a - int(0.025 * srs): a - int(0.005 * srs)]            # 20 ms that END 5 ms before the onset
        preF.append(np.concatenate([band_features(pre, srs), logmel_tf(np.concatenate([pre, pre[:max(0, 1440 - len(pre))]]), srs)]))
    return np.array(clipF), np.array(preF)


def session_normalise(X, sess):
    """Per-session z-scoring using every clip of that session, labels never used (unsupervised adaptation)."""
    Xn = X.copy()
    for s in np.unique(sess):
        m = sess == s; Xn[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    return Xn


# ====================================================================================================== models/eval
def models():
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")),
        "lda_shrink": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
        "knn5": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def scores(model, X):
    if hasattr(model, "predict_proba") and not isinstance(model[-1], SVC):
        return model.predict_proba(X)
    return model.decision_function(X)


def topk(S, y, classes, k):
    idx = np.argsort(-S, axis=1)[:, :k]
    return float(np.mean([y[i] in classes[idx[i]] for i in range(len(y))]))


def run_loso(make, X, y, sess, sessions, shuffle_train=False, cap=None, rng=None):
    pred = np.empty(len(y), dtype=object); per = {}
    top3 = []
    for s in sessions:
        te = sess == s; tr = ~te
        tri = np.where(tr)[0]
        if cap is not None:
            keep = []
            for k in np.unique(y[tri]):
                ids = tri[y[tri] == k]; keep.extend(rng.choice(ids, min(cap, len(ids)), replace=False))
            tri = np.array(sorted(keep))
        ytr = y[tri].copy()
        if shuffle_train: ytr = rng.permutation(ytr)
        m = make().fit(X[tri], ytr)
        pred[te] = m.predict(X[te])
        S = scores(m, X[te])
        top3.append(topk(S, y[te], m.classes_, 3) * te.sum())
        per[s] = {"n": int(te.sum()), "acc": float(np.mean(pred[te] == y[te])), "bal_acc": float(balanced_accuracy_score(y[te], pred[te]))}
    return {"acc": float(np.mean(pred == y)), "bal_acc": float(balanced_accuracy_score(y, pred)), "top3": float(sum(top3) / len(y)),
            "per_session": per, "pred": pred}


def run_random_cv(make, X, y, rng):
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return {"acc": float(np.mean(pred == y)), "bal_acc": float(balanced_accuracy_score(y, pred))}


def strip(r): return {k: v for k, v in r.items() if k != "pred"}


# ====================================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="datasets"); ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--out", default="analysis_output/classifier_baseline")
    ap.add_argument("--window", type=int, default=30); ap.add_argument("--min-per-key", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, src, sessions, keys, tot = load(a.datasets, a.recordings, a.window, a.min_per_key)
    y = np.array([r["label"] for r in rows]); sess = np.array([r["_session"] for r in rows])
    print(f"{len(rows)} clean {a.window} ms press clips, {len(keys)} keys, {len(sessions)} sessions")
    X, Xpre = featurise(rows, src)
    Xn, Xpre_n = session_normalise(X, sess), session_normalise(Xpre, sess)
    cnt = Counter(y)
    R = {"data": {"clips": len(rows), "keys": keys, "sessions": sessions, "window_ms": a.window, "min_per_key": a.min_per_key,
                  "per_key": dict(cnt.most_common()), "per_session": dict(Counter(sess)),
                  "excluded_keys_too_few_or_not_in_every_session": {k: n for k, n in tot.items() if k not in keys},
                  "features": f"{X.shape[1]} dims: 32 log-band spectrum (0.5-20 kHz) + 32-mel x 6-time-slice log-mel"},
         "chance": {"balanced_accuracy_uniform": 1 / len(keys), "accuracy_majority_class": cnt.most_common(1)[0][1] / len(y),
                    "top3_uniform": 3 / len(keys)}}

    # ---- main comparison
    R["loso"], R["random_cv"] = {}, {}
    best, best_score = None, -1
    for name, make in models().items():
        for norm, XX in (("global", X), ("per_session", Xn)):
            r = run_loso(make, XX, y, sess, sessions)
            R["loso"][f"{name}/{norm}"] = strip(r)
            print(f"LOSO {name:10s} {norm:11s} acc {r['acc']:.3f} bal {r['bal_acc']:.3f} top3 {r['top3']:.3f}  per-session bal " +
                  " ".join(f"{v['bal_acc']:.2f}" for v in r["per_session"].values()))
            if r["bal_acc"] > best_score: best, best_score, best_pred = (name, norm), r["bal_acc"], r["pred"]
        rc = run_random_cv(make, X, y, rng)
        R["random_cv"][name] = rc
        print(f"random 5-fold {name:10s} acc {rc['acc']:.3f} bal {rc['bal_acc']:.3f}")
    R["best"] = {"model": best[0], "normalisation": best[1], "bal_acc": best_score,
                 "note": "best of 8 fixed configurations on LOSO; picking the max of 8 is itself mildly optimistic"}
    make = models()[best[0]]; XB = Xn if best[1] == "per_session" else X; XPB = Xpre_n if best[1] == "per_session" else Xpre

    # ---- controls
    pre = run_loso(make, XPB, y, sess, sessions)
    shuf = [run_loso(make, XB, y, sess, sessions, shuffle_train=True, rng=rng)["bal_acc"] for _ in range(5)]
    R["controls"] = {"pre_onset_noise_20ms": strip(pre), "shuffled_training_labels_bal_acc": {"mean": float(np.mean(shuf)), "runs": shuf}}
    print(f"control: pre-onset noise bal {pre['bal_acc']:.3f}, shuffled labels bal {np.mean(shuf):.3f} (chance {1/len(keys):.3f})")

    # ---- per key + confusion
    labels = sorted(keys, key=lambda k: -cnt[k])
    cm = confusion_matrix(y, best_pred, labels=labels)
    rec = {k: float(cm[i, i] / cm[i].sum()) for i, k in enumerate(labels)}
    R["per_key_recall_best"] = rec
    conf_pairs = Counter((t, p) for t, p in zip(y, best_pred) if t != p)
    R["top_confusions"] = [f"{t} -> {p}: {n}" for (t, p), n in conf_pairs.most_common(15)]

    # ---- confusion geometry (letter-block keys only)
    errs = [(t, p) for t, p in zip(y, best_pred) if t != p and t in POS and p in POS]
    if errs:
        d = lambda a_, b_: float(np.hypot(POS[a_][0] - POS[b_][0], POS[a_][1] - POS[b_][1]))
        obs = np.mean([d(t, p) for t, p in errs])
        pk = [k for k in keys if k in POS]; prior = np.array([cnt[k] for k in pk], float)
        null = []
        for t, _ in errs:
            others = [k for k in pk if k != t]; w = np.array([cnt[k] for k in others], float)
            null.append(np.dot([d(t, o) for o in others], w / w.sum()))
        R["confusion_geometry"] = {"errors_between_letter_block_keys": len(errs), "mean_distance_of_errors_keys": obs,
                                   "expected_if_errors_ignored_position": float(np.mean(null)),
                                   "share_of_errors_to_adjacent_keys(<=1.3)": float(np.mean([d(t, p) <= 1.3 for t, p in errs]))}

    # ---- same-finger confusions (standard touch-typing assignment; assumes the typist uses it)
    fingers = {}
    for f, ks in {"L5": "qaz", "L4": "wsx", "L3": "edc", "L2": "rfvtgb", "R2": "yhnujm", "R3": "ik,", "R4": "ol.", "R5": "p"}.items():
        for c in ks: fingers[f"Key{c.upper()}" if c.isalpha() else {",": "Comma", ".": "Period"}[c]] = f
    ferr = [(t, p) for t, p in zip(y, best_pred) if t != p and t in fingers and p in fingers]
    if ferr:
        fk = [k for k in keys if k in fingers]
        sims = []
        for _ in range(2000):
            hits = 0
            for t, _ in ferr:
                o = [k for k in fk if k != t]; w = np.array([cnt[k] for k in o], float)
                hits += fingers[o[rng.choice(len(o), p=w / w.sum())]] == fingers[t]
            sims.append(hits / len(ferr))
        obs = float(np.mean([fingers[t] == fingers[p] for t, p in ferr]))
        R["same_finger_confusions"] = {"errors": len(ferr), "observed_share": obs, "null_share": float(np.mean(sims)),
                                       "p_value": float((np.sum(np.array(sims) >= obs) + 1) / (len(sims) + 1)),
                                       "assumption": "standard touch-typing finger assignment",
                                       "meaning": "errors over-represented between keys struck by the same finger suggest the model uses typist-specific strike dynamics"}
        print(f"same-finger errors {obs:.3f} vs null {np.mean(sims):.3f}")

    # ---- coarse task
    yc = np.array(["Space" if k == "Space" else "Backspace" if k == "Backspace" else "other" for k in y])
    rcoarse = run_loso(make, XB, yc, sess, sessions)
    R["coarse_space_backspace_other"] = {k: v for k, v in strip(rcoarse).items()}
    print(f"coarse 3-way: bal {rcoarse['bal_acc']:.3f} (chance 0.333)")

    # ---- learning curve
    caps = [5, 10, 20, 40, 80, 10**6]
    lc = {}
    for c in caps:
        v = [run_loso(make, XB, y, sess, sessions, cap=c, rng=rng)["bal_acc"] for _ in range(3 if c < 10**6 else 1)]
        lc["all" if c == 10**6 else str(c)] = {"mean": float(np.mean(v)), "sd": float(np.std(v))}
        print(f"learning curve cap {c}: bal {np.mean(v):.3f}")
    R["learning_curve_train_cap_per_key"] = lc

    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    plots(R, cm, labels, rec, cnt, a.out)
    report(R, a.out)
    print(f"\nwrote {a.out}/report.md")


def plots(R, cm, labels, rec, cnt, out):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([l.replace("Key", "") for l in labels], rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([f"{l.replace('Key', '')} ({cnt[l]})" for l in labels])
    ax.set_xlabel("predicted"); ax.set_ylabel("true (n clean clips)")
    b = R["best"]; ax.set_title(f"Leave-one-session-out confusion (row-normalised), {b['model']}/{b['normalisation']}, balanced acc {b['bal_acc']:.2f}")
    plt.colorbar(im, fraction=0.04); plt.tight_layout(); plt.savefig(os.path.join(out, "confusion_loso.png"), dpi=120); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    lc = R["learning_curve_train_cap_per_key"]; xs = list(lc)
    axs[0].errorbar(range(len(xs)), [lc[k]["mean"] for k in xs], yerr=[lc[k]["sd"] for k in xs], marker="o")
    axs[0].axhline(R["chance"]["balanced_accuracy_uniform"], color="grey", ls="--", label="chance")
    axs[0].set_xticks(range(len(xs))); axs[0].set_xticklabels(xs); axs[0].set_xlabel("max training clips per key (from the other sessions)")
    axs[0].set_ylabel("LOSO balanced accuracy"); axs[0].set_title("Learning curve"); axs[0].legend(); axs[0].grid(alpha=0.3)
    names = list(R["loso"])
    axs[1].barh(range(len(names)), [R["loso"][n]["bal_acc"] for n in names], color="#2980b9", label="LOSO")
    axs[1].scatter([R["random_cv"][n.split("/")[0]]["bal_acc"] for n in names], range(len(names)), color="#c0392b", zorder=3, label="random 5-fold (optimistic)")
    axs[1].axvline(R["chance"]["balanced_accuracy_uniform"], color="grey", ls="--")
    axs[1].axvline(R["controls"]["pre_onset_noise_20ms"]["bal_acc"], color="#e67e22", ls=":", label="pre-onset noise control")
    axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels(names, fontsize=8); axs[1].set_xlabel("balanced accuracy"); axs[1].legend(fontsize=8)
    axs[1].set_title("Models: honest (LOSO) vs random split")
    plt.tight_layout(); plt.savefig(os.path.join(out, "learning_curve_and_models.png"), dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(labels)), [rec[k] for k in labels], color="#27ae60")
    ax.axhline(1 / len(labels), color="grey", ls="--", label="chance")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([f"{l.replace('Key', '')}\n{cnt[l]}" for l in labels], fontsize=8)
    ax.set_ylabel("recall (LOSO)"); ax.set_title("Per-key recall, held-out sessions (x labels: key and number of clean clips)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "per_key_recall.png"), dpi=120); plt.close()


def report(R, out):
    D, C, B = R["data"], R["chance"], R["best"]
    L = ["# SneaKey baseline classifier", "",
         f"{D['clips']} clean {D['window_ms']} ms press clips, {len(D['keys'])} keys (>= {D['min_per_key']} clips and present in all {len(D['sessions'])} sessions), features: {D['features']}.",
         f"Chance: balanced accuracy {C['balanced_accuracy_uniform']:.3f}; always-Space accuracy {C['accuracy_majority_class']:.3f}; top-3 {C['top3_uniform']:.3f}.", "",
         "## Held-out-session results (LOSO)", "", "| model / normalisation | accuracy | balanced acc | top-3 | random 5-fold balanced (optimistic) |", "|---|---|---|---|---|"]
    for n, r in R["loso"].items():
        L.append(f"| {n} | {r['acc']:.3f} | {r['bal_acc']:.3f} | {r['top3']:.3f} | {R['random_cv'][n.split('/')[0]]['bal_acc']:.3f} |")
    br = R["loso"][f"{B['model']}/{B['normalisation']}"]
    L += ["", f"Best: **{B['model']} / {B['normalisation']}**, balanced accuracy **{B['bal_acc']:.3f}** ({B['note']}).", "",
          "Per held-out session: " + ", ".join(f"{s[-6:]}: {v['bal_acc']:.2f} (n={v['n']})" for s, v in br["per_session"].items()), "",
          "## Controls", "",
          f"* Same model on the 20 ms *before* each keystroke: balanced accuracy **{R['controls']['pre_onset_noise_20ms']['bal_acc']:.3f}** (chance {C['balanced_accuracy_uniform']:.3f}). Above chance means context/neighbour sounds leak label information.",
          f"* Shuffled training labels: {R['controls']['shuffled_training_labels_bal_acc']['mean']:.3f}.",
          f"* Coarse Space / Backspace / other: balanced accuracy {R['coarse_space_backspace_other']['bal_acc']:.3f} (chance 0.333).", "",
          "## Learning curve (LOSO, training capped at k clips per key)", "", "| k | balanced acc |", "|---|---|"]
    L += [f"| {k} | {v['mean']:.3f} +/- {v['sd']:.3f} |" for k, v in R["learning_curve_train_cap_per_key"].items()]
    L += ["", "## Per-key recall", "", ", ".join(f"{k.replace('Key', '')}: {v:.2f}" for k, v in R["per_key_recall_best"].items()), "",
          "Top confusions: " + "; ".join(R["top_confusions"]), ""]
    if "confusion_geometry" in R:
        g = R["confusion_geometry"]
        L += [f"Confusion geometry: mean physical distance of letter-block errors {g['mean_distance_of_errors_keys']:.2f} key widths vs {g['expected_if_errors_ignored_position']:.2f} if errors ignored position; {100*g['share_of_errors_to_adjacent_keys(<=1.3)']:.0f}% of errors go to an adjacent key.", ""]
    if "same_finger_confusions" in R:
        s = R["same_finger_confusions"]
        L += [f"Same-finger confusions: {100*s['observed_share']:.1f}% of letter errors go to a key struck by the same finger vs {100*s['null_share']:.1f}% expected (p={s['p_value']:.4f}; {s['assumption']}). "
              "This suggests part of the signal is how the typist strikes keys, which is unlikely to transfer to another typist.", ""]
    L += ["## Caveats", "",
          "* All sessions are the same laptop, position and evening: this measures cross-session generalisation only (level 1), not other conditions or devices.",
          "* Only clips that passed the clean/overlap filter are classified; real typing includes the excluded, overlapping keystrokes.",
          "* Keys with too few clean clips (rare letters, digits, punctuation) are not in the label set at all.",
          "* Hyper-parameters were fixed, not tuned; the 'best' configuration is the max of 8, which is mildly optimistic.",
          "", "![confusion](confusion_loso.png)", "![curve](learning_curve_and_models.png)", "![recall](per_key_recall.png)"]
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
