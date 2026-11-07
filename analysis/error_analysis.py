#!/usr/bin/env python3
"""
Error analysis of the current best model (shrinkage LDA on band64 + MFCC20x10 + deltas, trained on clean 30 ms
press clips), on the 9 DEVELOPMENT sessions only. The 3 locked test sessions are removed before any feature is
computed, and every fold asserts that neither they nor the held-out session reach training.

    .venv/bin/python analysis/error_analysis.py      # -> analysis_output/error_analysis/

Fitting performed: the 9 LOSO fold models once (reproduces stage B exactly; predictions are saved), plus two
pre-declared imbalance variants (uniform priors; balanced subsampling, 3 seeds). Everything else -- confusion
analysis, session analysis, anchor-shift sensitivity, re-anchoring of undetected presses, post-hoc prior
correction -- is inference with the already-fitted fold models.
"""
import csv, glob, hashlib, json, os, sys
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E

OUT = "analysis_output/error_analysis"
FEATS = "band64+mfcc20x10+d"
REF_BAL, REF_ACC = 0.650, 0.792          # stage B numbers this refit must reproduce

ROWS = {0: ("qwertyuiop", 0.0), 1: ("asdfghjkl", 0.25), 2: ("zxcvbnm,.", 0.75)}
POS = {(f"Key{c.upper()}" if c.isalpha() else {",": "Comma", ".": "Period"}[c]): (x + i, float(r)) for r, (s, x) in ROWS.items() for i, c in enumerate(s)}
FINGER = {}
for f, ks in {"L5": "qaz", "L4": "wsx", "L3": "edc", "L2": "rfvtgb", "R2": "yhnujm", "R3": "ik,", "R4": "ol.", "R5": "p"}.items():
    for c in ks: FINGER[f"Key{c.upper()}" if c.isalpha() else {",": "Comma", ".": "Period"}[c]] = f
short = lambda k: k.replace("Key", "")


def wcsv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# ====================================================================================================== data + folds
def load_dev():
    X, meta = E.load_cache()
    S0 = E.split(meta)
    dev_idx = np.where(S0["dev"])[0]
    assert not set(S0["sess"][dev_idx]) & set(E.TEST_SESSIONS)
    X = X[dev_idx]; meta = [meta[i] for i in dev_idx]          # locked sessions dropped BEFORE any feature computation
    S = E.split(meta)
    assert S["dev"].all() and not S["test_sessions"], "a locked session reached the analysis"
    assert S["keys"] == S0["keys"]                             # label set unchanged (it was defined from dev only)
    return X, meta, S


def fit_folds(F, S, train_mask, make=None, fit_hook=None):
    """Fit one model per held-out dev session; returns models, probabilities over S['keys'], predictions, fold log."""
    make = make or E.model_zoo()["lda_shrink"]
    keys = np.array(S["keys"]); n = len(S["y"])
    P = np.zeros((n, len(keys))); models = {}; fold_log = []
    for s in S["dev_sessions"]:
        te = (S["sess"] == s) & S["inset"]
        tr = (S["sess"] != s) & S["inset"] & train_mask
        # --- leakage guards
        assert s not in set(S["sess"][tr]), "held-out session in training"
        assert not set(S["sess"][tr]) & set(E.TEST_SESSIONS), "locked session in training"
        assert set(S["sess"][te]) == {s}
        tr_idx = np.where(tr)[0]
        if fit_hook: tr_idx = fit_hook(tr_idx, s)
        m = make().fit(F[tr_idx], S["y"][tr_idx])
        cls = list(m.classes_); pr = m.predict_proba(F[te])
        for j, k in enumerate(keys):
            if k in cls: P[te, j] = pr[:, cls.index(k)]
        models[s] = m
        fold_log.append(dict(session=s, n_train=len(tr_idx), n_test=int(te.sum()), train_idx=tr_idx))
    return models, P, keys[P.argmax(1)], fold_log


def predict_with(models, F, S, rows_mask):
    keys = np.array(S["keys"]); P = np.zeros((len(S["y"]), len(keys)))
    for s, m in models.items():
        te = rows_mask & (S["sess"] == s)
        if not te.any(): continue
        cls = list(m.classes_); pr = m.predict_proba(F[te])
        for j, k in enumerate(keys):
            if k in cls: P[te, j] = pr[:, cls.index(k)]
    return P


def bal_acc(y, p):
    ks = sorted(set(y)); return float(np.mean([np.mean(p[y == k] == k) for k in ks]))


# ====================================================================================================== technical checks
def technical_checks(X, F, S, models, fold_log):
    chk = {}
    # 1. scaler statistics come from the fold's training rows only
    devs = []
    for fl in fold_log:
        sc = models[fl["session"]][0]
        devs.append(float(np.abs(sc.mean_ - F[fl["train_idx"]].mean(0)).max()))
        allm = F[S["inset"]].mean(0)
        assert not np.allclose(sc.mean_, allm), "scaler looks fitted on all dev rows"
    chk["scaler_mean_equals_training_rows_mean_max_abs_diff"] = max(devs)
    # 2. LDA covariance/shrinkage: refitting on the training rows alone reproduces the fold model exactly
    s0 = fold_log[0]
    ref = E.model_zoo()["lda_shrink"]().fit(F[s0["train_idx"]], S["y"][s0["train_idx"]])
    chk["lda_refit_on_training_rows_identical_coef"] = bool(np.allclose(ref[1].coef_, models[s0["session"]][1].coef_))
    # 3. features are per clip: a clip featurised alone == featurised in the batch
    one = E.feat(FEATS, X[:1]); chk["features_are_per_clip"] = bool(np.allclose(one[0], F[0]))
    # 4. held-out predictions do not depend on which other held-out rows are predicted together
    s = fold_log[1]["session"]; te = np.where((S["sess"] == s) & S["inset"])[0]
    a = models[s].predict_proba(F[te]); b = models[s].predict_proba(F[te[::-1]])[::-1]
    chk["predictions_independent_of_batch"] = bool(np.allclose(a, b))
    # 5. no duplicated audio across sessions (would let a held-out clip be 'seen' in training)
    h = defaultdict(set)
    for i in range(len(X)): h[hashlib.sha1(X[i].tobytes()).hexdigest()].add(S["sess"][i])
    chk["identical_audio_slices_shared_across_sessions"] = int(sum(1 for v in h.values() if len(v) > 1))
    chk["distinct_sessions_in_cache"] = sorted(set(S["sess"]))
    # 6. label set: defined from clean counts in all 9 dev sessions (includes each fold's held-out session)
    keys = S["keys"]; drop = {}
    for fl in fold_log:
        trc = Counter(S["y"][fl["train_idx"]])
        low = [k for k in keys if trc[k] < 20]
        if low: drop[fl["session"][-6:]] = low
    chk["label_set_note"] = ("label set = keys with >=20 clean clips across all 9 dev sessions, so the held-out session's counts help decide which keys exist "
                             "(class membership only, no features or labels of individual clips). Keys that would fall below 20 using training sessions alone, per fold: "
                             + (json.dumps(drop) if drop else "none"))
    chk["selection_note"] = "the feature set and model were chosen using these same 9-fold LOSO results (stages A/B), so 0.650 is mildly optimistic as an estimate; the locked test sessions are the unbiased check"
    chk["segmentation_note"] = "the clean/confident/estimated status of held-out presses comes from keyboard timing + acoustic onset detection, never from key labels"
    chk["all_fold_guards_passed"] = True
    return chk


# ====================================================================================================== main
def main():
    os.makedirs(OUT, exist_ok=True)
    X, meta, S = load_dev()
    F = E.feat(FEATS, X)
    keys = list(S["keys"]); K = np.array(keys); y = S["y"]
    models, P, pred, fold_log = fit_folds(F, S, S["clean"])
    clean = S["clean"] & S["inset"]; conf = S["conf"] & S["inset"]; allp = S["inset"]
    b, a = bal_acc(y[clean], pred[clean]), float(np.mean(pred[clean] == y[clean]))
    assert abs(b - REF_BAL) < 1e-3 and abs(a - REF_ACC) < 1e-3, f"refit does not reproduce stage B ({b:.4f}, {a:.4f})"
    np.save(f"{OUT}/P_dev_lda.npy", P)
    with open(f"{OUT}/dev_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["session", "event_index", "label", "pred", "p_pred", "p_true", "rank_true", "clean30", "alignment", "exclusion"])
        for i in np.where(allp)[0]:
            rk = int((P[i] > P[i, keys.index(y[i])]).sum()) + 1
            w.writerow([S["sess"][i], meta[i]["event_index"], y[i], pred[i], f"{P[i].max():.4f}", f"{P[i, keys.index(y[i])]:.4f}", rk, int(S["clean"][i]), meta[i]["alignment"], meta[i]["exclusion"]])
    R = {"reproduces_stage_B": {"clean_bal_acc": b, "clean_acc": a}}
    print(f"refit reproduces stage B: clean bal {b:.4f} acc {a:.4f}")

    R["technical_checks"] = technical_checks(X, F, S, models, fold_log)
    print("technical checks:", {k: v for k, v in R["technical_checks"].items() if not k.endswith("note")})

    # ---------------------------------------------------------------------------------------------- 1. confusion
    order = sorted(keys, key=lambda k: -int((clean & (y == k)).sum()))
    idx = {k: i for i, k in enumerate(order)}
    C = np.zeros((len(order), len(order)), int)
    for t, p in zip(y[clean], pred[clean]): C[idx[t], idx[p]] += 1
    Cn = C / C.sum(1, keepdims=True)
    wcsv("confusion_raw.csv", ["true\\pred"] + order, [[k] + C[i].tolist() for i, k in enumerate(order)])
    wcsv("confusion_normalised.csv", ["true\\pred"] + order, [[k] + [f"{v:.4f}" for v in Cn[i]] for i, k in enumerate(order)])
    pk = {}
    for i, k in enumerate(order):
        tp = C[i, i]; sup = C[i].sum(); npred = C[:, i].sum()
        prec = tp / npred if npred else 0.0; rec = tp / sup; f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        top_wrong = sorted(((C[i, j], order[j]) for j in range(len(order)) if j != i), reverse=True)[:3]
        pk[k] = dict(support=int(sup), predicted=int(npred), precision=prec, recall=rec, f1=f1,
                     top3_recall=float(np.mean([y[r] in K[np.argsort(-P[r])[:3]] for r in np.where(clean & (y == k))[0]])),
                     most_confused_with=", ".join(f"{short(n)}:{c}" for c, n in top_wrong if c))
    R["per_key"] = pk
    pairs = Counter()
    for t, p in zip(y[clean], pred[clean]):
        if t != p: pairs[(t, p)] += 1
    unordered = Counter()
    for (t, p), n in pairs.items(): unordered[tuple(sorted((t, p)))] += n
    top15 = []
    for (a_, b_), n in unordered.most_common(15):
        na, nb = pk[a_]["support"], pk[b_]["support"]
        d = float(np.hypot(POS[a_][0] - POS[b_][0], POS[a_][1] - POS[b_][1])) if a_ in POS and b_ in POS else None
        top15.append(dict(pair=f"{short(a_)}<->{short(b_)}", errors=n, a_to_b=pairs[(a_, b_)], b_to_a=pairs[(b_, a_)],
                          rate=n / (na + nb), physical_distance=d, same_finger=(FINGER.get(a_) == FINGER.get(b_) and a_ in FINGER)))
    R["top15_confused_pairs"] = top15
    wcsv("top_confused_pairs.csv", list(top15[0].keys()), [list(t.values()) for t in top15])

    # proximity: every unordered pair of letter-block keys
    lk = [k for k in order if k in POS]
    prs = [(a_, b_) for i_, a_ in enumerate(lk) for b_ in lk[i_ + 1:]]
    dist = np.array([np.hypot(POS[a_][0] - POS[b_][0], POS[a_][1] - POS[b_][1]) for a_, b_ in prs])
    rate = np.array([(C[idx[a_], idx[b_]] + C[idx[b_], idx[a_]]) / (pk[a_]["support"] + pk[b_]["support"]) for a_, b_ in prs])
    samef = np.array([FINGER[a_] == FINGER[b_] for a_, b_ in prs]); sameh = np.array([FINGER[a_][0] == FINGER[b_][0] for a_, b_ in prs])
    rho, p_rho = stats.spearmanr(dist, rate)
    adj = dist <= 1.3
    # acoustic distance between class means (descriptive): Mahalanobis under the pooled shrunk covariance of all dev clean clips
    from sklearn.covariance import ledoit_wolf
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(F[clean]); yc = y[clean]
    mu = {k: Z[yc == k].mean(0) for k in order}
    Wc = np.concatenate([Z[yc == k] - mu[k] for k in order]); cov, _ = ledoit_wolf(Wc); icov = np.linalg.inv(cov)
    ad = np.array([float(np.sqrt((mu[a_] - mu[b_]) @ icov @ (mu[a_] - mu[b_]))) for a_, b_ in prs])
    rho_ad, p_ad = stats.spearmanr(ad, rate); rho_pd, p_pd = stats.spearmanr(dist, ad)
    R["proximity"] = {
        "pairs": len(prs), "spearman_physical_distance_vs_confusion_rate": [float(rho), float(p_rho)],
        "confusion_rate_adjacent_vs_not": [float(rate[adj].mean()), float(rate[~adj].mean())], "adjacent_pairs": int(adj.sum()),
        "mannwhitney_adjacent_gt_not_p": float(stats.mannwhitneyu(rate[adj], rate[~adj], alternative="greater").pvalue),
        "confusion_rate_same_finger_vs_not": [float(rate[samef].mean()), float(rate[~samef].mean())], "same_finger_pairs": int(samef.sum()),
        "mannwhitney_same_finger_gt_not_p": float(stats.mannwhitneyu(rate[samef], rate[~samef], alternative="greater").pvalue),
        "confusion_rate_same_hand_vs_not": [float(rate[sameh].mean()), float(rate[~sameh].mean())],
        "mannwhitney_same_hand_gt_not_p": float(stats.mannwhitneyu(rate[sameh], rate[~sameh], alternative="greater").pvalue),
        "spearman_acoustic_distance_vs_confusion_rate": [float(rho_ad), float(p_ad)],
        "spearman_physical_vs_acoustic_distance": [float(rho_pd), float(p_pd)],
    }
    # same-finger effect controlled for distance: compare same-finger vs other pairs at matched distance bands
    bands = [(0, 1.3), (1.3, 2.3), (2.3, 99)]
    R["proximity"]["same_finger_by_distance_band"] = [
        dict(band=f"{lo}-{hi}", same_finger=[float(rate[samef & (dist > lo) & (dist <= hi)].mean()) if (samef & (dist > lo) & (dist <= hi)).any() else None, int((samef & (dist > lo) & (dist <= hi)).sum())],
             other=[float(rate[~samef & (dist > lo) & (dist <= hi)].mean()), int((~samef & (dist > lo) & (dist <= hi)).sum())]) for lo, hi in bands]
    print("proximity:", json.dumps({k: v for k, v in R["proximity"].items() if k != "same_finger_by_distance_band"}))

    # ---------------------------------------------------------------------------------------------- 2. sessions
    sess_rows, cov = [], {}
    for s in S["dev_sessions"]:
        m_c, m_f, m_a = clean & (S["sess"] == s), conf & (S["sess"] == s), allp & (S["sess"] == s)
        md = json.load(open(f"recordings/{s}/metadata.json")); st = json.load(open(f"analysis_output/{s}/sync_stats.json"))
        ks = list(csv.DictReader(open(f"recordings/{s}/keystrokes.csv")))
        kd = np.array([float(r["time_ms"]) for r in ks if r["event_type"] == "keydown" and r["action"] in ("char", "backspace")])
        ikis = np.diff(kd); ikis = ikis[(ikis > 0) & (ikis < 2000)]
        em = list(csv.DictReader(open(f"analysis_output/{s}/event_matches.csv")))
        strength = np.array([float(r["onset_strength_db"]) for r in em if r["event_type"] == "keydown" and r["onset_strength_db"]])
        # noise floor: median band power in the 20 ms pre-onset region of every press (relative; same mic)
        pre = X[S["sess"] == s][:, E.C0 - E.ms(30):E.C0 - E.ms(10)]
        nf = float(np.median(10 * np.log10(np.mean(np.diff(pre, axis=1) ** 2, axis=1) + 1e-20)))
        # session shift: distance of this session's per-key mean (fold-scaled) from the other sessions' per-key means
        sc = models[s][0]; Zs = sc.transform(F)
        shift = []; level = []
        for k in keys:
            a_m = clean & (S["sess"] == s) & (y == k); o_m = clean & (S["sess"] != s) & (y == k)
            if a_m.sum() >= 5 and o_m.sum() >= 5:
                shift.append(float(np.linalg.norm(Zs[a_m].mean(0) - Zs[o_m].mean(0)) / np.sqrt(F.shape[1])))
                level.append(float((F[a_m][:, :64].mean() - F[o_m][:, :64].mean())))
        row = dict(session=s, n_clean=int(m_c.sum()), n_confident=int(m_f.sum()), n_all=int(m_a.sum()),
                   clean_acc=float(np.mean(pred[m_c] == y[m_c])), clean_bal_acc=bal_acc(y[m_c], pred[m_c]),
                   confident_acc=float(np.mean(pred[m_f] == y[m_f])), all_acc=float(np.mean(pred[m_a] == y[m_a])),
                   clean_fraction=float(m_c.sum() / m_a.sum()), wpm=md["stats"]["wpm"], median_iki_ms=float(np.median(ikis)),
                   iki_cv=float(ikis.std() / ikis.mean()), backspace_rate=md["stats"]["backspace_keystrokes"] / max(1, md["stats"]["char_keystrokes"]),
                   peak_amplitude=md["audio"]["peak_amplitude_float"], noise_floor_db=nf,
                   onset_strength_median_db=float(np.median(strength)), detection_rate=st["verdict"]["detection_rate"],
                   onset_lag_sd_ms=st["offset_stats_ms"]["down/char/matched"]["std"], lag_ms=st["offset_stats_ms"]["down/char/matched"]["median"],
                   feature_shift=float(np.mean(shift)), level_offset_db=float(np.mean(level)), start=md["session"]["started_iso"][11:19])
        sess_rows.append(row)
    wcsv("per_session_metrics.csv", list(sess_rows[0].keys()), [[(f"{v:.4f}" if isinstance(v, float) else v) for v in r.values()] for r in sess_rows])
    covs = ["wpm", "median_iki_ms", "iki_cv", "backspace_rate", "peak_amplitude", "noise_floor_db", "onset_strength_median_db", "detection_rate", "onset_lag_sd_ms", "clean_fraction", "feature_shift", "level_offset_db"]
    R["session_correlations_vs_clean_bal_acc"] = {c: [float(x) for x in stats.spearmanr([r[c] for r in sess_rows], [r["clean_bal_acc"] for r in sess_rows])] for c in covs}
    R["session_correlations_vs_all_acc"] = {c: [float(x) for x in stats.spearmanr([r[c] for r in sess_rows], [r["all_acc"] for r in sess_rows])] for c in covs}
    R["sessions"] = sess_rows
    # per key x session recall, inconsistency
    ksr = {}
    for k in keys:
        v = {}
        for s in S["dev_sessions"]:
            m_ = clean & (S["sess"] == s) & (y == k)
            if m_.sum() >= 5: v[s[-6:]] = float(np.mean(pred[m_] == k))
        if len(v) >= 4: ksr[k] = v
    incons = sorted(((float(np.std(list(v.values()))), k) for k, v in ksr.items()), reverse=True)
    R["inconsistent_keys"] = [dict(key=k, sd=sd, min=min(ksr[k].values()), max=max(ksr[k].values()), sessions=len(ksr[k]), worst_session=min(ksr[k], key=ksr[k].get)) for sd, k in incons[:8]]
    wcsv("per_key_by_session_recall.csv", ["key"] + [s[-6:] for s in S["dev_sessions"]], [[k] + [f"{ksr[k][s[-6:]]:.3f}" if s[-6:] in ksr[k] else "" for s in S["dev_sessions"]] for k in order if k in ksr])
    # what differs in the worst session: per-key recall there vs elsewhere, and its predicted-class distribution
    worst = min(sess_rows, key=lambda r: r["clean_bal_acc"])["session"]
    mw = clean & (S["sess"] == worst)
    R["worst_session"] = {"session": worst,
                          "key_recall_drop_vs_other_sessions": sorted([dict(key=k, recall_here=float(np.mean(pred[mw & (y == k)] == k)), recall_elsewhere=float(np.mean(pred[clean & (S["sess"] != worst) & (y == k)] == k)), n=int((mw & (y == k)).sum()))
                                                                  for k in keys if (mw & (y == k)).sum() >= 5], key=lambda d: d["recall_here"] - d["recall_elsewhere"])[:8],
                          "most_predicted_wrongly_here": Counter(pred[mw & (pred != y)]).most_common(6)}

    # ---------------------------------------------------------------------------------------------- 3. imbalance
    fold_counts = []
    for fl in fold_log:
        trc = Counter(y[fl["train_idx"]]); tec = Counter(y[clean & (S["sess"] == fl["session"])])
        fold_counts.append([fl["session"][-6:]] + [f"{trc[k]}/{tec[k]}" for k in order])
    wcsv("fold_counts_train_over_validation.csv", ["held_out"] + order, fold_counts)
    N = clean.sum(); Kn = len(order)
    contrib = sorted(((pk[k]["support"] / N - 1 / Kn) * pk[k]["recall"], k) for k in order)
    R["accuracy_minus_balanced_decomposition"] = {
        "accuracy": a, "balanced": b, "gap": a - b,
        "identity": "acc - bal = sum_k (n_k/N - 1/K) * recall_k",
        "largest_positive_contributors": [dict(key=k, share=pk[k]["support"] / N, recall=pk[k]["recall"], contribution=c) for c, k in sorted(contrib, reverse=True)[:5]],
        "largest_negative_contributors": [dict(key=k, share=pk[k]["support"] / N, recall=pk[k]["recall"], contribution=c) for c, k in contrib[:6]]}
    # (a) post-hoc prior correction with the SAME fitted models: divide by each fold's training priors
    Pa = P.copy()
    for s, m in models.items():
        pri = dict(zip(m.classes_, m[1].priors_)); rows_ = S["sess"] == s
        Pa[rows_] = P[rows_] / np.array([pri.get(k, 1.0) for k in keys]); Pa[rows_] /= Pa[rows_].sum(1, keepdims=True)
    pa = K[Pa.argmax(1)]
    # (b) refit with uniform priors (also changes the prior-weighted pooled covariance)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler as SS
    _, Pu, pu, _ = fit_folds(F, S, S["clean"], make=lambda: make_pipeline(SS(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(len(keys), 1 / len(keys)))))
    # (c) balanced subsampling: cap every class at the median class count of the fold's training set, 3 seeds
    sub = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        def hook(tr_idx, s, rng=rng):
            c = Counter(y[tr_idx]); cap = int(np.median(list(c.values())))
            keep = np.concatenate([rng.choice(tr_idx[y[tr_idx] == k], min(cap, c[k]), replace=False) for k in c])
            return np.sort(keep)
        _, Pb, pb, _ = fit_folds(F, S, S["clean"], fit_hook=hook); sub.append(pb)
    def m3(pp):
        return dict(clean_acc=float(np.mean(pp[clean] == y[clean])), clean_bal_acc=bal_acc(y[clean], pp[clean]),
                    all_acc=float(np.mean(pp[allp] == y[allp])), all_bal_acc=bal_acc(y[allp], pp[allp]))
    R["imbalance_options"] = {"baseline_frequency_priors": m3(pred), "posthoc_prior_correction_same_models": m3(pa),
                              "refit_uniform_priors": m3(pu),
                              "balanced_subsampling_median_cap_3_seeds": {k: float(np.mean([m3(pb)[k] for pb in sub])) for k in m3(pred)},
                              "balanced_subsampling_sd": {k: float(np.std([m3(pb)[k] for pb in sub])) for k in m3(pred)}}
    print("imbalance:", json.dumps(R["imbalance_options"]))

    # ---------------------------------------------------------------------------------------------- 4. classifier vs segmentation
    cats = {}
    for i in np.where(allp)[0]:
        al = meta[i]["alignment"]
        if S["clean"][i]: c_ = "clean"
        elif al in ("detected", "detected_resolved"): c_ = "detected_but_overlapping"
        elif al == "uncertain": c_ = "uncertain_onset"
        else: c_ = "no_onset_detected(estimated)"
        cats.setdefault(c_, []).append(i)
    seg = {}
    for c_, ii in cats.items():
        ii = np.array(ii); seg[c_] = dict(n=len(ii), share=len(ii) / allp.sum(), acc=float(np.mean(pred[ii] == y[ii])),
                                         top3=float(np.mean([y[r] in K[np.argsort(-P[r])[:3]] for r in ii])))
    acc_all = float(np.mean(pred[allp] == y[allp])); acc_c = seg["clean"]["acc"]
    loss = {c_: v["share"] * (acc_c - v["acc"]) for c_, v in seg.items() if c_ != "clean"}
    R["classifier_vs_segmentation"] = {"categories": seg, "all_press_acc": acc_all,
                                      "error_budget_all_presses": {"classifier_error_at_clean_quality": 1 - acc_c,
                                                                   "extra_error_from_each_category": loss, "check_sum": (1 - acc_c) + sum(loss.values()), "observed_error": 1 - acc_all},
                                      "set_accuracies": {"clean": acc_c, "confident": float(np.mean(pred[conf] == y[conf])), "all": acc_all}}
    # 4b. how much does anchor misplacement cost? same fold models, clean clips featurised with the anchor shifted
    sh = {}
    for d in (-10, -5, -2, 0, 2, 5, 10):
        Fs = F.copy(); Fs[clean] = E.feat(FEATS, X[clean], shift=E.ms(d))
        Ps = predict_with(models, Fs, S, clean); ps = K[Ps.argmax(1)]
        sh[d] = float(np.mean(ps[clean] == y[clean]))
    R["anchor_shift_sensitivity_clean_acc"] = sh
    # 4c. accuracy vs onset strength (confident presses)
    em_str = {}
    for s in S["dev_sessions"]:
        for r in csv.DictReader(open(f"analysis_output/{s}/event_matches.csv")):
            if r["onset_strength_db"]: em_str[(s, int(r["event_index"]))] = float(r["onset_strength_db"])
    stv = np.array([em_str.get((S["sess"][i], meta[i]["event_index"]), np.nan) for i in range(len(y))])
    qb = []
    m_ = conf & ~np.isnan(stv)
    edges = np.percentile(stv[m_], [0, 20, 40, 60, 80, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = m_ & (stv >= lo) & (stv <= hi)
        qb.append(dict(strength_db=f"{lo:.1f}-{hi:.1f}", n=int(mm.sum()), acc=float(np.mean(pred[mm] == y[mm]))))
    R["accuracy_by_onset_strength_confident"] = qb
    # 4d. re-anchor presses without a confident onset at the strongest high-frequency peak near the expected time (inference only)
    est = allp & ~conf
    re_idx = np.where(est)[0]
    seg_w = X[re_idx][:, E.C0 - E.ms(15): E.C0 + E.ms(15) + 1]
    d1 = np.diff(seg_w, axis=1) ** 2
    k_ = E.ms(0.5); sm = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k_) / k_, mode="same"), 1, d1)
    peak = sm.argmax(1); new_shift = peak + 1 - E.ms(15)                          # samples relative to the old anchor
    Fr = np.stack([E.feat(FEATS, X[i:i + 1], shift=int(new_shift[j]))[0] for j, i in enumerate(re_idx)])
    Fall = F.copy(); Fall[re_idx] = Fr
    Pr = predict_with(models, Fall, S, est); pr_ = K[Pr.argmax(1)]
    before = float(np.mean(pred[est] == y[est])); after = float(np.mean(pr_[est] == y[est]))
    combined = pred.copy(); combined[est] = pr_[est]
    R["reanchor_unconfident_presses"] = {"n": int(est.sum()), "acc_before": before, "acc_after": after,
                                         "shift_ms_median_abs": float(np.median(np.abs(new_shift)) * 1000 / E.SR),
                                         "all_press_acc_before": acc_all, "all_press_acc_after": float(np.mean(combined[allp] == y[allp])),
                                         "method": "strongest 0.5 ms high-frequency (first-difference) energy peak within +/-15 ms of the original anchor; same fold models, no retraining"}
    print("segmentation:", json.dumps({k: v for k, v in R["classifier_vs_segmentation"].items() if k != "categories"}), "\nshift:", sh, "\nreanchor:", R["reanchor_unconfident_presses"])

    # ---------------------------------------------------------------------------------------------- outputs
    wcsv("per_key_metrics.csv", ["key", "support_clean", "predicted", "precision", "recall", "f1", "top3_recall", "most_confused_with"],
         [[k, v["support"], v["predicted"], f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}", f"{v['top3_recall']:.4f}", v["most_confused_with"]] for k, v in pk.items()])
    json.dump(R, open(f"{OUT}/error_analysis.json", "w"), indent=2, default=float)
    plots(C, Cn, order, pk, sess_rows, sh, qb, dist, rate, samef, OUT)
    print(f"\nwrote {OUT}/")


def plots(C, Cn, order, pk, sess_rows, sh, qb, dist, rate, samef, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lab = [short(k) for k in order]
    for M, name, fmt in ((C, "confusion_raw.png", "d"), (Cn, "confusion_normalised.png", ".2f")):
        fig, ax = plt.subplots(figsize=(13, 11.5))
        im = ax.imshow(M if name.endswith("normalised.png") else np.log1p(M), cmap="viridis")
        for i in range(len(order)):
            for j in range(len(order)):
                v = M[i, j]
                if (v >= 0.05 if fmt == ".2f" else v >= 3) or i == j:
                    ax.text(j, i, format(v, fmt) if fmt == ".2f" else str(v), ha="center", va="center", fontsize=6.5, color="w" if (Cn[i, j] < 0.5) else "k")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, rotation=90); ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{l} ({pk[k]['support']})" for l, k in zip(lab, order)])
        ax.set_xlabel("predicted"); ax.set_ylabel("true (clean clips)")
        ax.set_title(("Row-normalised" if fmt == ".2f" else "Raw counts (colour = log)") + " confusion, LDA, 9 dev sessions LOSO, clean 30 ms clips")
        plt.colorbar(im, fraction=0.04); plt.tight_layout(); plt.savefig(os.path.join(out, name), dpi=120); plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    ks = list(order)
    axs[0, 0].bar(range(len(ks)), [pk[k]["recall"] for k in ks], color="#2980b9", label="recall")
    axs[0, 0].plot(range(len(ks)), [pk[k]["precision"] for k in ks], "o", color="#e67e22", label="precision")
    axs[0, 0].set_xticks(range(len(ks))); axs[0, 0].set_xticklabels([f"{short(k)}\n{pk[k]['support']}" for k in ks], fontsize=7); axs[0, 0].legend(); axs[0, 0].set_title("Per-key recall / precision (x: key, clean n)")
    sr = sess_rows
    axs[0, 1].scatter([r["wpm"] for r in sr], [r["clean_bal_acc"] for r in sr], c="#c0392b", label="clean balanced acc")
    axs[0, 1].scatter([r["wpm"] for r in sr], [r["all_acc"] for r in sr], c="#2980b9", label="every-press acc")
    for r in sr: axs[0, 1].annotate(r["session"][-6:], (r["wpm"], r["clean_bal_acc"]), fontsize=7)
    axs[0, 1].set_xlabel("typing speed (wpm)"); axs[0, 1].legend(); axs[0, 1].set_title("Held-out session accuracy vs typing speed")
    d = sorted(sh); axs[1, 0].plot(d, [sh[k] for k in d], "o-"); axs[1, 0].set_xlabel("anchor shift (ms)"); axs[1, 0].set_ylabel("clean accuracy"); axs[1, 0].grid(alpha=.3)
    axs[1, 0].set_title("Cost of onset misplacement (same models, clean clips)")
    axs[1, 1].scatter(dist[~samef], rate[~samef], s=12, alpha=.5, label="different finger"); axs[1, 1].scatter(dist[samef], rate[samef], s=22, c="#c0392b", label="same finger")
    axs[1, 1].set_xlabel("physical distance (key widths)"); axs[1, 1].set_ylabel("pair confusion rate"); axs[1, 1].legend(); axs[1, 1].set_title("Confusion vs physical distance (letter-block pairs)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "diagnostics.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()
