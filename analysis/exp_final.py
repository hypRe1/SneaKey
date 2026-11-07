#!/usr/bin/env python3
"""
Ensembles on DEV (from saved leave-one-session-out probabilities) and the single FINAL evaluation on the locked TEST sessions.

    .venv/bin/python analysis/exp_final.py ensembles                      # dev only: LDA + CNN probability averaging
    .venv/bin/python analysis/exp_final.py final --cnn <cfg> [--lda-weight w] [--tta tta1|tta5]
        trains the chosen configuration on ALL dev sessions, scores the locked test sessions ONCE, runs the
        pre-onset-noise control with the same pipeline, and writes analysis_output/experiments/final_report.md
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E

LDA_FEATS = "band64+mfcc20x10+d"


def lda_dev_P(X, S):
    P_path = f"{E.OUT}/P_dev_lda.npy"
    if os.path.exists(P_path): return np.load(P_path)
    _, P, _ = E.loso(E.model_zoo()["lda_shrink"], E.feat(LDA_FEATS, X), S, S["clean"], E.eval_masks(S, S["dev_sessions"]), S["dev_sessions"])
    np.save(P_path, P); return P


def score(P, S, sessions):
    keys = np.array(S["keys"]); masks = E.eval_masks(S, sessions)
    return E.evaluate(keys[P.argmax(1)], P, keys, S["y"], masks)


def ensembles():
    X, meta = E.load_cache(); S = E.split(meta)
    Pl = lda_dev_P(X, S)
    E.log("E_ensemble", "lda_only", score(Pl, S, S["dev_sessions"]))
    for f in sorted(os.listdir(E.OUT)):
        if not f.startswith("P_dev_cnn"): continue
        name = f[6:-4]; Pc = np.load(f"{E.OUT}/{f}")
        for w in (0.0, 0.25, 0.5, 0.75):
            P = w * Pl + (1 - w) * Pc
            E.log("E_ensemble", f"{name} lda_w={w}", score(P, S, S["dev_sessions"]))


def final(cnn_cfg_name, lda_w, tta):
    import torch, torch.nn as nn
    import exp_cnn as C
    torch.set_num_threads(12)
    X, meta = E.load_cache(); S = E.split(meta)
    keys = list(S["keys"]); kidx = {k: i for i, k in enumerate(keys)}; K = np.array(keys)
    test = S["test_sessions"]; assert sorted(test) == sorted(E.TEST_SESSIONS)
    rep = {"protocol": "configuration chosen from DEV leave-one-session-out results only; trained on all DEV sessions; scored once on the locked TEST sessions",
           "dev_sessions": S["dev_sessions"], "test_sessions": test, "keys": keys, "cnn_config": cnn_cfg_name, "lda_weight": lda_w, "tta": tta}
    te_all = np.where(np.isin(S["sess"], test) & S["inset"])[0]

    # ---- CNN on all dev
    cfg = C.cfg_of(cnn_cfg_name)
    Xt = torch.tensor(X); yi = torch.tensor([kidx.get(k, -1) for k in S["y"]])
    trm = S["clean"] if cfg["train"] == "clean" else S["conf"]
    tr = np.where(S["dev"] & S["inset"] & trm)[0]
    t0 = time.time()
    Pc, net, fe = C.train_predict(torch, nn, Xt, yi, torch.tensor(tr), torch.tensor(te_all), len(keys), cfg)
    Pc = Pc[tta]
    rep["cnn_train_seconds"] = round(time.time() - t0)

    # ---- LDA on all dev (clean)
    from sklearn.base import clone
    F = E.feat(LDA_FEATS, X)
    trl = S["dev"] & S["inset"] & S["clean"]
    lda = E.model_zoo()["lda_shrink"]().fit(F[trl], S["y"][trl])
    Pl_raw = lda.predict_proba(F[te_all]); Pl = np.zeros_like(Pc)
    for j, k in enumerate(keys): Pl[:, j] = Pl_raw[:, list(lda.classes_).index(k)]

    Pfull = np.zeros((len(S["y"]), len(keys)))
    variants = {"final_ensemble": lda_w * Pl + (1 - lda_w) * Pc, "cnn_alone": Pc, "lda_alone": Pl}
    res = {}
    for name, Pv in variants.items():
        Pfull[:] = 0; Pfull[te_all] = Pv
        res[name] = score(Pfull, S, test)
        E.log("F_final_TEST", name, res[name], {"cnn": cnn_cfg_name, "lda_w": lda_w, "tta": tta})
    rep["test_results"] = res
    # per-session and per-key for the chosen variant
    Pfull[:] = 0; Pfull[te_all] = variants["final_ensemble"]
    pred = K[Pfull.argmax(1)]
    rep["per_session"] = {s: score(Pfull, S, [s]) for s in test}
    cm = (np.isin(S["sess"], test) & S["inset"] & S["clean"])
    rep["per_key_recall_clean"] = {k: float(np.mean(pred[cm & (S["y"] == k)] == k)) for k in keys if (cm & (S["y"] == k)).any()}
    rep["per_key_n_clean"] = {k: int((cm & (S["y"] == k)).sum()) for k in keys}
    rep["top_confusions_clean"] = [f"{t} -> {p}: {n}" for (t, p), n in Counter((t, p) for t, p in zip(S["y"][cm], pred[cm]) if t != p).most_common(12)]
    rep["chance"] = {"balanced_uniform": 1 / len(keys), "top3_uniform": 3 / len(keys),
                     "majority_acc_clean": Counter(S["y"][cm]).most_common(1)[0][1] / cm.sum()}

    # ---- pre-onset control with the SAME pipeline: shift every clip so the model sees only audio ending 5 ms before the onset
    shift_ms = -(cfg["pre"] + cfg["post"] + 5)   # whole window moved before the onset
    assert E.C0 - E.ms(cfg["pre"]) + E.ms(shift_ms) >= 0
    ctrl_cfg = dict(cfg, shift_ms=0.0)                      # no random shifting: control must stay strictly pre-onset
    class Shifted:                                           # move the window for train and test alike
        pass
    Xs = torch.roll(Xt, shifts=-E.ms(shift_ms), dims=1)       # sample i of a row now holds original sample i + shift
    Pctrl, _, _ = C.train_predict(torch, nn, Xs, yi, torch.tensor(tr), torch.tensor(te_all), len(keys), ctrl_cfg)
    Pfull[:] = 0; Pfull[te_all] = Pctrl["tta1"]
    rep["control_pre_onset_cnn"] = score(Pfull, S, test)
    E.log("F_final_TEST", "control_pre_onset_cnn", rep["control_pre_onset_cnn"], {"window_ms_before_onset": [shift_ms - cfg["pre"], shift_ms + cfg["post"]]})
    # shuffled-label control (LDA, cheap)
    rng = np.random.default_rng(0)
    lda_s = E.model_zoo()["lda_shrink"]().fit(F[trl], rng.permutation(S["y"][trl]))
    Ps = lda_s.predict_proba(F[te_all]); Pfull[:] = 0
    for j, k in enumerate(keys): Pfull[te_all, j] = Ps[:, list(lda_s.classes_).index(k)]
    rep["control_shuffled_labels_lda"] = score(Pfull, S, test)
    json.dump(rep, open(f"{E.OUT}/final_results.json", "w"), indent=2, default=float)
    torch.save({"state_dict": net.state_dict(), "cfg": cfg, "keys": keys, "mel_mean": fe.mean, "mel_std": fe.std}, f"{E.OUT}/final_cnn.pt")
    print(json.dumps({k: rep[k] for k in ("test_results", "control_pre_onset_cnn", "control_shuffled_labels_lda")}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--cnn"); ap.add_argument("--lda-weight", type=float, default=0.0); ap.add_argument("--tta", default="tta5")
    a = ap.parse_args()
    if a.cmd == "ensembles": ensembles()
    elif a.cmd == "final": final(a.cnn, a.lda_weight, a.tta)
