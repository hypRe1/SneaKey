#!/usr/bin/env python3
"""
Export held-out baseline predictions for the prediction dashboard.

    .venv/bin/python analysis/export_predictions.py      # -> analysis_output/predictions/{index.json, <session>.json}

For each session, the baseline model (classify_baseline.py's best configuration: shrinkage LDA, global scaling) is trained on
the clean 30 ms press clips of the OTHER sessions only, then predicts every press in this session that has an acoustic
anchor (clean, overlapping, uncertain or estimated). Each prediction carries the alignment/clean status, whether the true key
is in the model's label set at all, and the top-5 class probabilities.
"""
import argparse, csv, json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_baseline as cb

CHAR = {"Space": " ", "Comma": ",", "Period": ".", "Backspace": "\b"}


def char_of(code):
    if code in CHAR: return CHAR[code]
    if code.startswith("Key") and len(code) == 4: return code[3].lower()
    return None


def slice_features(full, sr, anchor):
    pre, post = int(round(0.005 * sr)), int(round(0.025 * sr))
    if anchor - pre < 0 or anchor + post > len(full): return None
    seg = full[anchor - pre: anchor + post]
    return np.concatenate([cb.band_features(seg, sr), cb.logmel_tf(seg, sr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="datasets"); ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--out", default="analysis_output/predictions")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, src, sessions, keys, _ = cb.load(a.datasets, a.recordings, 30, 20)
    y = np.array([r["label"] for r in rows]); sess = np.array([r["_session"] for r in rows])
    X, _ = cb.featurise(rows, src)
    make = cb.models()["lda_shrink"]
    index = []
    for s in sessions:
        model = make().fit(X[sess != s], y[sess != s])           # never sees session s
        classes = list(model.classes_)
        full, sr = src[s]
        man = [r for r in csv.DictReader(open(os.path.join(a.datasets, s, "manifest.csv"), newline=""))
               if r["dataset_class"] == "press" and r["window_ms"] == "30"]
        # parity check: features from the source slice must equal the features the classifier was trained/evaluated on
        own = [i for i, r in enumerate(rows) if r["_session"] == s][:20]
        for i in own:
            f = slice_features(full, sr, int(rows[i]["anchor_sample"]))
            assert np.allclose(f, X[i]), "feature parity failure"
        ev = []
        feats, idx = [], []
        for r in man:
            if r["anchor_sample"] == "": continue
            f = slice_features(full, sr, int(r["anchor_sample"]))
            if f is None: continue
            feats.append(f); idx.append(r)
        P = model.predict_proba(np.array(feats))
        for r, p in zip(idx, P):
            order = np.argsort(-p)[:5]
            in_set = r["label"] in classes
            ev.append({
                "event_index": int(r["event_index"]), "t": int(r["anchor_sample"]) / sr, "t_est": int(r["sample_index_est"]) / sr,
                "code": r["label"], "key": r["key"], "category": r["category"], "action": r["action"], "correct_char": r["correct"],
                "clean": r["clean"] == "1", "alignment": r["alignment"], "exclusion": r["exclusion_reasons"],
                "in_label_set": in_set, "pred": classes[order[0]], "top5": [[classes[j], round(float(p[j]), 4)] for j in order],
                "hit": classes[order[0]] == r["label"], "hit3": r["label"] in [classes[j] for j in order[:3]],
            })
        ev.sort(key=lambda e: e["t"])
        def acc(sel):
            sel = list(sel)
            return {"n": len(sel), "top1": float(np.mean([e["hit"] for e in sel])) if sel else None,
                    "top3": float(np.mean([e["hit3"] for e in sel])) if sel else None}
        summ = {
            "clean_in_set": acc(e for e in ev if e["clean"] and e["in_label_set"]),
            "in_set_any_alignment": acc(e for e in ev if e["in_label_set"]),
            "all_presses_with_anchor": acc(ev),
            "presses_total": len(man), "presses_with_anchor": len(ev),
            "out_of_label_set": sum(1 for e in ev if not e["in_label_set"]),
        }
        pk = {}
        for k in classes:
            sel = [e for e in ev if e["clean"] and e["code"] == k]
            if sel: pk[k] = {"n": len(sel), "top1": float(np.mean([e["hit"] for e in sel]))}
        meta = json.load(open(os.path.join(a.recordings, s, "metadata.json")))
        out = {"session": s, "audio": f"{a.recordings}/{s}/audio.wav", "sample_rate": sr, "duration_s": len(full) / sr,
               "model": "LDA (shrinkage), trained on the other sessions' clean 30 ms press clips", "label_set": classes,
               "training_sessions": [x for x in sessions if x != s], "summary": summ, "per_key_clean": pk,
               "target_text": meta.get("target_text", ""), "typed_text": meta.get("typed_text", ""), "events": ev,
               "char_map": {k: char_of(k) for k in classes}}
        json.dump(out, open(os.path.join(a.out, f"{s}.json"), "w"))
        index.append({"session": s, "file": f"{s}.json", "summary": summ})
        print(f"{s}: clean in-set top1 {summ['clean_in_set']['top1']:.3f} (n={summ['clean_in_set']['n']}), "
              f"any in-set {summ['in_set_any_alignment']['top1']:.3f} (n={summ['in_set_any_alignment']['n']}), "
              f"all presses {summ['all_presses_with_anchor']['top1']:.3f} (n={summ['all_presses_with_anchor']['n']})")
    json.dump({"sessions": index, "label_set": keys}, open(os.path.join(a.out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
