#!/usr/bin/env python3
"""
Export the baseline model + feature config for the live/browser predictor, with a validation set so the
JavaScript reimplementation can prove it matches Python before any live audio is trusted.

    .venv/bin/python analysis/export_model.py     # -> analysis_output/live_model/{model.json, validation.json}

The model is StandardScaler + shrinkage-LDA (solver='lsqr'), whose prediction is exactly a linear map followed
by a softmax: proba = softmax((f - mean) / scale @ coef.T + intercept). This is fit on the clean 30 ms press
clips of ALL sessions (the live predictor runs on the same laptop), on the same 22-key label set as the dashboard.
"""
import csv, json, os, sys, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_baseline as cb


def main():
    out = "analysis_output/live_model"; os.makedirs(out, exist_ok=True)
    rows, src, sessions, keys, _ = cb.load("datasets", "recordings", 30, 20)
    y = np.array([r["label"] for r in rows])
    X, _ = cb.featurise(rows, src)
    pipe = cb.models()["lda_shrink"]().fit(X, y)
    scaler, lda = pipe[0], pipe[1]
    classes = list(lda.classes_)

    # manual linear+softmax must reproduce sklearn predict_proba
    Z = (X - scaler.mean_) / scaler.scale_
    D = Z @ lda.coef_.T + lda.intercept_
    man = np.exp(D - D.max(1, keepdims=True)); man /= man.sum(1, keepdims=True)
    skl = pipe.predict_proba(X)
    err = float(np.abs(man - skl).max())
    assert err < 1e-6, f"manual softmax != sklearn ({err})"
    print(f"manual vs sklearn predict_proba max abs diff {err:.2e}  (train acc {np.mean(pipe.predict(X) == y):.3f}, {len(rows)} clips, {len(classes)} keys)")

    sr_train = int(src[sessions[0]][1])
    assert all(int(s[1]) == sr_train for s in src.values()), "sessions differ in sample rate"
    seg_len = int(round(0.030 * sr_train))   # 30 ms clip = 1440 @ 48 kHz
    model = {
        "classes": classes,
        "char_map": {k: cb_char(k) for k in classes},
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coef": lda.coef_.tolist(), "intercept": lda.intercept_.tolist(),
        "n_features": int(X.shape[1]), "sr_train": sr_train, "clip_samples": seg_len,
        "pre_ms": 5.0, "post_ms": 25.0,
        "feature_config": {
            "band": {"nfft": 4096, "bands": 32, "fmin": 500.0, "fmax": 20000.0, "tukey_alpha": min(1.0, 2 * 0.001 * sr_train / seg_len)},
            "mel": {"n_fft": 256, "hop": 64, "n_mels": 32, "fmin": 300.0, "fmax": 20000.0, "t_pool": 6},
            "order": "band(32) then logmel(32 mels x 6 time slices, slice-major)"},
        "training": {"sessions": sessions, "clips": len(rows), "per_key": {k: int((y == k).sum()) for k in classes}},
        "note": "trained on all sessions of one laptop; predictions on other laptops/typists are not expected to hold",
    }
    json.dump(model, open(os.path.join(out, "model.json"), "w"))

    # validation clips: raw int16 of the exact 30 ms slice + Python features + model top5 (JS must reproduce)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(rows), min(36, len(rows)), replace=False))
    val = []
    wavs = {s: read_i16(f"recordings/{s}/audio.wav") for s in sessions}
    pre, post = int(round(0.005 * sr_train)), int(round(0.025 * sr_train))
    for i in pick:
        r = rows[i]; a = int(r["anchor_sample"]); pcm = wavs[r["_session"]]
        val.append({
            "label": r["label"], "samples_i16": pcm[a - pre:a + post].tolist(), "sr": sr_train,
            "features": [round(float(v), 6) for v in X[i]],
            "top5": [[classes[j], round(float(skl[i][j]), 5)] for j in np.argsort(-skl[i])[:5]],
        })
    json.dump({"clips": val, "feature_max_abs": float(np.abs(X).max())}, open(os.path.join(out, "validation.json"), "w"))
    print(f"wrote {out}/model.json and {len(val)} validation clips")


def cb_char(code):
    if code == "Space": return " "
    if code in ("Comma", "Period"): return {"Comma": ",", "Period": "."}[code]
    if code.startswith("Key") and len(code) == 4: return code[3].lower()
    return None


def read_i16(p):
    with wave.open(p) as w: return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


if __name__ == "__main__":
    main()
