"""
Train the CNN key classifier (network and spectrogram code shared with the app in sneakey/classifier.py).

Unchanged from the original analysis/key_model_cnn.py: log-mel input, +/-3 ms random shifts during training,
AdamW + one-cycle schedule, 30 epochs, label smoothing 0.1; evaluated by key accuracy at the true press times
(auto-repeats excluded), with test-time averaging over +/-2 ms shifts.

    python training/train_cnn.py validate   # train on train sessions, score on validation sessions
    python training/train_cnn.py final      # train on train+validation, save models/key_cnn.pt (+ manifest),
                                            # score once on the test sessions
Options: --epochs N, --out path (default models/key_cnn.pt; the previous model is kept as key_cnn.previous.pt).
"""
import argparse
import json
import shutil
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from common import OUT
from sneakey.classifier import (MAX_SHIFT_MS, MODEL_FILE, TEST_SHIFTS_MS, KeyNet, crop, normalise,
                                predict_probabilities, spectrograms)
from sneakey.evaluation import key_metrics


def train(spectra, labels, n_keys, epochs=30, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = KeyNet(n_keys)
    optimiser = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(spectra) / 128))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=3e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(epochs):
        network.train()
        order = rng.permutation(len(spectra))
        total = 0.0
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            shift = int(rng.integers(-MAX_SHIFT_MS, MAX_SHIFT_MS + 1))
            logits = network(torch.from_numpy(normalise(crop(spectra[batch], shift))))
            loss = loss_fn(logits, torch.from_numpy(labels[batch]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(batch)
        print(f"  epoch {epoch + 1}: loss {total / len(order):.3f}", flush=True)
    return network


def load():
    data = np.load(OUT / "presses.npz")
    splits = json.loads((OUT / "dataset.json").read_text())["splits"]
    names = data["sessions"]
    masks = {role: np.isin(names, splits[role]) for role in ("train", "validation", "test")}
    return data, splits, masks, spectrograms(data["slices"])


def score(network, spectra, keys, truth):
    probabilities = predict_probabilities(network, spectra, TEST_SHIFTS_MS)
    return key_metrics(keys[probabilities.argmax(1)], truth, probabilities, keys)


def fit(data, spectra, mask, epochs):
    keys = np.array(sorted(set(data["codes"][mask])))
    labels = np.searchsorted(keys, data["codes"]).clip(0, len(keys) - 1)
    network = train(spectra[mask], labels[mask], len(keys), epochs)
    return network, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "final"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", default=str(MODEL_FILE))
    args = parser.parse_args()
    data, splits, masks, spectra = load()
    started = time.time()

    if args.mode == "validate":
        if not masks["validation"].any():
            raise SystemExit("No validation sessions: list some under \"validation\" in data/splits.json.")
        network, keys = fit(data, spectra, masks["train"], args.epochs)
        result = score(network, spectra[masks["validation"]], keys, data["codes"][masks["validation"]])
        print("VALIDATION", json.dumps({**result, "fit_s": round(time.time() - started)}))
        return

    fit_mask = masks["train"] | masks["validation"]
    network, keys = fit(data, spectra, fit_mask, args.epochs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.copy2(out, out.with_name(out.stem + ".previous.pt"))
        if out.with_suffix(".json").exists():
            shutil.copy2(out.with_suffix(".json"), out.with_name(out.stem + ".previous.json"))
    torch.save({"keys": [str(k) for k in keys], "state": network.state_dict()}, out)
    result = score(network, spectra[masks["test"]], keys, data["codes"][masks["test"]]) if masks["test"].any() else None
    manifest = {
        "description": "CNN key classifier (log-mel, 64 bands, -15..45 ms around each press, 700 Hz high-pass).",
        "trained_on": splits["train"] + splits["validation"],
        "test_sessions": splits["test"],
        "test_result_logged_timing": result,
        "epochs": args.epochs,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    print("FINAL", json.dumps(manifest["test_result_logged_timing"]), f"-> {out}")
    print("Re-tune the word decoder for the new model: python training/tune_decoder.py --save")


if __name__ == "__main__":
    main()
