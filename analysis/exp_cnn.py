#!/usr/bin/env python3
"""
Stage D: small CNN on log-mel spectrograms of the press clip, trained leave-one-session-out on the DEV sessions only
(same protocol, label set and metrics as analysis/experiments.py). The locked TEST sessions are never touched here.

    .venv/bin/python analysis/exp_cnn.py <config> [--folds N] [--threads T]

Each run appends its metrics to analysis_output/experiments/results.jsonl and saves the held-out probabilities
(P_dev_<config>.npy) so ensembles can be evaluated without retraining. Training length is fixed in the config:
there is no early stopping on the held-out session.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E

BASE = dict(pre=5, post=25, n_fft=256, hop=48, n_mels=48, width=24, epochs=40, lr=3e-3, wd=1e-2, bs=128,
            shift_ms=2.0, gain_db=6.0, noise_p=0.5, noise_max=0.6, specaug=True, train="clean", tta=[0.0], ls=0.05, seed=0)
CFGS = {
    "cnn_base": {},
    "cnn_noaug": dict(shift_ms=0.0, gain_db=0.0, noise_p=0.0, specaug=False),
    "cnn_conf": dict(train="confident"),
    "cnn_wide": dict(width=32, epochs=60),
    "cnn_conf_wide": dict(train="confident", width=32, epochs=60),
}


TTA5 = [-1.0, -0.5, 0.0, 0.5, 1.0]


def cfg_of(name):
    c = dict(BASE); c.update(CFGS[name]); return c


def build(torch, nn, n_cls, n_mels, width, drop=0.3):
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            w = width
            s.f = nn.Sequential(blk(1, w), blk(w, w), nn.MaxPool2d(2), blk(w, 2 * w), blk(2 * w, 2 * w), nn.MaxPool2d(2), blk(2 * w, 4 * w), blk(4 * w, 4 * w))
            fm = n_mels // 4
            s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(4 * w * fm * 2, 128), nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(128, n_cls))
        def forward(s, x):
            h = s.f(x)                                         # (B, C, M', T')  pool over TIME only: frequency position matters
            h = torch.cat([h.mean(3), h.amax(3)], 1)           # (B, 2C, M')
            return s.head(h.flatten(1))
    return Net()


class Frontend:
    """waveform slices -> normalised log-mel, with optional augmentation."""
    def __init__(self, torch, Xt, cfg):
        self.t, self.Xt, self.c = torch, Xt, cfg
        self.L = E.ms(cfg["pre"] + cfg["post"])
        self.win = torch.hann_window(cfg["n_fft"])
        self.fb = torch.tensor(E.mel_fb(cfg["n_fft"], cfg["n_mels"]), dtype=torch.float32)
        self.mean = self.std = None

    def wave(self, rows, shift):
        t = self.t
        start = E.C0 - E.ms(self.c["pre"]) + shift
        idx = start[:, None] + t.arange(self.L)[None]
        return self.Xt[rows[:, None], idx]

    def mel(self, w):
        t = self.t
        S = t.stft(w, self.c["n_fft"], self.c["hop"], window=self.win, center=False, return_complex=True).abs() ** 2   # (B, F, T)
        return t.log(t.clamp(self.fb @ S, min=1e-12))                                                                 # (B, M, T)

    def fit_norm(self, rows):
        with self.t.no_grad():
            M = self.mel(self.wave(rows, self.t.zeros(len(rows), dtype=self.t.long)))
        self.mean = M.mean((0, 2), keepdim=True)[0]; self.std = M.std((0, 2), keepdim=True)[0] + 1e-5

    def __call__(self, rows, train, shift_ms=0.0, gen=None):
        t, c = self.t, self.c
        B = len(rows)
        if train and c["shift_ms"] > 0:
            sh = t.randint(-E.ms(c["shift_ms"]), E.ms(c["shift_ms"]) + 1, (B,), generator=gen)
        else:
            sh = t.full((B,), E.ms(shift_ms), dtype=t.long)
        w = self.wave(rows, sh)
        if train:
            if c["gain_db"] > 0:
                w = w * (10 ** ((t.rand(B, 1, generator=gen) * 2 - 1) * c["gain_db"] / 20))
            if c["noise_p"] > 0:   # real background: pre-onset audio of other random training clips
                other = self.train_rows[t.randint(0, len(self.train_rows), (B,), generator=gen)]
                n = self.Xt[other[:, None], (t.arange(min(self.L, E.C0 - E.ms(1))))[None]]
                if n.shape[1] < self.L: n = t.nn.functional.pad(n, (0, self.L - n.shape[1]))
                on = (t.rand(B, 1, generator=gen) < c["noise_p"]).float() * t.rand(B, 1, generator=gen) * c["noise_max"]
                w = w + on * n
        M = (self.mel(w) - self.mean) / self.std
        if train and c["specaug"]:
            for _ in range(2):
                f0 = t.randint(0, M.shape[1], (B,), generator=gen); fw = t.randint(0, 7, (B,), generator=gen)
                fi = t.arange(M.shape[1])[None]
                M = M.masked_fill(((fi >= f0[:, None]) & (fi < (f0 + fw)[:, None]))[:, :, None], 0.0)
            t0 = t.randint(0, M.shape[2], (B,), generator=gen); tw = t.randint(0, 3, (B,), generator=gen)
            ti = t.arange(M.shape[2])[None]
            M = M.masked_fill(((ti >= t0[:, None]) & (ti < (t0 + tw)[:, None]))[:, None, :], 0.0)
        return M[:, None]


def train_predict(torch, nn, Xt, yi, tr_rows, te_rows, n_cls, cfg):
    torch.manual_seed(cfg["seed"]); gen = torch.Generator().manual_seed(cfg["seed"])
    fe = Frontend(torch, Xt, cfg); fe.train_rows = tr_rows; fe.fit_norm(tr_rows)
    net = build(torch, nn, n_cls, cfg["n_mels"], cfg["width"])
    cnt = torch.bincount(yi[tr_rows], minlength=n_cls).float()
    wts = (cnt.clamp(min=1) ** -0.5); wts = wts / wts[cnt > 0].mean()
    lossf = nn.CrossEntropyLoss(weight=wts, label_smoothing=cfg["ls"])
    steps_per = int(np.ceil(len(tr_rows) / cfg["bs"]))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per)
    for ep in range(cfg["epochs"]):
        net.train()
        perm = tr_rows[torch.randperm(len(tr_rows), generator=gen)]
        for b in range(steps_per):
            rows = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            loss = lossf(net(fe(rows, True, gen=gen)), yi[rows])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); out = {}
    with torch.no_grad():
        per_shift = {}
        for s in sorted(set([0.0] + TTA5)):
            P = torch.zeros(len(te_rows), n_cls)
            for b in range(0, len(te_rows), 512):
                P[b:b + 512] = torch.softmax(net(fe(te_rows[b:b + 512], False, shift_ms=s)), 1)
            per_shift[s] = P
        out["tta1"] = per_shift[0.0].numpy()
        out["tta5"] = (sum(per_shift[s] for s in TTA5) / len(TTA5)).numpy()   # same trained net, inference-only averaging
    return out, net, fe


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--folds", type=int, default=0); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    import torch, torch.nn as nn
    torch.set_num_threads(a.threads)
    cfg = cfg_of(a.config)
    X, meta = E.load_cache(); S = E.split(meta)
    keys = list(S["keys"]); kidx = {k: i for i, k in enumerate(keys)}
    Xt = torch.tensor(X); yi = torch.tensor([kidx.get(k, -1) for k in S["y"]])
    train_mask = S["clean"] if cfg["train"] == "clean" else S["conf"]
    sessions = S["dev_sessions"][:a.folds] if a.folds else S["dev_sessions"]
    Ps = {"tta1": np.zeros((len(S["y"]), len(keys))), "tta5": np.zeros((len(S["y"]), len(keys)))}; done = np.zeros(len(S["y"]), bool)
    t0 = time.time()
    for s in sessions:
        tr = np.where((S["sess"] != s) & S["dev"] & S["inset"] & train_mask)[0]
        te = np.where((S["sess"] == s) & S["inset"])[0]
        tf = time.time()
        Pf, _, _ = train_predict(torch, nn, Xt, yi, torch.tensor(tr), torch.tensor(te), len(keys), cfg)
        for k in Ps: Ps[k][te] = Pf[k]
        done[te] = True
        acc = np.mean(np.array(keys)[Pf["tta1"].argmax(1)][S["clean"][te]] == S["y"][te][S["clean"][te]])
        print(f"  fold {s[-6:]}: train {len(tr)} test {len(te)} clean acc {acc:.3f} ({time.time() - tf:.0f}s)", flush=True)
    masks = {k: v & done for k, v in E.eval_masks(S, sessions).items()}
    for k, P in Ps.items():
        res = E.evaluate(np.array(keys)[P.argmax(1)], P, np.array(keys), S["y"], masks)
        tag = f"{a.config}+{k}" + (f"_{a.folds}folds" if a.folds else "")
        E.log("D_cnn", tag, res, {"cfg": cfg, "seconds": round(time.time() - t0)})
        if not a.folds: np.save(f"{E.OUT}/P_dev_{a.config}+{k}.npy", P)


if __name__ == "__main__":
    main()
