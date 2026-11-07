#!/usr/bin/env python3
"""
SneaKey synchronisation analysis.

Measures the real offset between keyboard events (keystrokes.csv) and the acoustic transients in audio.wav,
WITHOUT assuming that `sample_index_est` is aligned.  Original timestamps are never modified.

    python analysis/sync_analysis.py                       # newest session found in ~/Downloads or ./recordings
    python analysis/sync_analysis.py PATH                  # session dir, <prefix>, or any of the session files
    python analysis/sync_analysis.py PATH --audio-only     # transient detection only (no keystrokes.csv needed)
    python analysis/sync_analysis.py PATH --write-corrected-csv 37.5   # explicit, opt-in: adds a corrected column to a COPY

Sign convention everywhere: offset_ms = acoustic_onset_time - estimated_event_time.  Positive = sound is later in the
WAV than sample_index_est says (what input latency produces).
"""
import argparse, csv, glob, json, math, os, sys, wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, optimize, signal, stats

# ---- verdict criteria (stated in the report so they can be challenged) -------------------------------------------
CRIT = dict(
    detect_min=0.90,        # fraction of eligible character keydowns that have >=1 acoustic onset in the window
    unique_min=0.80,        # fraction that are unambiguous one-to-one matches
    fixed_p95_ms=10.0,      # 95th percentile |residual| after removing the constant offset, for fixed-offset clips
    snapped_p95_ms=25.0,    # ... for "onset-snapped" clips the residual only has to stay inside the search gate
    min_lag_peak_ratio=3.0  # lag-histogram peak vs background level for the lag to be considered significant
)
MODIFIERS = {"Shift", "Control", "Alt", "Meta", "AltGraph", "CapsLock", "OS"}


# ================================================================================================ loading
def find_session(arg):
    """Return dict(wav, csv, meta, name). csv/meta are None when absent."""
    def pair(wav):
        d, b = os.path.dirname(wav), os.path.basename(wav)
        if b == "audio.wav":
            c, m = os.path.join(d, "keystrokes.csv"), os.path.join(d, "metadata.json")
            name = os.path.basename(d) or "session"
        else:
            pre = wav[:-len("audio.wav")]
            c, m, name = pre + "keystrokes.csv", pre + "metadata.json", os.path.basename(pre).rstrip("_")
        return dict(wav=wav, csv=c if os.path.exists(c) else None, meta=m if os.path.exists(m) else None, name=name)

    def newest(root):
        cands = glob.glob(os.path.join(root, "**", "*audio.wav"), recursive=True)
        cands = [c for c in cands if "session_" in c]
        return max(cands, key=os.path.getmtime) if cands else None

    if arg is None:
        for root in (os.path.expanduser("~/Downloads"), "recordings", "."):
            w = newest(root)
            if w: return pair(w)
        sys.exit("No session_*audio.wav found in ~/Downloads, ./recordings or '.'. Pass a path.")
    arg = os.path.expanduser(arg)
    if os.path.isdir(arg):
        if os.path.exists(os.path.join(arg, "audio.wav")): return pair(os.path.join(arg, "audio.wav"))
        w = newest(arg)
        if w: return pair(w)
        sys.exit(f"No audio.wav / *_audio.wav inside {arg}")
    for suf in ("audio.wav", "keystrokes.csv", "metadata.json"):
        if arg.endswith(suf): return pair(arg[:-len(suf)] + "audio.wav")
    if os.path.exists(arg + "_audio.wav"): return pair(arg + "_audio.wav")
    sys.exit(f"Cannot interpret session path: {arg}")


def load_wav(path):
    with wave.open(path) as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw != 2: sys.exit(f"Expected 16-bit PCM, got {8*sw}-bit")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1: x = x.reshape(-1, ch)[:, 0]
    return x, sr


def load_events(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows: sys.exit("keystrokes.csv has no rows")
    ev = dict(
        idx=np.array([int(r["event_index"]) for r in rows]),
        cls=np.array(["down" if r["event_type"] == "keydown" else "up" for r in rows]),
        key=np.array([r["key"] for r in rows]), code=np.array([r["code"] for r in rows]),
        time_ms=np.array([float(r["time_ms"]) for r in rows]),
        t_rel_ms=np.array([float(r["t_rel_ms"]) for r in rows]),
        sample_est=np.array([int(r["sample_index_est"]) for r in rows]),
        repeat=np.array([r["repeat"] == "1" for r in rows]),
        action=np.array([r.get("action", "") for r in rows]),
    )
    cat = []
    for k in ev["key"]:
        cat.append("modifier" if k in MODIFIERS else "backspace" if k == "Backspace" else "char" if len(k) == 1 else "other")
    ev["cat"] = np.array(cat)
    return ev


# ================================================================================================ detection
def band_envelope(x, sr, lo=1500.0, hi=10000.0, fast_ms=0.5):
    hi = min(hi, 0.45 * sr)
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x)                 # zero-phase: no group delay added to onset times
    k = max(1, int(round(fast_ms * 1e-3 * sr)))
    fast = np.sqrt(ndimage.uniform_filter1d(y * y, size=k, mode="nearest"))
    hop = max(1, int(round(sr * 1e-3)))
    m = len(fast) // hop
    e1 = fast[: m * hop].reshape(m, hop).max(axis=1)
    pool = 10
    mm = len(e1) // pool
    e10 = e1[: mm * pool].reshape(mm, pool).max(axis=1)
    win = max(3, int(round(1.0 / (pool * hop / sr))) | 1)
    floor10 = ndimage.median_filter(e10, size=win, mode="reflect")
    t10 = (np.arange(mm) * pool + pool / 2) * hop / sr
    t1 = (np.arange(m) + 0.5) * hop / sr
    floor1 = np.interp(t1, t10, floor10)
    return dict(fast=fast, e1=e1, floor1=np.maximum(floor1, 1e-7), hop=hop, sr=sr, dt=hop / sr)


def detect_onsets(env, ratio=4.0, rise_frac=0.2, min_gap_ms=12.0, dip_frac=0.5, abs_min=2e-4):
    e1, fl, hop, sr, dt, fast = env["e1"], env["floor1"], env["hop"], env["sr"], env["dt"], env["fast"]
    thr = np.maximum(fl * ratio, abs_min)
    pk, _ = signal.find_peaks(e1, height=thr, distance=max(1, int(min_gap_ms * 1e-3 / dt)))
    keep = []                                       # drop 'peaks' that are just ringing of the previous transient
    for p in pk:
        if keep:
            q = keep[-1]
            if e1[q:p + 1].min() > dip_frac * min(e1[q], e1[p]): continue
        keep.append(p)
    onset, peak_t, strength = [], [], []
    look = int(0.015 * sr)
    for p in keep:
        a, b = max(0, p * hop - hop), min(len(fast), p * hop + 2 * hop)
        pi = a + int(np.argmax(fast[a:b])); pv = fast[pi]
        fv = fl[p]
        level = fv + rise_frac * (pv - fv)
        j = pi
        lim = max(0, pi - look)
        while j > lim and fast[j] > level: j -= 1
        onset.append(j / sr); peak_t.append(pi / sr); strength.append(20 * math.log10(pv / fv))
    return dict(onset=np.array(onset), peak=np.array(peak_t), strength_db=np.array(strength), n=len(keep))


# ================================================================================================ matching
def lag_histogram(t_ev, t_on, lo, hi, bin_s=0.001, smooth_ms=2.0):
    """Histogram of (onset - event) over [lo, hi]. Significance = Poisson tail probability of the count within
    +/-3 ms of the mode given the background rate measured away from the mode (a bare peak/background ratio is
    beaten by chance in small samples)."""
    diffs = []
    for t in t_ev:
        a, b = np.searchsorted(t_on, [t + lo, t + hi])
        diffs.append(t_on[a:b] - t)
    d = np.concatenate(diffs) if diffs else np.array([])
    edges = np.arange(lo, hi + bin_s, bin_s)
    h, _ = np.histogram(d, bins=edges)
    hs = ndimage.gaussian_filter1d(h.astype(float), smooth_ms)
    centres = (edges[:-1] + edges[1:]) / 2
    res = dict(centres=centres, hist=hs, mode=None, ratio=0.0, p=1.0, k=0, bg=0.0, n_events=int(len(t_ev)))
    if hs.max() <= 0: return res
    m = float(centres[int(np.argmax(hs))])
    near = d[np.abs(d - m) < 0.005]
    mode = float(np.median(near)) if len(near) else m
    bgmask = np.abs(d - mode) > 0.025
    bg_per_s = bgmask.sum() / max(hi - lo - 0.05, 1e-6)
    lam = bg_per_s * 0.006
    k = int((np.abs(d - mode) <= 0.003).sum())
    p = float(stats.poisson.sf(k - 1, max(lam, 1e-9)))
    res.update(mode=mode, ratio=float(hs.max() / max(np.median(hs[hs > 0]), 1e-9)), p=p, k=k, bg=float(lam))
    return res


def is_significant(h):
    return h["mode"] is not None and h["p"] < 1e-6 and h["k"] >= 8 and h["ratio"] >= CRIT["min_lag_peak_ratio"]


def track_lag(t_ev, t_on, lo, hi, win_s=20.0, step_s=10.0, min_events=25):
    """Lag mode in sliding time windows, so drift / slow wander moves the matching window instead of falling out of it."""
    out = []
    if len(t_ev) == 0: return out
    t0, t1 = t_ev.min(), t_ev.max()
    c = t0 + win_s / 2
    while True:
        a, b = c - win_s / 2, c + win_s / 2
        sel = t_ev[(t_ev >= a) & (t_ev < b)]
        if len(sel) >= min_events:
            h = lag_histogram(sel, t_on, lo, hi)
            out.append(dict(t=float(sel.mean()), mode=h["mode"], p=h["p"], k=h["k"], ratio=h["ratio"], n=int(len(sel)), significant=bool(is_significant(h))))
        if c >= t1 - win_s / 2 + 1e-9 and c >= t0 + win_s / 2: break
        c += step_s
        if c > t1 + step_s: break
    return out


def lag_function(track, fallback):
    """Piecewise-linear lag(t) from significant windows; constant fallback when none are significant."""
    ok = [w for w in track if w["significant"]]
    if not ok: return (lambda t: np.full(np.shape(t), fallback, dtype=float)), False
    tt = np.array([w["t"] for w in ok]); mm = np.array([w["mode"] for w in ok])
    return (lambda t: np.interp(t, tt, mm)), True


def match_events(t_exp, eligible, onsets, mode_i, gate):
    """mode_i: expected (onset - estimate) in seconds for each event, i.e. the local lag, so the window follows drift."""
    n = len(t_exp)
    cand = [np.array([], dtype=int)] * n
    for i in np.where(eligible)[0]:
        a, b = np.searchsorted(onsets, [t_exp[i] + mode_i[i] - gate, t_exp[i] + mode_i[i] + gate])
        cand[i] = np.arange(a, b)
    claims = np.zeros(len(onsets), dtype=int)
    for i in np.where(eligible)[0]: claims[cand[i]] += 1
    rows = [i for i in np.where(eligible)[0] if len(cand[i])]
    cols = sorted({j for i in rows for j in cand[i]})
    chosen = np.full(n, -1)
    if rows:
        cmap = {j: k for k, j in enumerate(cols)}
        cost = np.full((len(rows), len(cols)), 1e6)
        for r, i in enumerate(rows):
            for j in cand[i]: cost[r, cmap[j]] = abs(onsets[j] - t_exp[i] - mode_i[i])
        r_ind, c_ind = optimize.linear_sum_assignment(cost)
        for r, c in zip(r_ind, c_ind):
            if cost[r, c] < 1e5: chosen[rows[r]] = cols[c]
    status, reason = np.array(["ineligible"] * n, dtype=object), np.array([""] * n, dtype=object)
    for i in np.where(eligible)[0]:
        c = cand[i]
        if len(c) == 0: status[i] = "missed"
        elif len(c) > 1: status[i] = "ambiguous"; reason[i] = f"{len(c)} onsets in window"
        elif claims[c[0]] > 1: status[i] = "ambiguous"; reason[i] = "onset also claimed by another event"
        else: status[i] = "matched"
        if chosen[i] < 0 and len(c): chosen[i] = c[int(np.argmin(np.abs(onsets[c] - t_exp[i] - mode_i[i])))]
    return dict(chosen=chosen, status=status, reason=reason, claims=claims, ncand=np.array([len(c) for c in cand]))


def describe(v):
    v = np.asarray(v, float)
    if len(v) == 0: return dict(n=0)
    q = np.percentile(v, [2.5, 5, 25, 50, 75, 95, 97.5])
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(q[3]), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                mad_sigma=float(1.4826 * np.median(np.abs(v - q[3]))), min=float(v.min()), max=float(v.max()),
                p2_5=float(q[0]), p5=float(q[1]), p25=float(q[2]), p75=float(q[4]), p95=float(q[5]), p97_5=float(q[6]),
                iqr=float(q[4] - q[2]))


# ================================================================================================ main analysis
def analyse(x, sr, ev, meta, a):
    out = {}
    dur = len(x) / sr
    env = band_envelope(x, sr)
    det = detect_onsets(env, ratio=a.ratio, rise_frac=a.rise)
    onsets = det["onset"]
    out["audio"] = dict(sample_rate=sr, duration_s=dur, n_onsets=int(len(onsets)),
                        onset_rate_per_s=len(onsets) / dur, strength_db=describe(det["strength_db"]),
                        detector=dict(band_hz=[1500, min(10000, 0.45 * sr)], threshold_ratio_vs_rolling_floor=a.ratio,
                                      onset_rise_fraction=a.rise, floor_window_s=1.0))
    if ev is None: return out, env, det, None
    t_exp = ev["sample_est"] / sr
    eligible = ~ev["repeat"]
    out["events"] = dict(total=int(len(t_exp)), keydown=int((ev["cls"] == "down").sum()), keyup=int((ev["cls"] == "up").sum()),
                         repeat_keydown_excluded=int(ev["repeat"].sum()),
                         first_event_s=float(t_exp.min()), last_event_s=float(t_exp.max()))

    # -- 1. find the lag from the data, separately for press and release, globally AND tracked over time ---------
    lagh, tracks, lagfn, sig = {}, {}, {}, {}
    for c in ("down", "up"):
        m = eligible & (ev["cls"] == c) & ((ev["cat"] == "char") | (ev["cat"] == "backspace"))
        lagh[c] = lag_histogram(t_exp[m], onsets, a.lag_lo, a.lag_hi)
        tracks[c] = track_lag(t_exp[m], onsets, a.lag_lo, a.lag_hi, win_s=a.track_win_s, step_s=a.track_win_s / 2)
    fallback = lagh["down"]["mode"] if lagh["down"]["mode"] is not None else 0.0
    for c in ("down", "up"):
        fb = lagh[c]["mode"] if (lagh[c]["mode"] is not None and is_significant(lagh[c])) else fallback
        lagfn[c], sig[c] = lag_function(tracks[c], fb)
    # Press and release sounds travel the same microphone/driver path, so they share the input latency: releases are
    # matched around the PRESS lag. The keyup's own free lag search is kept only as a diagnostic, because with key
    # rollover (next key pressed before this one is released) its strongest peak is the NEXT key's press click, at a
    # physically impossible negative lag (seen on real data: -36 ms).
    if sig["down"]:
        lagfn["up"] = lagfn["down"]
    out["keyup_lag_note"] = ("keyups matched around the keydown lag (shared input path); keyup free-search mode "
                             f"{'n/a' if lagh['up']['mode'] is None else '%.1f ms' % (lagh['up']['mode'] * 1e3)} is diagnostic only "
                             "(a negative value indicates rollover onto the next key's press)")
    mode_i = np.where(ev["cls"] == "down", lagfn["down"](t_exp), lagfn["up"](t_exp))
    modes = {c: float(np.median(lagfn[c](t_exp))) for c in ("down", "up")}
    out["lag_search"] = {}
    for c in ("down", "up"):
        tr = tracks[c]; okw = [w for w in tr if w["significant"]]
        out["lag_search"][c] = dict(
            global_mode_ms=None if lagh[c]["mode"] is None else lagh[c]["mode"] * 1e3,
            global_peak_to_background=lagh[c]["ratio"], global_p_value=lagh[c]["p"], global_significant=bool(is_significant(lagh[c])),
            windows=len(tr), windows_significant=len(okw),
            tracked_lag_range_ms=[min(w["mode"] for w in okw) * 1e3, max(w["mode"] for w in okw) * 1e3] if okw else None,
            significant=bool(tr and len(okw) >= 0.5 * len(tr)), track=[dict(t=w["t"], mode_ms=w["mode"] * 1e3 if w["mode"] is not None else None, p=w["p"], significant=w["significant"]) for w in tr],
            window_ms=[a.lag_lo * 1e3, a.lag_hi * 1e3], track_window_s=a.track_win_s)
    out["lag_search"]["used_modes_ms"] = {c: modes[c] * 1e3 for c in modes}
    out["lag_search"]["gate_follows_tracked_lag"] = True

    # -- 2. match ------------------------------------------------------------------------------------------------
    M = match_events(t_exp, eligible, onsets, mode_i, a.gate)
    off_ms = np.where(M["chosen"] >= 0, (onsets[np.maximum(M["chosen"], 0)] - t_exp) * 1e3, np.nan)
    out["gate_ms"] = a.gate * 1e3

    summary = {}
    for cls in ("down", "up"):
        for cat in ("char", "backspace", "modifier", "other"):
            m = eligible & (ev["cls"] == cls) & (ev["cat"] == cat)
            if not m.any(): continue
            s = M["status"][m]
            summary[f"{cls}/{cat}"] = {k: int((s == k).sum()) for k in ("matched", "ambiguous", "missed")} | {"eligible": int(m.sum())}
    out["match_summary"] = summary

    # -- 3. offset statistics (clean = unambiguous one-to-one) ---------------------------------------------------
    def sel(cls, cat, status=("matched",)):
        return eligible & (ev["cls"] == cls) & (ev["cat"] == cat) & np.isin(M["status"], status)
    st = {}
    for cls in ("down", "up"):
        for cat in ("char", "backspace", "modifier"):
            m = sel(cls, cat)
            if m.any(): st[f"{cls}/{cat}/matched"] = describe(off_ms[m])
    for cls in ("down", "up"):
        m = sel(cls, "char", ("matched", "ambiguous"))
        if m.any(): st[f"{cls}/char/matched+ambiguous"] = describe(off_ms[m])
    out["offset_stats_ms"] = st

    # un-gated view: nearest onset in the wide window, so the gate cannot hide tails
    wide, wide_ref = [], []
    m = eligible & (ev["cls"] == "down") & (ev["cat"] == "char")
    for i in np.where(m)[0]:
        lo_i, hi_i = np.searchsorted(onsets, [t_exp[i] + a.lag_lo, t_exp[i] + a.lag_hi])
        if hi_i > lo_i:
            c = onsets[lo_i:hi_i]
            wide.append((c[np.argmin(np.abs(c - t_exp[i] - mode_i[i]))] - t_exp[i]) * 1e3); wide_ref.append(mode_i[i] * 1e3)
    inside = float(np.mean(np.abs(np.array(wide) - np.array(wide_ref)) <= a.gate * 1e3)) if wide else float("nan")
    out["ungated_nearest_onset"] = dict(stats=describe(wide), fraction_inside_gate=float(inside))

    # -- 4. latency vs clock error: constant, drift, slow, fast --------------------------------------------------
    dm = sel("down", "char")
    dec = {}
    if dm.sum() >= 20:
        t, y = t_exp[dm], off_ms[dm]
        ts = stats.theilslopes(y, t)
        slope_ms_per_s, icpt = ts[0], ts[1]
        dec["drift_ms_per_s"] = float(slope_ms_per_s)
        dec["drift_ppm"] = float(slope_ms_per_s * 1000.0)
        dec["drift_ci95_ppm"] = [float(ts[2] * 1000), float(ts[3] * 1000)]
        dec["accumulated_drift_over_events_ms"] = float(slope_ms_per_s * (t.max() - t.min()))
        dec["t_start_s"], dec["t_end_s"] = float(t.min()), float(t.max())
        dec["offset_at_start_ms"] = float(icpt + slope_ms_per_s * t.min())
        dec["offset_at_end_ms"] = float(icpt + slope_ms_per_s * t.max())
        dec["drift_significant"] = bool((ts[2] > 0 or ts[3] < 0) and abs(dec["accumulated_drift_over_events_ms"]) >= 5.0)
        res_lin = y - (icpt + slope_ms_per_s * t)
        dec["residual_after_linear"] = describe(res_lin)
        dec["p95_abs_residual_after_linear_ms"] = float(np.percentile(np.abs(res_lin), 95))
        bins = np.arange(t.min(), t.max() + a.bin_s, a.bin_s)
        bi = np.digitize(t, bins)
        slow = np.full(len(t), np.nan); binfo = []
        for b in np.unique(bi):
            mm = bi == b
            if mm.sum() >= 8:
                slow[mm] = np.median(y[mm]); binfo.append((float(t[mm].mean()), float(np.median(y[mm])), int(mm.sum())))
        okb = ~np.isnan(slow)
        dec["bins"] = [dict(t_s=b[0], median_offset_ms=b[1], n=b[2]) for b in binfo]
        if okb.sum() >= 10:
            dec["std_total_ms"] = float(y[okb].std(ddof=1))
            dec["std_slow_ms"] = float(np.std([b[1] for b in binfo], ddof=1)) if len(binfo) > 1 else 0.0
            dec["std_fast_ms"] = float((y[okb] - slow[okb]).std(ddof=1))
            dec["slow_range_ms"] = [float(min(b[1] for b in binfo)), float(max(b[1] for b in binfo))]
            resid_slow = y - slow
        else:
            resid_slow = y - np.median(y)
        dec["residual_after_constant"] = describe(y - np.median(y))
        dec["residual_after_slow"] = describe(resid_slow[~np.isnan(resid_slow)])
        thirds = np.array_split(np.argsort(t), 3)
        dec["median_offset_by_third_ms"] = [float(np.median(y[ix])) for ix in thirds]
    out["decomposition"] = dec

    # press vs release consistency
    pr = {}
    if sel("down", "char").any() and sel("up", "char").any():
        d_, u_ = off_ms[sel("down", "char")], off_ms[sel("up", "char")]
        pr = dict(median_down_ms=float(np.median(d_)), median_up_ms=float(np.median(u_)),
                  up_minus_down_ms=float(np.median(u_) - np.median(d_)), std_down_ms=float(d_.std(ddof=1)), std_up_ms=float(u_.std(ddof=1)))
    out["press_vs_release"] = pr

    # alternative clock mappings from metadata (same matched onsets, different event->sample mapping)
    alt = {}
    if meta and "sync" in meta and "models" in meta["sync"] and dm.any():
        for name, mdl in meta["sync"]["models"].items():
            if not mdl: continue
            est = (mdl["slope_samples_per_ms"] * ev["time_ms"] + mdl["intercept_samples"]) / sr
            o = (onsets[M["chosen"][dm]] - est[dm]) * 1e3
            alt[name] = dict(median_offset_ms=float(np.median(o)), std_ms=float(o.std(ddof=1)),
                             slope_source=mdl.get("slope_source"), residual_rms_ms=float(mdl["residual_rms_samples"] / sr * 1e3),
                             effective_rate_hz=mdl.get("effective_sample_rate_hz"))
        out["alt_mappings"] = alt
        out["primary_mapping"] = meta["sync"].get("primary_mapping")

    # -- 5. sensitivity of the headline numbers to the detector settings ------------------------------------------
    sens = []
    for ratio in (3.0, 4.0, 6.0, 10.0):
        for rise in (0.1, 0.2, 0.5):
            d2 = detect_onsets(env, ratio=ratio, rise_frac=rise)
            M2 = match_events(t_exp, eligible, d2["onset"], mode_i, a.gate)
            m2 = eligible & (ev["cls"] == "down") & (ev["cat"] == "char")
            s2 = M2["status"][m2]
            ok2 = m2 & (M2["status"] == "matched")
            o2 = (d2["onset"][M2["chosen"][ok2]] - t_exp[ok2]) * 1e3
            sens.append(dict(ratio=ratio, rise=rise, onsets=int(d2["n"]), matched=int((s2 == "matched").sum()),
                             ambiguous=int((s2 == "ambiguous").sum()), missed=int((s2 == "missed").sum()),
                             median_ms=float(np.median(o2)) if len(o2) else None, std_ms=float(o2.std(ddof=1)) if len(o2) > 1 else None))
    out["sensitivity"] = sens

    # -- 6. clip-extraction suitability ---------------------------------------------------------------------------
    clip = {}
    if dm.any():
        y = off_ms[dm]; med = float(np.median(y))
        res = y - med
        clip["constant_offset_ms"] = med
        clip["coverage_fixed_offset"] = {f"within_{w}ms": float(np.mean(np.abs(res) <= w)) for w in (5, 10, 20, 30, 50)}
        clip["p95_abs_residual_ms"] = float(np.percentile(np.abs(res), 95))
        clip["p95_abs_residual_after_slow_ms"] = float(np.percentile(np.abs(resid_slow[~np.isnan(resid_slow)]), 95)) if dec else None
        # crowding: how often another event (any key, press or release) falls within +-W of a character keydown
        allm = eligible
        te = np.sort(t_exp[allm])
        crowd = {}
        for w in (0.05, 0.1, 0.15):
            cnt = 0; tot = 0
            for i in np.where(eligible & (ev["cls"] == "down") & (ev["cat"] == "char"))[0]:
                a_, b_ = np.searchsorted(te, [t_exp[i] - w, t_exp[i] + w])
                tot += 1; cnt += (b_ - a_ - 1) > 0
            crowd[f"other_event_within_{int(w*1000)}ms"] = cnt / max(tot, 1)
        clip["crowding"] = crowd
    out["clip"] = clip

    # -- 7. unexplained onsets ------------------------------------------------------------------------------------
    unexp = M["claims"] == 0
    out["unexplained_onsets"] = dict(count=int(unexp.sum()), fraction=float(unexp.mean()) if len(unexp) else 0.0,
                                     strength_db=describe(det["strength_db"][unexp]) if unexp.any() else None)
    out["_dm"] = dm
    return out, env, det, dict(M=M, off_ms=off_ms, t_exp=t_exp, modes=modes, lagh=lagh, tracks=tracks, eligible=eligible)


def verdict(out, a):
    v = {"criteria": CRIT}
    cm = out.get("match_summary", {}).get("down/char")
    if not cm: return {"error": "no character keydowns analysed"}
    det_rate = 1 - cm["missed"] / cm["eligible"]; uniq = cm["matched"] / cm["eligible"]
    clip = out["clip"]
    v["detection_rate"] = det_rate; v["unambiguous_rate"] = uniq
    v["lag_significant"] = out["lag_search"]["down"]["significant"]  # majority of time windows show a significant peak
    v["fixed_offset_ok"] = bool(v["lag_significant"] and det_rate >= CRIT["detect_min"] and clip["p95_abs_residual_ms"] <= CRIT["fixed_p95_ms"])
    v["fixed_offset_p95_ms"] = clip["p95_abs_residual_ms"]
    v["drift_corrected_p95_ms"] = clip.get("p95_abs_residual_after_slow_ms")
    dec = out.get("decomposition", {})
    v["drift_significant"] = dec.get("drift_significant")
    v["linear_corrected_p95_ms"] = dec.get("p95_abs_residual_after_linear_ms")
    v["linear_corrected_ok"] = bool(v["lag_significant"] and det_rate >= CRIT["detect_min"] and v["linear_corrected_p95_ms"] is not None
                                    and v["linear_corrected_p95_ms"] <= CRIT["fixed_p95_ms"])
    v["snapped_ok"] = bool(v["lag_significant"] and uniq >= CRIT["unique_min"] and det_rate >= CRIT["detect_min"])
    return v


# ================================================================================================ plotting
def plot_all(x, sr, ev, env, det, A, out, od, name, aud_only=False):
    paths = []
    dur = len(x) / sr
    def save(fig, fn):
        p = os.path.join(od, fn); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); paths.append(fn)

    # 1 overview
    nb = 4000
    seg = len(x) // nb
    xs = x[: seg * nb].reshape(nb, seg)
    tt = (np.arange(nb) + 0.5) * seg / sr
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True, gridspec_kw=dict(height_ratios=[4, 1.6]))
    ax[0].fill_between(tt, xs.min(1), xs.max(1), color="#4a6fa5", lw=0.3)
    ax[0].set_ylabel("amplitude"); ax[0].set_title(f"{name}: waveform with estimated keydown times (uncorrected sample_index_est) and detected acoustic onsets")
    rows = [("detected acoustic onsets", det["onset"], "k")]
    if ev is not None:
        t = A["t_exp"]; el = A["eligible"]
        rows = [("keydown (est.)", t[el & (ev["cls"] == "down")], "#c0392b"), ("keyup (est.)", t[el & (ev["cls"] == "up")], "#2980b9")] + rows
    ax[1].eventplot([r[1] for r in rows], colors=[r[2] for r in rows], lineoffsets=range(len(rows)), linelengths=0.8, linewidths=0.5)
    ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([r[0] for r in rows]); ax[1].set_xlabel("time in WAV (s)"); ax[1].set_xlim(0, dur)
    save(fig, "01_overview_waveform.png")
    if aud_only:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ioi = np.diff(det["onset"]) * 1e3
        ax[0].hist(ioi[ioi < 800], bins=80, color="#4a6fa5"); ax[0].set_xlabel("inter-onset interval (ms)"); ax[0].set_title("Detected onset spacing")
        ax[1].hist(det["strength_db"], bins=50, color="#8e44ad"); ax[1].set_xlabel("onset strength vs local floor (dB)")
        save(fig, "02_onset_statistics.png")
        return paths

    # 2 zoom windows
    tk = A["t_exp"][A["eligible"] & (ev["cls"] == "down") & (ev["cat"] == "char")]
    starts = [max(0, np.quantile(tk, q) - 0.4) for q in (0.08, 0.5, 0.92)]
    wlen = 2.5
    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    med = out["clip"].get("constant_offset_ms", 0.0) / 1e3 if out.get("clip") else 0.0
    for ax_, s0 in zip(axs, starts):
        i0, i1 = int(s0 * sr), int((s0 + wlen) * sr)
        tx = np.arange(i0, i1) / sr
        ax_.plot(tx, x[i0:i1], lw=0.4, color="#4a6fa5")
        ax2 = ax_.twinx(); ax2.plot(tx, env["fast"][i0:i1], lw=0.6, color="#e67e22", alpha=0.7); ax2.set_ylabel("band envelope", color="#e67e22")
        for i in np.where((A["t_exp"] >= s0) & (A["t_exp"] <= s0 + wlen) & ~ev["repeat"])[0]:
            dn = ev["cls"][i] == "down"
            ax_.axvline(A["t_exp"][i], color="#c0392b" if dn else "#2980b9", ls="-" if dn else "--", lw=0.9, alpha=0.9)
            if dn and ev["cat"][i] in ("char", "backspace"):
                ax_.text(A["t_exp"][i], ax_.get_ylim()[1] * 0.92, ev["key"][i] if ev["key"][i] != " " else "␣", fontsize=8, color="#c0392b", ha="left")
        m = (det["onset"] >= s0) & (det["onset"] <= s0 + wlen)
        ax_.plot(det["onset"][m], np.zeros(m.sum()), "kv", ms=6)
        ax_.set_xlim(s0, s0 + wlen); ax_.set_ylabel("amplitude")
    axs[0].set_title("Zoom: red solid = est. keydown, blue dashed = est. keyup (both UNcorrected); black triangles = detected onsets")
    axs[-1].set_xlabel("time in WAV (s)")
    save(fig, "02_zoom_windows.png")

    # 3 lag search: global histograms + lag tracked over time
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
    for ax_, c, col in zip(axs[:2], ("down", "up"), ("#c0392b", "#2980b9")):
        h = A["lagh"][c]
        ax_.plot(h["centres"] * 1e3, h["hist"], color=col)
        if h["mode"] is not None:
            ax_.axvline(h["mode"] * 1e3, color="k", ls=":")
            ax_.set_title(f"{'keydown' if c=='down' else 'keyup'}, whole recording: mode {h['mode']*1e3:.1f} ms, p={h['p']:.1e}", fontsize=10)
        ax_.axvline(0, color="grey", lw=0.6); ax_.set_xlabel("onset time - estimated event time (ms)"); ax_.set_ylabel("(event, onset) pairs")
    for c, col, mk in (("down", "#c0392b", "o"), ("up", "#2980b9", "s")):
        tr = A["tracks"][c]
        if not tr: continue
        for w in tr:
            if w["mode"] is None: continue
            axs[2].plot(w["t"], w["mode"] * 1e3, mk, color=col, mfc=col if w["significant"] else "none", ms=6)
        axs[2].plot([w["t"] for w in tr if w["mode"] is not None], [w["mode"] * 1e3 for w in tr if w["mode"] is not None], "-", color=col, lw=0.8, alpha=0.5, label=c)
    axs[2].set_xlabel("time in WAV (s)"); axs[2].set_ylabel("lag mode (ms)"); axs[2].legend(); axs[2].grid(alpha=0.3)
    axs[2].set_title("Lag tracked in sliding windows (filled = significant peak; hollow = no significant peak)", fontsize=10)
    fig.suptitle("Lag search over a wide window, no alignment assumed. Sharp peak = consistent offset; slanted track = drift", y=1.03)
    save(fig, "03_lag_search.png")

    # 4 offset distributions
    M = A["M"]; off = A["off_ms"]; el = A["eligible"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    for c, col, lab in (("down", "#c0392b", "keydown"), ("up", "#2980b9", "keyup")):
        m = el & (ev["cls"] == c) & (ev["cat"] == "char") & (M["status"] == "matched")
        if not m.any(): continue
        v = off[m]
        axs[0].hist(v, bins=np.arange(v.min() - 1, v.max() + 2, 1.0), alpha=0.6, color=col, label=f"{lab} (n={m.sum()}, median {np.median(v):.1f}, sd {v.std(ddof=1):.1f} ms)")
        axs[0].axvline(np.median(v), color=col, ls="--")
        s = np.sort(v); axs[1].plot(s, np.arange(1, len(s) + 1) / len(s), color=col, label=lab)
    axs[0].set_xlabel("offset = onset - est. event time (ms)"); axs[0].legend(fontsize=8); axs[0].set_title("Unambiguous matches, character keys")
    axs[1].set_xlabel("offset (ms)"); axs[1].set_ylabel("ECDF"); axs[1].grid(alpha=0.3); axs[1].set_title("Cumulative distribution")
    dm = out["_dm"]
    if dm.any():
        r = off[dm] - np.median(off[dm])
        axs[2].hist(r, bins=np.arange(-30, 30.5, 1), color="#27ae60"); axs[2].set_title("Keydown residual after removing constant offset")
        for w in (10, 20): axs[2].axvspan(-w, w, color="grey", alpha=0.08)
        axs[2].set_xlabel("ms")
    save(fig, "04_offset_distribution.png")

    # 5 offset vs time
    fig, ax = plt.subplots(figsize=(15, 5))
    for c, col, mk in (("down", "#c0392b", "o"), ("up", "#2980b9", "s")):
        m = el & (ev["cls"] == c) & (ev["cat"] == "char") & (M["status"] == "matched")
        ax.scatter(A["t_exp"][m], off[m], s=9, color=col, marker=mk, alpha=0.5, label=f"{c} (matched)")
    m = el & (ev["cls"] == "down") & (ev["cat"] == "char") & (M["status"] == "ambiguous") & ~np.isnan(off)
    ax.scatter(A["t_exp"][m], off[m], s=12, facecolors="none", edgecolors="#7f8c8d", label="keydown (ambiguous)")
    d = out["decomposition"]
    if d.get("bins"):
        ax.plot([b["t_s"] for b in d["bins"]], [b["median_offset_ms"] for b in d["bins"]], "k-o", ms=4, label=f"{out['decomposition'].get('std_slow_ms', float('nan')):.2f} ms sd of {len(d['bins'])} time-bin medians")
        tt_ = np.array([b["t_s"] for b in d["bins"]])
    if "drift_ppm" in d:
        mm = dm; t_ = A["t_exp"][mm]; y_ = off[mm]
        ts = stats.theilslopes(y_, t_)
        xx = np.array([t_.min(), t_.max()]); ax.plot(xx, ts[1] + ts[0] * xx, "g--", lw=2, label=f"Theil-Sen drift {d['drift_ppm']:+.0f} ppm (95% CI {d['drift_ci95_ppm'][0]:+.0f}..{d['drift_ci95_ppm'][1]:+.0f})")
    ax.set_xlabel("time in WAV (s)"); ax.set_ylabel("offset (ms)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Offset over time: slope/slow wander = clock-sync error; constant level = latency + physical delay")
    save(fig, "05_offset_vs_time.png")

    # 6 event-triggered envelope
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw=dict(height_ratios=[3, 1.3]))
    pre, post = 0.1, 0.35; dt = env["dt"]
    k0, k1 = int(-pre / dt), int(post / dt)
    for col, (c, lab) in enumerate((("down", "keydown"), ("up", "keyup"))):
        idx = np.where(el & (ev["cls"] == c) & (ev["cat"] == "char"))[0]
        idx = idx[np.argsort(A["t_exp"][idx])]
        mats = []
        for i in idx:
            c0 = int(round(A["t_exp"][i] / dt))
            if c0 + k0 < 0 or c0 + k1 >= len(env["e1"]): continue
            sl = slice(c0 + k0, c0 + k1)
            mats.append(20 * np.log10(env["e1"][sl] / env["floor1"][sl]))
        if not mats: continue
        Mx = np.array(mats)
        im = axs[0, col].imshow(Mx, aspect="auto", origin="lower", extent=[-pre * 1e3, post * 1e3, 0, len(Mx)], cmap="magma", vmin=0, vmax=np.percentile(Mx, 99))
        axs[0, col].axvline(0, color="w", lw=0.8); axs[0, col].set_ylabel(f"{lab} #, in time order")
        axs[0, col].set_title(f"{lab}-triggered envelope (dB over floor)\naligned on UNcorrected estimate", fontsize=10)
        tx = np.arange(k0, k1) * dt * 1e3
        axs[1, col].plot(tx, np.median(Mx, 0), color="#c0392b" if c == "down" else "#2980b9")
        pk = tx[int(np.argmax(np.median(Mx, 0)))]
        axs[1, col].axvline(pk, color="k", ls=":"); axs[1, col].axvline(0, color="grey", lw=0.6)
        axs[1, col].set_xlabel("ms relative to estimated event time"); axs[1, col].set_ylabel("median dB"); axs[1, col].set_title(f"median trace, peak at {pk:.0f} ms")
    save(fig, "06_event_triggered_envelope.png")

    # 7 match status + crowding
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    keys = list(out["match_summary"].keys())
    bottoms = np.zeros(len(keys))
    for k, col in (("matched", "#27ae60"), ("ambiguous", "#f1c40f"), ("missed", "#c0392b")):
        v = np.array([out["match_summary"][kk][k] for kk in keys]); axs[0].bar(keys, v, bottom=bottoms, color=col, label=k); bottoms += v
    axs[0].legend(); axs[0].set_title("Match status by event class"); axs[0].tick_params(axis="x", rotation=45)
    ioi = np.diff(np.sort(A["t_exp"][el & (ev["cls"] == "down") & (ev["cat"] == "char")])) * 1e3
    axs[1].hist(ioi[ioi < 600], bins=60, color="#4a6fa5"); axs[1].axvline(a_gate_ms(out), color="r", ls="--"); axs[1].set_title("Inter-keydown interval (red = matching gate half-width)")
    axs[1].set_xlabel("ms")
    save(fig, "07_match_status_and_intervals.png")
    return paths


def a_gate_ms(out): return out.get("gate_ms", 40)


# ================================================================================================ report
def f(v, d=2, unit=""):
    return "n/a" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{v:.{d}f}{unit}"

def stat_row(label, s):
    if not s or s.get("n", 0) == 0: return f"| {label} | 0 | | | | | | | |"
    return f"| {label} | {s['n']} | {f(s['mean'])} | {f(s['median'])} | {f(s['std'])} | {f(s['mad_sigma'])} | {f(s['p2_5'])} .. {f(s['p97_5'])} | {f(s['min'])} .. {f(s['max'])} | {f(s['iqr'])} |"


def write_report(out, vd, files, sess, od, a, plots, audio_only):
    L = []
    L += [f"# Synchronisation report: {sess['name']}", "",
          f"* WAV: `{sess['wav']}`  ({out['audio']['duration_s']:.2f} s, {out['audio']['sample_rate']} Hz)",
          f"* keystrokes.csv: `{sess['csv']}`", f"* metadata.json: `{sess['meta']}`", ""]
    au = out["audio"]
    L += ["## Acoustic transient detection", "",
          f"Band-pass {au['detector']['band_hz'][0]:.0f}-{au['detector']['band_hz'][1]:.0f} Hz (zero-phase, so no delay is added), 0.5 ms RMS envelope, "
          f"peaks > {au['detector']['threshold_ratio_vs_rolling_floor']:.1f}x the rolling 1 s median floor, onset = leading edge where the envelope first rises "
          f"{au['detector']['onset_rise_fraction']*100:.0f}% of the way from floor to peak. **{au['n_onsets']} onsets** ({au['onset_rate_per_s']:.2f}/s).", ""]
    if audio_only:
        L += ["**Audio-only mode: no keystrokes.csv was supplied, so no keyboard/audio offsets can be computed.**", "",
              "Plots: " + ", ".join(plots), ""]
        open(os.path.join(od, "sync_report.md"), "w").write("\n".join(L)); return
    ev = out["events"]
    L += ["## Events", "", f"{ev['total']} events ({ev['keydown']} keydown, {ev['keyup']} keyup). {ev['repeat_keydown_excluded']} auto-repeat keydowns are excluded "
          "from matching because a held key makes no new sound. Offsets below are `onset - sample_index_est/sample_rate`; positive = sound later than the estimate.", ""]

    L += ["## 1. Does a timing relationship exist? (lag search, no alignment assumed)", "",
          "All (onset - event) differences within the window were histogrammed. A peak is called significant when the number of pairs within +/-3 ms of the mode is "
          f"implausible under the measured background rate (Poisson p < 1e-6) and the peak is at least {CRIT['min_lag_peak_ratio']:.0f}x the background. "
          f"This is done for the whole recording and again in sliding {a.track_win_s:.0f} s windows; the matching gate follows the *tracked* lag, so drift cannot push real onsets out of the window.", "",
          "| class | whole-recording mode (ms) | p-value | significant | windows significant | tracked lag range (ms) | search window (ms) |", "|---|---|---|---|---|---|---|"]
    for c in ("down", "up"):
        s_ = out["lag_search"][c]
        rng = "n/a" if not s_["tracked_lag_range_ms"] else f"{s_['tracked_lag_range_ms'][0]:.1f} .. {s_['tracked_lag_range_ms'][1]:.1f}"
        L.append(f"| {c} | {f(s_['global_mode_ms'],1)} | {s_['global_p_value']:.1e} | {'yes' if s_['global_significant'] else 'no'} | {s_['windows_significant']}/{s_['windows']} "
                 f"({'yes' if s_['significant'] else '**NO**'}) | {rng} | {s_['window_ms'][0]:.0f} .. {s_['window_ms'][1]:.0f} |")
    L += ["", "If the whole-recording peak is weak but most windows are significant with a moving mode, the offset is **drifting** (a sync error), not absent.",
          "", f"Note: {out.get('keyup_lag_note', '')}.", "", "![lag search](03_lag_search.png)", ""]
    L += ["## 2. Matching, missed detections and ambiguity", "",
          f"Each eligible event gets a window of +/-{out['gate_ms']:.0f} ms around (estimate + its class's lag mode). *matched* = exactly one onset in the window and no other event claims it; "
          "*ambiguous* = several onsets in the window, or the onset is also in another event's window (typical in fast typing, including a release click landing in the next press window); "
          "*missed* = no onset in the window.", "",
          "| class/category | eligible | matched | ambiguous | missed | miss rate |", "|---|---|---|---|---|---|"]
    for k, s in out["match_summary"].items():
        L.append(f"| {k} | {s['eligible']} | {s['matched']} | {s['ambiguous']} | {s['missed']} | {100*s['missed']/s['eligible']:.1f}% |")
    u = out["unexplained_onsets"]
    L += ["", f"Onsets not inside any event window: **{u['count']}** ({100*u['fraction']:.1f}% of all onsets): other sounds (speech, desk, mouse, touchpad), quiet keys detected late, or windows the gate excluded.",
          "", "![match status](07_match_status_and_intervals.png)", "![zoom](02_zoom_windows.png)", "![overview](01_overview_waveform.png)", ""]
    ug = out["ungated_nearest_onset"]
    L += [f"Gate check: nearest onset in the full lag window for each character keydown lies inside the gate for **{100*ug['fraction_inside_gate']:.1f}%**; "
          "a low value means the gate is cutting off a real tail and the statistics below under-state the spread.", ""]

    L += ["## 3. Offset statistics (ms; onset - estimate)", "",
          "| set | n | mean | median | std | robust sigma (MAD) | 2.5%..97.5% | min..max | IQR |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in out["offset_stats_ms"].items(): L.append(stat_row(k, s))
    L += [stat_row("ungated nearest onset, keydown/char", ug["stats"]), "",
          "![offset distribution](04_offset_distribution.png)", "![event-triggered](06_event_triggered_envelope.png)", ""]
    pr = out["press_vs_release"]
    if pr:
        L += ["### Press versus release", "",
              f"Median offset for keydown **{pr['median_down_ms']:.1f} ms**, for keyup **{pr['median_up_ms']:.1f} ms** (difference {pr['up_minus_down_ms']:+.1f} ms; sd {pr['std_down_ms']:.1f} / {pr['std_up_ms']:.1f} ms). "
              "Both go through the same input chain and clock mapping, so a shared constant offset points to latency / mapping. A *difference* between them is physical (release click timing versus when the OS reports the release) and cannot be a clock error. "
              "Every key can click on both press and release, so release events are matched in their own class rather than counted as misses of the press.", ""]

    d = out["decomposition"]
    L += ["## 4. Clock-synchronisation error versus input latency", ""]
    if d:
        if d.get("drift_significant"):
            L += [f"* **Offset is not constant**: it moves from **{d['offset_at_start_ms']:.1f} ms** at the start of typing (t={d['t_start_s']:.0f} s) to **{d['offset_at_end_ms']:.1f} ms** at the end (t={d['t_end_s']:.0f} s). "
                  "A single median would mix the two, so it is not reported as 'the latency'. The *level* (offset at the start) is input latency + physical key delay + constant mapping bias + detector definition; "
                  "the *change* is clock-synchronisation error, because input latency does not grow during a recording.",
                  ]
        else:
            L += [f"* **Constant component**: median keydown offset **{out['clip']['constant_offset_ms']:.1f} ms**, no significant drift. This is the sum of microphone/driver/browser input latency, "
                  "the physical delay between finger contact and the OS key event, any constant bias in the clock mapping, and the detector's definition of 'onset'. "
                  "These cannot be separated by this recording alone; they are not clock *error* because they do not change over time."]
        L += [
              f"* **Drift (clock rate mismatch)**: Theil-Sen slope **{d['drift_ppm']:+.0f} ppm** (95% CI {d['drift_ci95_ppm'][0]:+.0f} .. {d['drift_ci95_ppm'][1]:+.0f}); accumulates **{d['accumulated_drift_over_events_ms']:+.1f} ms** across the typed span. "
              "A CI excluding 0 means the sync model's slope is measurably wrong; real oscillators differ by <~100 ppm."]
        if "std_slow_ms" in d:
            L += [f"* **Slow wander** (medians of {a.bin_s:.0f} s bins): sd **{d['std_slow_ms']:.2f} ms**, range {d['slow_range_ms'][0]:.1f} .. {d['slow_range_ms'][1]:.1f} ms. Time-varying, so this is sync error (or changing buffer fill), not a fixed latency.",
                  f"* **Fast jitter** (per-event scatter around the bin median): sd **{d['std_fast_ms']:.2f} ms**. Mixes physical key-travel variation, event dispatch jitter, and detector error; it is a floor on how well *any* timestamp can predict the onset."]
        L += [f"* Median offset in first / middle / last third of the typing: {', '.join(f(v,1) for v in d['median_offset_by_third_ms'])} ms."]
    if "alt_mappings" in out:
        L += ["", f"Same matched onsets re-timed with each clock mapping saved in metadata.json (primary = `{out.get('primary_mapping')}`):", "",
              "| mapping | median offset (ms) | std (ms) | slope source | effective rate (Hz) | its own fit residual RMS (ms) |", "|---|---|---|---|---|---|"]
        for k, s in out["alt_mappings"].items():
            L.append(f"| {k} | {s['median_offset_ms']:.1f} | {s['std_ms']:.2f} | {s['slope_source']} | {f(s['effective_rate_hz'],1)} | {s['residual_rms_ms']:.2f} |")
        L += ["", "The spread of medians across mappings is mapping uncertainty alone; the page's own sync-fit residual shows how well it tracked its clock pairs, which says nothing about microphone input latency."]
    L += ["", "![offset vs time](05_offset_vs_time.png)", ""]

    L += ["## 5. Sensitivity to detector settings", "",
          "The headline offset depends on how 'onset' is defined (a later threshold crossing gives a later onset). If the median moves by several ms across rows, treat the constant offset as uncertain by that much.", "",
          "| threshold (x floor) | rise fraction | onsets | matched | ambiguous | missed | median (ms) | std (ms) |", "|---|---|---|---|---|---|---|---|"]
    for s in out["sensitivity"]:
        L.append(f"| {s['ratio']:.0f} | {s['rise']:.1f} | {s['onsets']} | {s['matched']} | {s['ambiguous']} | {s['missed']} | {f(s['median_ms'],1)} | {f(s['std_ms'],2)} |")
    meds = [s["median_ms"] for s in out["sensitivity"] if s["median_ms"] is not None]
    if meds: L += ["", f"Median offset ranges over **{min(meds):.1f} .. {max(meds):.1f} ms** across detector settings."]

    L += ["", "## 6. Can we cut labelled per-keystroke clips?", ""]
    if "error" in vd: L += [vd["error"]]
    else:
        c = out["clip"]
        L += ["| criterion | value | threshold | pass |", "|---|---|---|---|",
              f"| significant lag peak in a majority of time windows (keydown) | {out['lag_search']['down']['windows_significant']}/{out['lag_search']['down']['windows']} | >= 50% | {'yes' if vd['lag_significant'] else '**NO**'} |",
              f"| detection rate, char keydowns | {100*vd['detection_rate']:.1f}% | >= {100*CRIT['detect_min']:.0f}% | {'yes' if vd['detection_rate']>=CRIT['detect_min'] else '**NO**'} |",
              f"| unambiguous one-to-one matches | {100*vd['unambiguous_rate']:.1f}% | >= {100*CRIT['unique_min']:.0f}% | {'yes' if vd['unambiguous_rate']>=CRIT['unique_min'] else '**NO**'} |",
              f"| p95 abs residual, constant offset removed | {c['p95_abs_residual_ms']:.1f} ms | <= {CRIT['fixed_p95_ms']} ms | {'yes' if c['p95_abs_residual_ms']<=CRIT['fixed_p95_ms'] else '**NO**'} |", ""]
        if c.get("p95_abs_residual_after_slow_ms") is not None:
            L += [f"With the slow component also removed the p95 residual would be {c['p95_abs_residual_after_slow_ms']:.1f} ms (this is what an offset that tracks drift could reach).", ""]
        L += ["Fraction of character keydown onsets landing within +/-X ms of (estimate + constant offset): " +
              ", ".join(f"{k.replace('within_','').replace('ms','')} ms: {100*v:.0f}%" for k, v in c["coverage_fixed_offset"].items()) + ".", "",
              "Crowding (another key press or release within +/-W of a character keydown): " + ", ".join(f"{k.split('within_')[1]}: {100*v:.0f}%" for k, v in c["crowding"].items()) + ".", ""]
        if vd.get("drift_significant") and vd.get("linear_corrected_p95_ms") is not None:
            L += [f"Drift is significant, so a constant offset is the wrong model. After removing a fitted *line* (offset + drift) the p95 residual is **{vd['linear_corrected_p95_ms']:.1f} ms**.", ""]
        L += [f"* **Fixed-offset clips** (cut at estimate + one constant offset, no re-anchoring): **{'SUPPORTED' if vd['fixed_offset_ok'] else 'NOT supported'}** at this precision.",
              f"* **Offset + linear drift clips** (cut at estimate + fitted line): **{'SUPPORTED' if vd['linear_corrected_ok'] else 'NOT supported'}** (p95 residual {f(vd['linear_corrected_p95_ms'],1)} ms vs {CRIT['fixed_p95_ms']} ms).",
              f"* **Onset-snapped clips** (find the onset near each estimate, cut around it, label from the matching keystroke): **{'SUPPORTED' if vd['snapped_ok'] else 'NOT supported'}** for the {100*vd['unambiguous_rate']:.0f}% of keystrokes with an unambiguous match; "
              "ambiguous and missed keystrokes should be dropped or reviewed rather than labelled by guess."]
    L += ["", "## 7. Suggested correction (NOT applied)", ""]
    if out["clip"]:
        rng = f"{min(meds):.1f} .. {max(meds):.1f}" if meds else "n/a"
        if d.get("drift_significant"):
            ref = d["t_start_s"]
            L += [f"The evidence supports a *line*, not a constant: offset(t) = **{d['offset_at_start_ms']:+.1f} ms** at t={ref:.0f} s, changing at **{d['drift_ppm']:+.0f} ppm** (detector-setting range of the level {rng} ms; see sections 3-5). ",
                  "`keystrokes.csv` and `metadata.json` are untouched. To write a *copy* with an extra `sample_index_corrected` column once you have decided to use it: ",
                  f"`python analysis/sync_analysis.py {sess['wav']} --write-corrected-csv {d['offset_at_start_ms']:.1f} --corrected-drift-ppm {d['drift_ppm']:.0f} --corrected-ref-s {ref:.1f}`"]
        else:
            L += [f"Evidence-backed constant offset: **{out['clip']['constant_offset_ms']:+.1f} ms** (keydown median; detector-setting range {rng} ms; see sections 3-5). "
                  "`keystrokes.csv` and `metadata.json` are untouched. To write a *copy* with an extra `sample_index_corrected` column using a value you have decided on: "
                  f"`python analysis/sync_analysis.py {sess['wav']} --write-corrected-csv <ms>`."]
    L += ["", "## Limitations", "",
          "* Onset detection is a heuristic; a quiet key, overlapping keys or room noise change what is found. The sensitivity table bounds this but does not remove it.",
          "* The keyboard event time is not the moment of finger contact: switch actuation and key travel add a physical delay of a few ms to tens of ms that varies key to key, and it is folded into 'latency' above.",
          "* This recording can show that offsets are stable or drifting; it cannot split input latency from physical delay without an independent reference (e.g. a loopback or a known acoustic source).",
          "", "## Files", ""] + [f"* `{p}`" for p in plots + files]
    open(os.path.join(od, "sync_report.md"), "w").write("\n".join(L) + "\n")


# ================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--out", help="output dir (default analysis_output/<session>)")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--ratio", type=float, default=4.0, help="onset threshold as a multiple of the rolling noise floor")
    ap.add_argument("--rise", type=float, default=0.2, help="fraction of floor->peak rise that defines the onset instant")
    ap.add_argument("--gate-ms", type=float, default=40.0, help="half-width of the matching window around the lag mode")
    ap.add_argument("--lag-window-ms", type=float, nargs=2, default=(-150.0, 400.0))
    ap.add_argument("--corrected-drift-ppm", type=float, default=0.0, help="with --write-corrected-csv: also apply a linear drift term")
    ap.add_argument("--corrected-ref-s", type=float, default=0.0, help="time (s in WAV) at which the offset equals --write-corrected-csv; drift is applied relative to it")
    ap.add_argument("--track-win-s", type=float, default=20.0, help="window for the time-tracked lag search")
    ap.add_argument("--bin-s", type=float, default=10.0)
    ap.add_argument("--write-corrected-csv", type=float, metavar="MS", help="opt-in: write keystrokes_corrected.csv (a copy) shifted by MS")
    a = ap.parse_args()
    a.gate = a.gate_ms / 1e3; a.lag_lo, a.lag_hi = a.lag_window_ms[0] / 1e3, a.lag_window_ms[1] / 1e3

    sess = find_session(a.session)
    print("WAV :", sess["wav"]); print("CSV :", sess["csv"]); print("META:", sess["meta"])
    audio_only = a.audio_only
    if sess["csv"] is None and not audio_only:
        print("\nNo keystrokes.csv was found next to this WAV, so keyboard/audio offsets cannot be computed.\nRunning transient detection only (--audio-only).")
        audio_only = True
    x, sr = load_wav(sess["wav"])
    ev = None if audio_only else load_events(sess["csv"])
    meta = json.load(open(sess["meta"])) if sess["meta"] else None
    if meta and meta.get("audio", {}).get("sample_rate_hz") not in (None, sr): sys.exit("metadata sample rate != WAV sample rate")
    od = a.out or os.path.join("analysis_output", sess["name"]); os.makedirs(od, exist_ok=True)

    out, env, det, A = analyse(x, sr, ev, meta, a)
    vd = verdict(out, a) if ev is not None else {}
    plots = plot_all(x, sr, ev, env, det, A if A else {}, out, od, sess["name"], aud_only=audio_only)
    files = []
    if ev is not None:
        M = A["M"]
        with open(os.path.join(od, "event_matches.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["event_index", "event_type", "key", "code", "category", "repeat", "time_ms", "t_rel_ms", "sample_index_est",
                        "status", "reason", "n_candidates", "onset_sample", "onset_strength_db", "offset_ms"])
            for i in range(len(ev["idx"])):
                j = M["chosen"][i]
                w.writerow([ev["idx"][i], "keydown" if ev["cls"][i] == "down" else "keyup", ev["key"][i], ev["code"][i], ev["cat"][i], int(ev["repeat"][i]),
                            ev["time_ms"][i], ev["t_rel_ms"][i], ev["sample_est"][i], M["status"][i], M["reason"][i], M["ncand"][i],
                            "" if j < 0 else int(round(det["onset"][j] * sr)), "" if j < 0 else f"{det['strength_db'][j]:.1f}",
                            "" if np.isnan(A["off_ms"][i]) else f"{A['off_ms'][i]:.3f}"])
        files.append("event_matches.csv")
        if a.write_corrected_csv is not None:
            rows = list(csv.DictReader(open(sess["csv"], newline="")))
            for r in rows:
                t_s = int(r["sample_index_est"]) / sr
                shift_ms = a.write_corrected_csv + a.corrected_drift_ppm * 1e-3 * (t_s - a.corrected_ref_s)   # ppm*1e-6*s*1e3 = ms
                r["sample_index_corrected"] = int(round(int(r["sample_index_est"]) + shift_ms * sr / 1000.0))
            with open(os.path.join(od, "keystrokes_corrected.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            files.append("keystrokes_corrected.csv"); print(f"wrote keystrokes_corrected.csv (+{a.write_corrected_csv} ms, {a.corrected_drift_ppm:+.0f} ppm from t={a.corrected_ref_s}s; original untouched)")
    out.pop("_dm", None)
    dump = dict(out); dump["verdict"] = vd
    json.dump(dump, open(os.path.join(od, "sync_stats.json"), "w"), indent=2, default=float)
    files.append("sync_stats.json")
    write_report(out if ev is not None else out, vd, files, sess, od, a, plots, audio_only)
    # restore the helper mask for printing
    print(f"\nOutputs in {od}/: sync_report.md, sync_stats.json, " + ", ".join(plots + files[:1]))
    if audio_only:
        print(f"{out['audio']['n_onsets']} acoustic onsets detected ({out['audio']['onset_rate_per_s']:.2f}/s)")
    else:
        for k in ("down/char", "up/char"):
            if k in out["match_summary"]: print(k, out["match_summary"][k])
        s = out["offset_stats_ms"].get("down/char/matched")
        if s: print(f"keydown offset (matched): median {s['median']:.1f} ms, mean {s['mean']:.1f}, sd {s['std']:.2f}")
        print("verdict:", json.dumps({k: v for k, v in vd.items() if k != 'criteria'}, default=float))


if __name__ == "__main__":
    main()
