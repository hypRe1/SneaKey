#!/usr/bin/env python3
"""
SneaKey: keystroke audio segmentation + dataset preparation (no model training).

    .venv/bin/python analysis/extract_keystroke_segments.py                 # session/ + analysis_output/session/ -> dataset/
    .venv/bin/python analysis/extract_keystroke_segments.py --session session --analysis analysis_output/session --out dataset

Inputs (read-only, hashed before/after to prove they are untouched):
    <session>/audio.wav, <session>/keystrokes.csv, <session>/metadata.json
    <analysis>/event_matches.csv, <analysis>/sync_stats.json      (from analysis/sync_analysis.py)

Label = physical key (`event.code`). Press (keydown) and release (keyup) are separate datasets.
Clips are cut from the original int16 samples, bit-exact: no filtering, normalisation or gain.
Each event is cut with 5 ms pre-onset and 25 / 45 / 75 ms post-onset (windows of 30 / 50 / 80 ms).

Alignment of each event (class-specific window around sample_index_est + measured lag):
    detected           exactly one onset within the tolerance, the analysis also matched it
    detected_resolved  exactly one onset within the tolerance, not claimed by any other event (analysis said ambiguous)
    uncertain          several onsets within the tolerance, or the onset is also the candidate of another event
    estimated          no onset detected; anchored at sample_index_est + measured lag
Only detected / detected_resolved, non-overlapping clips enter dataset/press|release/<code>/; every other clip goes
to dataset/excluded/... and every event (including repeats and out-of-bounds) is in manifest.csv with the reason.
"""
import argparse, csv, hashlib, json, math, os, shutil, sys, wave
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sync_analysis import band_envelope, detect_onsets   # same detector as the sync analysis

PRE_MS = 5.0
POST_MS = (25.0, 45.0, 75.0)          # -> 30 / 50 / 80 ms windows
INSEPARABLE_MS = 12.0                  # another event's sound this close to the onset cannot be separated
RING_BEFORE_MS = 15.0                  # a neighbour starting up to this long before the clip still rings into it
RISE_DB = 6.0                          # energy rise at an undetected neighbour's expected time that counts as audible
SAME_KEY_MS = 20.0                     # transients within this of the onset are the same keystroke: on the first real session,
                                       # unexplained transients were 2.5x over-represented at 15-20 ms (own tail) and at baseline from 20 ms
SENS_RATIO = 3.0                       # second, more sensitive onset pass (10 dB over floor) used ONLY to find contaminating sounds;
                                       # 6 dB fired 4.2/s in quiet gaps (24% spurious exclusions at 80 ms), 10 dB ~1.3/s
NOISE_PCTL = 95.0                      # residual-energy test: outside the same-keystroke zone a clean clip must be no louder than
                                       # this percentile of equally long background-noise segments (=> ~5% false exclusions)
LOOKAHEAD_MS = 300.0                   # horizon for the "time until first contaminating sound" statistic
MODIFIERS = {"ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight", "MetaLeft", "MetaRight", "CapsLock", "OSLeft", "OSRight"}
CLASS_DIR = {"keydown": "press", "keyup": "release"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def read_wav_int16(p):
    with wave.open(p) as w:
        if w.getsampwidth() != 2 or w.getcomptype() != "NONE": sys.exit("expected 16-bit PCM")
        if w.getnchannels() != 1: sys.exit("expected mono")
        sr, n = w.getframerate(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2").copy()
    return data, sr


def write_wav_int16(p, data, sr):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(data.astype("<i2").tobytes())


def expected_keys(code):
    """Characters a physical key can produce on US or UK layouts; used only to flag questionable labels."""
    if code.startswith("Key") and len(code) == 4: return {code[3].lower(), code[3]}
    sym = {"Digit1": "1!", "Digit2": "2@\"", "Digit3": "3#\u00a3", "Digit4": "4$", "Digit5": "5%", "Digit6": "6^", "Digit7": "7&",
           "Digit8": "8*", "Digit9": "9(", "Digit0": "0)", "Minus": "-_", "Equal": "=+", "Comma": ",<", "Period": ".>",
           "Slash": "/?", "Semicolon": ";:", "Quote": "'\"@", "BracketLeft": "[{", "BracketRight": "]}", "Backslash": "\\|#~", "Backquote": "`~\u00ac", "IntlBackslash": "\\|"}
    if code in sym: return set(sym[code])
    return {"Space": {" "}, "Backspace": {"Backspace"}, "Enter": {"Enter"}, "Tab": {"Tab"},
            "ShiftLeft": {"Shift"}, "ShiftRight": {"Shift"}, "CapsLock": {"CapsLock"}}.get(code)


def category(code):
    if code in MODIFIERS: return "modifier"
    if code == "Backspace": return "backspace"
    if code == "Space": return "space"
    if code.startswith("Key"): return "letter"
    if code.startswith("Digit"): return "digit"
    if expected_keys(code): return "punctuation"
    return "other"


# ====================================================================================================== main pipeline
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="session")
    ap.add_argument("--analysis", default="analysis_output/session")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--montage-per-key", type=int, default=6)
    a = ap.parse_args()

    src = {k: os.path.join(a.session, f) for k, f in (("wav", "audio.wav"), ("csv", "keystrokes.csv"), ("meta", "metadata.json"))}
    src.update(matches=os.path.join(a.analysis, "event_matches.csv"), stats=os.path.join(a.analysis, "sync_stats.json"))
    for p in src.values():
        if not os.path.exists(p): sys.exit(f"missing input: {p}")
    hashes_before = {k: sha256(p) for k, p in src.items()}

    pcm, sr = read_wav_int16(src["wav"])
    x = pcm.astype(np.float64) / 32768.0
    meta = json.load(open(src["meta"]))
    stats = json.load(open(src["stats"]))
    ks = list(csv.DictReader(open(src["csv"], newline="")))
    em = list(csv.DictReader(open(src["matches"], newline="")))
    if meta["audio"]["sample_rate_hz"] != sr or meta["audio"]["frames"] != len(pcm): sys.exit("metadata does not describe this WAV")
    if [r["event_index"] for r in ks] != [r["event_index"] for r in em] or any(r["sample_index_est"] != m["sample_index_est"] for r, m in zip(ks, em)):
        sys.exit("keystrokes.csv and event_matches.csv do not describe the same events")

    # ---- lags + tolerances: the medians/robust sigmas ALREADY measured by the sync analysis (not re-estimated)
    lag_ms = {"keydown": stats["offset_stats_ms"]["down/char/matched"]["median"], "keyup": stats["offset_stats_ms"]["up/char/matched"]["median"]}
    tol_ms = {c: max(8.0, 2.5 * stats["offset_stats_ms"][f"{'down' if c == 'keydown' else 'up'}/char/matched"]["mad_sigma"]) for c in lag_ms}
    ms = lambda v: int(round(v * sr / 1000.0))

    # ---- all acoustic onsets, with the detector settings the sync analysis used
    det_cfg = stats["audio"]["detector"]
    env = band_envelope(x, sr)
    det = detect_onsets(env, ratio=det_cfg["threshold_ratio_vs_rolling_floor"], rise_frac=det_cfg["onset_rise_fraction"])
    onsets = np.round(det["onset"] * sr).astype(np.int64)
    strength = det["strength_db"]
    det_s = detect_onsets(env, ratio=SENS_RATIO, rise_frac=det_cfg["onset_rise_fraction"])
    s_on = np.round(det_s["onset"] * sr).astype(np.int64); s_str = det_s["strength_db"]
    # measured false-alarm rate of the sensitive pass: onsets in gaps >= 60 ms from any logged event's expected sound
    exp_t = np.sort(np.array([int(r["sample_index_est"]) / sr + lag_ms[r["event_type"]] / 1000 for r in ks]))
    gaps = [(u + 0.06, v - 0.06) for u, v in zip(exp_t, exp_t[1:]) if v - u > 0.2]
    gap_s = sum(v - u for u, v in gaps)
    fa_rate = (sum(int(((det_s["onset"] >= u) & (det_s["onset"] < v)).sum()) for u, v in gaps) / gap_s) if gap_s > 0 else float("nan")
    pw = env["fast"] ** 2; noise_pw = float(np.median(pw))
    def peak_db(s0, s1): return 10 * math.log10(max(float(pw[max(0, s0):max(s0 + 1, s1)].max()), 1e-20) / noise_pw)
    regions = {f"w{int(PRE_MS + p)}": (SAME_KEY_MS, p) for p in POST_MS}          # tail region per window
    regions["pre"] = (-(PRE_MS + RING_BEFORE_MS), -INSEPARABLE_MS)                 # just before the clip (ringing in)
    NOISE_THR = {}
    for name, (r0, r1) in regions.items():
        L = ms(r1 - r0); vals = []
        for u, v in gaps:
            for st in range(int(u * sr), int(v * sr) - L, ms(5)): vals.append(peak_db(st, st + L))
        NOISE_THR[name] = {"region_ms": [r0, r1], "threshold_db": float(np.percentile(vals, NOISE_PCTL)) if vals else float("inf"), "n_noise_segments": len(vals)}
    FALSE_ALARM = {"gaps": len(gaps), "gap_seconds": round(gap_s, 2), "onsets_per_s_in_gaps": round(fa_rate, 3),
                   "p_spurious_exclusion": {f"w{int(PRE_MS + p)}": round(1 - math.exp(-fa_rate * max(0.0, (p - SAME_KEY_MS) + (RING_BEFORE_MS + PRE_MS - INSEPARABLE_MS)) / 1000), 3) for p in POST_MS},
                   "note": "gap onsets may also be real unlogged sounds (desk, mouse, breathing), so this is an upper bound on detector false alarms"}
    em_onsets = [int(r["onset_sample"]) for r in em if r["onset_sample"] != ""]
    repro = sum(1 for o in em_onsets if np.min(np.abs(onsets - o)) <= 1)
    print(f"detector reproduced {repro}/{len(em_onsets)} onsets referenced by event_matches.csv; {len(onsets)} onsets total")
    if repro != len(em_onsets): sys.exit("onset detector did not reproduce the analysis onsets - refusing to continue")

    e1, floor1, dt = env["e1"], env["floor1"], env["dt"]
    def peak_db(t0, t1):        # max band envelope in [t0, t1) samples, dB over the rolling floor
        i0, i1 = max(0, int(t0 / sr / dt)), min(len(e1), max(int(t0 / sr / dt) + 1, int(math.ceil(t1 / sr / dt))))
        if i1 <= i0: return float("nan")
        return 20 * math.log10(max(e1[i0:i1].max(), 1e-12) / max(np.median(floor1[i0:i1]), 1e-12))
    def rise_db(tau):            # energy rise at tau vs just before it: a NEW sound, not our own decay
        return peak_db(tau - ms(3), tau + ms(8)) - peak_db(tau - ms(10), tau - ms(3))

    # ---- events
    ev = []
    for r, m in zip(ks, em):
        e = dict(r)
        e.update(event_index=int(r["event_index"]), est=int(r["sample_index_est"]), is_repeat=r["repeat"] == "1",
                 analysis_status=m["status"], analysis_reason=m["reason"], analysis_onset=int(m["onset_sample"]) if m["onset_sample"] else None,
                 analysis_offset_ms=m["offset_ms"], category=category(r["code"]), pair=None)
        ev.append(e)
    open_down = {}
    for e in ev:                 # pair keydown <-> keyup by physical key
        if e["event_type"] == "keydown" and not e["is_repeat"]: open_down[e["code"]] = e
        elif e["event_type"] == "keyup" and e["code"] in open_down:
            d = open_down.pop(e["code"]); d["pair"] = e["event_index"]; e["pair"] = d["event_index"]

    # ---- alignment (two passes: candidates, then claims)
    for e in ev:
        c = e["event_type"]
        e["expected"] = e["est"] + ms(lag_ms[c])
        lo, hi = np.searchsorted(onsets, [e["expected"] - ms(tol_ms[c]), e["expected"] + ms(tol_ms[c]) + 1])
        e["near"] = list(range(lo, hi)) if not e["is_repeat"] else []
    claims = Counter(j for e in ev for j in e["near"])
    for e in ev:
        near = e["near"]
        if e["is_repeat"]:
            e.update(alignment="repeat", anchor=None, align_note="auto-repeat keydown: no new key sound")
        elif not near:
            e.update(alignment="estimated", anchor=e["expected"], align_note=f"no onset within +/-{tol_ms[e['event_type']]:.1f} ms; anchored at estimate + {lag_ms[e['event_type']]:.2f} ms")
        elif len(near) > 1:
            j = min(near, key=lambda j: abs(onsets[j] - e["expected"]))
            e.update(alignment="uncertain", anchor=int(onsets[j]), align_note=f"{len(near)} onsets within tolerance")
        else:
            j = near[0]
            if claims[j] > 1:
                others = [o["event_index"] for o in ev if o is not e and j in o["near"]]
                e.update(alignment="uncertain", anchor=int(onsets[j]), align_note=f"onset is also the candidate of event(s) {others}")
            else:
                same = e["analysis_status"] == "matched" and e["analysis_onset"] is not None and abs(e["analysis_onset"] - onsets[j]) <= 1
                e.update(alignment="detected" if same else "detected_resolved", anchor=int(onsets[j]),
                         align_note="" if same else f"analysis status '{e['analysis_status']}' ({e['analysis_reason'] or 'outside its wider gate'}); unique onset under the tighter class window")
        e["anchor_onset_idx"] = (min(e["near"], key=lambda j: abs(onsets[j] - e["expected"])) if e["near"] else None)
        e["has_sound"] = e["alignment"] in ("detected", "detected_resolved", "uncertain")
        e["onset_strength_db"] = float(strength[e["anchor_onset_idx"]]) if e["anchor_onset_idx"] is not None else None
        e["anchor_minus_est_ms"] = None if e["anchor"] is None else (e["anchor"] - e["est"]) * 1000.0 / sr

    # sound time of every event (detected onset, else expected) for neighbour checks
    sounding = sorted((e for e in ev if e["anchor"] is not None), key=lambda e: e["anchor"])
    s_times = np.array([e["anchor"] for e in sounding])
    assigned = {e["anchor_onset_idx"] for e in ev if e["has_sound"] and e["anchor_onset_idx"] is not None}

    # ---- neighbours of each event inside [anchor - pre - ring, anchor + lookahead]
    for e in ev:
        e["neigh"] = []
        if e["anchor"] is None: continue
        A = e["anchor"]
        lo, hi = np.searchsorted(s_times, [A - ms(PRE_MS + RING_BEFORE_MS), A + ms(LOOKAHEAD_MS) + 1])
        for o in sounding[lo:hi]:
            if o is e: continue
            tau = o["anchor"]; dist = (tau - A) * 1000.0 / sr
            rel = "own_release" if (e["event_type"] == "keydown" and o["event_index"] == e["pair"]) else \
                  "own_press" if (e["event_type"] == "keyup" and o["event_index"] == e["pair"]) else "neighbour"
            if abs(dist) < INSEPARABLE_MS: kind = "inseparable"
            elif o["has_sound"]: kind = "detected_sound"
            else:
                r = rise_db(tau)
                kind = "audible_undetected" if (r == r and r >= RISE_DB) else "silent"
            e["neigh"].append(dict(event_index=o["event_index"], code=o["code"], type=o["event_type"], rel=rel, dt_ms=round(dist, 2), kind=kind))
        # every transient from the sensitive pass: near the onset = same keystroke; elsewhere = some other sound
        e["secondary"] = []
        lo, hi = np.searchsorted(s_on, [A - ms(PRE_MS + RING_BEFORE_MS), A + ms(LOOKAHEAD_MS) + 1])
        for j in range(lo, hi):
            dtm = (s_on[j] - A) * 1000.0 / sr
            if abs(dtm) < 1.0: continue                                     # the anchor itself
            if -INSEPARABLE_MS < dtm < SAME_KEY_MS:
                e["secondary"].append(dict(dt_ms=round(dtm, 2), strength_db=round(float(s_str[j]), 1))); continue
            near_n = [n for n in e["neigh"] if abs(n["dt_ms"] - dtm) < INSEPARABLE_MS]
            if near_n:                                                      # explained by a logged event
                for n in near_n:
                    if n["kind"] == "silent": n["kind"] = "detected_sound_sensitive"   # it did make a sound after all
                continue
            e["neigh"].append(dict(event_index=-1, code="", type="", rel="neighbour", dt_ms=round(dtm, 2), kind="unattributed_transient",
                                   strength_db=round(float(s_str[j]), 1)))
        contam = [n for n in e["neigh"] if n["kind"] != "silent"]
        after = [n["dt_ms"] for n in contam if n["dt_ms"] > -INSEPARABLE_MS]
        e["prior_contamination"] = any(n["dt_ms"] <= -INSEPARABLE_MS for n in contam)
        e["free_after_ms"] = 0.0 if any(abs(n["dt_ms"]) < INSEPARABLE_MS for n in contam) else (min(after) if after else LOOKAHEAD_MS)

    # ====================================================================================== write clips + manifest
    if os.path.exists(a.out):
        if not os.path.exists(os.path.join(a.out, "manifest.csv")): sys.exit(f"{a.out} exists and is not a SneaKey dataset; refusing to overwrite")
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    session_id = meta.get("session_id", ""); folder = meta.get("folder", "")
    rows, written = [], []
    preserved = ["event_index", "event_type", "key", "code", "location", "time_ms", "event_time_ms", "t_rel_ms", "sample_index_est",
                 "repeat", "shift", "ctrl", "alt", "meta", "action", "target_char", "typed_char", "correct"]
    for e in ev:
        cls = CLASS_DIR[e["event_type"]]
        exp_keys = expected_keys(e["code"])
        label_note = "" if exp_keys is None or e["key"] in exp_keys else f"key {e['key']!r} unexpected for code {e['code']} (US/UK layout)"
        for post in POST_MS:
            W = int(PRE_MS + post)
            base = {k: e[k] for k in preserved}
            base.update(session=folder, session_id=session_id, sample_id=f"{folder}_ev{e['event_index']:05d}_w{W}",
                        dataset_class=cls, label=e["code"], category=e["category"], pair_event_index=e["pair"] if e["pair"] is not None else "",
                        analysis_status=e["analysis_status"], analysis_reason=e["analysis_reason"],
                        analysis_onset_sample=e["analysis_onset"] if e["analysis_onset"] is not None else "", analysis_offset_ms=e["analysis_offset_ms"],
                        alignment=e["alignment"], alignment_note=e["align_note"], anchor_sample="" if e["anchor"] is None else e["anchor"],
                        anchor_minus_est_ms="" if e["anchor_minus_est_ms"] is None else round(e["anchor_minus_est_ms"], 3),
                        onset_strength_db="" if e["onset_strength_db"] is None else round(e["onset_strength_db"], 1),
                        window_ms=W, pre_ms=PRE_MS, post_ms=post, label_note=label_note)
            reasons = []
            if e["anchor"] is None:
                reasons.append("repeat")
                base.update(clip_start_sample="", clip_end_sample="", n_frames=0)
                contam_here, secondary_here = [], []
            else:
                s0, s1 = e["anchor"] - ms(PRE_MS), e["anchor"] + ms(post)
                base.update(clip_start_sample=s0, clip_end_sample=s1, n_frames=s1 - s0)
                if s0 < 0 or s1 > len(pcm): reasons.append("out_of_bounds")
                if e["alignment"] not in ("detected", "detected_resolved"): reasons.append("alignment_" + e["alignment"])
                inwin = [n for n in e["neigh"] if -(PRE_MS + RING_BEFORE_MS) <= n["dt_ms"] <= post]
                contam_here = [n for n in inwin if n["kind"] != "silent"]
                for n in contam_here: reasons.append(f"{n['rel'] if n['rel'] != 'neighbour' else 'overlap'}_{n['kind']}")
                A = e["anchor"]
                tail_db = peak_db(A + ms(SAME_KEY_MS), A + ms(post)); pre_db = peak_db(A - ms(PRE_MS + RING_BEFORE_MS), A - ms(INSEPARABLE_MS))
                if tail_db > NOISE_THR[f"w{W}"]["threshold_db"]: reasons.append("overlap_energy_after_same_key_zone")
                if pre_db > NOISE_THR["pre"]["threshold_db"]: reasons.append("overlap_energy_before_onset")
                base.update(tail_peak_db=round(tail_db, 2), pre_peak_db=round(pre_db, 2))
                secondary_here = [s for s in e["secondary"] if s["dt_ms"] <= post]
                base["silent_neighbours_in_window"] = ";".join(f"{n['event_index']}:{n['code']}:{n['type']}@{n['dt_ms']}ms" for n in inwin if n["kind"] == "silent")
            base.setdefault("tail_peak_db", ""); base.setdefault("pre_peak_db", "")
            reasons = sorted(set(reasons))
            clean = not reasons
            base.update(clean=int(clean), exclusion_reasons=";".join(reasons),
                        contaminating_events=";".join((f"{n['event_index']}:{n['code']}:{n['type']}:{n['kind']}@{n['dt_ms']}ms" if n["event_index"] >= 0
                                                       else f"unattributed@{n['dt_ms']}ms/{n['strength_db']}dB") for n in contam_here),
                        secondary_transients=";".join(f"{s['dt_ms']}ms/{s['strength_db']}dB" for s in secondary_here),
                        n_secondary_transients=len(secondary_here),
                        free_after_ms="" if e["anchor"] is None else round(e["free_after_ms"], 2),
                        prior_contamination="" if e["anchor"] is None else int(e["prior_contamination"]))
            base.setdefault("silent_neighbours_in_window", "")
            path = ""
            if e["anchor"] is not None and "out_of_bounds" not in reasons:
                sub = os.path.join(cls, e["code"]) if clean else os.path.join("excluded", cls, e["code"])
                path = os.path.join(sub, f"{e['code']}_ev{e['event_index']:05d}_w{W}ms.wav")
                write_wav_int16(os.path.join(a.out, path), pcm[s0:s1], sr)
                written.append((path, s0, s1))
            base["clip_path"] = path
            rows.append(base)

    cols = ["sample_id", "session", "session_id", "dataset_class", "label", "category", "clean", "exclusion_reasons", "window_ms", "pre_ms", "post_ms",
            "clip_path", "clip_start_sample", "clip_end_sample", "n_frames", "alignment", "alignment_note", "anchor_sample", "anchor_minus_est_ms",
            "onset_strength_db", "analysis_status", "analysis_reason", "analysis_onset_sample", "analysis_offset_ms",
            "contaminating_events", "tail_peak_db", "pre_peak_db", "silent_neighbours_in_window", "secondary_transients", "n_secondary_transients", "free_after_ms", "prior_contamination",
            "pair_event_index", "label_note"] + preserved
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ====================================================================================== verification
    ver = verify(a.out, rows, written, pcm, sr)
    hashes_after = {k: sha256(p) for k, p in src.items()}
    ver["inputs_unchanged"] = hashes_before == hashes_after
    print("verification:", json.dumps({k: v for k, v in ver.items() if k != "errors"}), "errors:", ver["errors"][:5])

    # ====================================================================================== summary + analysis
    summary = summarise(ev, rows, lag_ms, tol_ms, sr, meta, src, hashes_before, ver, len(onsets))
    summary["method"]["contamination_detector_false_alarm"] = FALSE_ALARM
    summary["method"]["residual_energy_test"] = {"rule": f"peak 0.5 ms band (1.5-10 kHz) power in the tail region (+{SAME_KEY_MS} ms .. end of clip) and in the pre region must not exceed the {NOISE_PCTL:.0f}th percentile of equally long background-noise segments taken from gaps >= 60 ms from any logged key sound", "thresholds": NOISE_THR}
    summary["length_comparison"] = compare_lengths(ev, rows, pcm, sr, env, a.out)
    plot_counts(rows, a.out)
    for cls in ("press", "release"): montage(rows, pcm, sr, a.out, cls, a.montage_per_key)
    json.dump(summary, open(os.path.join(a.out, "dataset_summary.json"), "w"), indent=2, default=float)
    print_summary(summary)


def verify(out, rows, written, pcm, sr):
    errors, n = [], 0
    for path, s0, s1 in written:
        with wave.open(os.path.join(out, path)) as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, sr, "NONE"): errors.append(f"{path}: bad format")
            d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if not np.array_equal(d, pcm[s0:s1]): errors.append(f"{path}: samples differ from source")
        n += 1
    on_disk = set()
    for root, _, files in os.walk(out):
        for f in files:
            if f.endswith(".wav"): on_disk.add(os.path.relpath(os.path.join(root, f), out))
    in_manifest = {r["clip_path"] for r in rows if r["clip_path"]}
    if on_disk != in_manifest: errors.append(f"files vs manifest mismatch: {len(on_disk ^ in_manifest)}")
    for r in rows:
        if r["clip_path"]:
            parts = r["clip_path"].split(os.sep)
            expect = [r["dataset_class"], r["label"]] if r["clean"] else ["excluded", r["dataset_class"], r["label"]]
            if parts[:len(expect)] != expect: errors.append(f"{r['clip_path']}: wrong folder for label/clean flag")
            if r["clean"] and r["alignment"] not in ("detected", "detected_resolved"): errors.append(f"{r['sample_id']}: clean but not detected")
    ids = Counter(r["sample_id"] for r in rows)
    if any(v > 1 for v in ids.values()): errors.append("duplicate sample ids")
    return dict(clips_checked=n, bit_exact_and_pcm16=n - sum(1 for e in errors if "differ" in e or "bad format" in e), errors=errors,
                manifest_rows=len(rows), files_on_disk=len(on_disk))


def summarise(ev, rows, lag_ms, tol_ms, sr, meta, src, hashes, ver, n_onsets):
    S = {"session": meta.get("folder"), "session_id": meta.get("session_id"), "sample_rate_hz": sr, "inputs": src, "input_sha256": hashes,
         "verification": ver,
         "method": {"label": "event.code (physical key)", "pre_ms": PRE_MS, "post_ms": list(POST_MS), "windows_ms": [int(PRE_MS + p) for p in POST_MS],
                    "lag_ms_used": lag_ms, "alignment_tolerance_ms": tol_ms,
                    "lag_source": "median offsets of unambiguous matches in sync_stats.json (keydown: down/char/matched, keyup: up/char/matched); not re-estimated",
                    "onset_detector": "analysis/sync_analysis.py detect_onsets with the settings in sync_stats.json", "n_onsets": n_onsets,
                    "overlap_rule": f"another event (any key, press or release) whose sound time falls in [-{PRE_MS + RING_BEFORE_MS}, +post] ms of the onset contaminates the clip if it is within {INSEPARABLE_MS} ms (inseparable), has a detected onset, or produces a >= {RISE_DB} dB band-energy rise at its expected time; otherwise it is logged as a silent neighbour",
"secondary_transients": f"transients within -{INSEPARABLE_MS}..+{SAME_KEY_MS} ms of the onset are treated as part of the same keystroke (one keypress can produce several) and only counted; any transient (from a second, {20*math.log10(SENS_RATIO):.0f} dB-over-floor detection pass) elsewhere in the clip that no logged event explains is contamination ('unattributed_transient')",
                    "processing": "none - clips are bit-exact slices of the original int16 PCM",
                    "same_keystroke_zone_ms": [-INSEPARABLE_MS, SAME_KEY_MS], "sensitive_pass_db": round(20 * math.log10(SENS_RATIO), 1)}}
    for cls in ("press", "release"):
        typ = "keydown" if cls == "press" else "keyup"
        E = [e for e in ev if e["event_type"] == typ]
        al = Counter(e["alignment"] for e in E)
        S[cls] = {"events": len(E), "alignment": dict(al),
                  "alignment_vs_analysis_status": {f"{s}->{t}": c for (s, t), c in Counter((e["analysis_status"], e["alignment"]) for e in E).items()}}
        for post in POST_MS:
            W = int(PRE_MS + post)
            R = [r for r in rows if r["dataset_class"] == cls and r["window_ms"] == W]
            reasons = Counter(x for r in R for x in r["exclusion_reasons"].split(";") if x)
            per_key = defaultdict(lambda: {"clean": 0, "total": 0, "category": ""})
            for r in R:
                k = per_key[r["label"]]; k["total"] += 1; k["clean"] += r["clean"]; k["category"] = r["category"]
            S[cls][f"w{W}"] = {
                "clean": sum(r["clean"] for r in R),
                "excluded": sum(1 for r in R if not r["clean"]),
                "overlapping_any": sum(1 for r in R if r["contaminating_events"]),
                "overlapping_among_confident": sum(1 for r in R if r["alignment"] in ("detected", "detected_resolved") and r["contaminating_events"]),
                "exclusion_reasons": dict(reasons.most_common()),
                "clip_frames": dict(Counter(r["n_frames"] for r in R if r["clip_path"])),
                "clip_duration_ms": sorted({round(r["n_frames"] * 1000 / sr, 3) for r in R if r["clip_path"]}),
                "with_secondary_transients": sum(1 for r in R if r["clean"] and r["n_secondary_transients"]),
                "per_key": dict(sorted(per_key.items(), key=lambda kv: -kv[1]["clean"])),
            }
        conf = [e for e in E if e["alignment"] in ("detected", "detected_resolved")]
        fa = np.array([e["free_after_ms"] for e in conf])
        if len(fa):
            S[cls]["free_after_onset_ms"] = {"n": len(fa), "quantiles": {q: float(np.percentile(fa, q)) for q in (5, 10, 25, 50, 75, 90)},
                                             "prior_contamination": int(sum(e["prior_contamination"] for e in conf)),
                                             "meaning": "time from onset to the first contaminating sound of another event (0 = inseparable)"}
        ins = Counter()
        for e in conf:
            for n in e["neigh"]:
                if n["kind"] == "inseparable":
                    o = ev[n["event_index"]]
                    ins[f"{n['rel']}:{n['type']}:{'neighbour_has_detected_onset_elsewhere' if o['has_sound'] else 'neighbour_never_detected'}"] += 1
        S[cls]["inseparable_breakdown"] = dict(ins.most_common())
        S[cls]["anchor_minus_est_ms_confident"] = dict(zip(("p5", "p50", "p95"), np.percentile([e["anchor_minus_est_ms"] for e in conf], [5, 50, 95]).round(2).tolist())) if conf else None
    notes = Counter((r["code"], r["key"]) for r in rows if r["label_note"] and r["window_ms"] == int(PRE_MS + POST_MS[0]))
    uk = [e for e in ev if e["code"] == "Digit2" and e["key"] == '"'] + [e for e in ev if e["code"] == "Quote" and e["key"] == "@"]
    us = [e for e in ev if e["code"] == "Digit2" and e["key"] == "@"] + [e for e in ev if e["code"] == "Quote" and e["key"] == '"']
    S["questionable_labels"] = {"key_code_mismatch": [f"{c} produced {k!r} x{n}" for (c, k), n in notes.items()],
                                "inferred_layout": ("UK" if uk and not us else "US" if us and not uk else "unknown") + f" (UK-only evidence {len(uk)}, US-only {len(us)}); event.code is positional, so labels are layout-independent",
                                "wrong_character_keystrokes_kept": sum(1 for e in ev if e["event_type"] == "keydown" and e["action"] == "char" and e["correct"] == "0"),
                                "note_wrong_chars": "keystrokes that produced the wrong character are still correctly labelled by event.code (the physical key pressed)"}
    return S


# ====================================================================================================== 30 / 50 / 80 ms
def features(seg, sr, nfft=4096, bands=32, fmin=500.0, fmax=20000.0):
    # Flat window with 1 ms cosine ramps: a Hann window would suppress the onset (5 ms into the clip) more in
    # longer clips and bias the length comparison.
    w = seg * signal.windows.tukey(len(seg), alpha=min(1.0, 2 * 0.001 * sr / len(seg)))
    p = np.abs(np.fft.rfft(w, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.geomspace(fmin, min(fmax, sr / 2 * 0.98), bands + 1)
    return np.array([10 * np.log10(p[(f >= lo) & (f < hi)].mean() + 1e-20) for lo, hi in zip(edges[:-1], edges[1:])])


def pseudo_f(X, y):
    keys = np.unique(y); N, K = len(y), len(keys)
    mu = X.mean(0)
    sb = sum((y == k).sum() * np.sum((X[y == k].mean(0) - mu) ** 2) for k in keys)
    sw = sum(np.sum((X[y == k] - X[y == k].mean(0)) ** 2) for k in keys)
    return (sb / (K - 1)) / (sw / (N - K))


def separability(X, y, rng, perms=300):
    f = pseudo_f(X, y)
    null = np.array([pseudo_f(X, rng.permutation(y)) for _ in range(perms)])
    return {"F": float(f), "F_null_mean": float(null.mean()), "F_null_p95": float(np.percentile(null, 95)), "ratio_to_null": float(f / null.mean()),
            "p_perm": float((np.sum(null >= f) + 1) / (perms + 1))}


CUTS = {"w30 (-5..25)": (-5, 25), "w50 (-5..45)": (-5, 45), "w80 (-5..75)": (-5, 75), "tail 25..45": (25, 45), "tail 45..75": (45, 75), "pre-onset noise -25..-5": (-25, -5)}


def sep_block(evs, x, sr, rng, ms):
    y = np.array([e["code"] for e in evs])
    keep = np.isin(y, [k for k, n in Counter(y).items() if n >= 5])
    if keep.sum() < 20 or len(set(y[keep])) < 3: return {}
    sel = [e for e, k in zip(evs, keep) if k]
    out = {}
    for name, (s, t) in CUTS.items():
        X = np.array([features(x[e["anchor"] + ms(s): e["anchor"] + ms(t)], sr) for e in sel])
        out[name] = {"with_level": separability(X, y[keep], rng), "shape_only": separability(X - X.mean(1, keepdims=True), y[keep], rng)}
    out["_samples"] = int(keep.sum()); out["_keys"] = sorted(set(y[keep]))
    return out


def compare_lengths(ev, rows, pcm, sr, env, out):
    """Model-free comparison: clean yield, how soon neighbours arrive, how much own energy lies past 25/45 ms, and whether the
    extra audio adds key-discriminative spectral information (pseudo-F of between-key vs within-key variance, vs a label-permutation null)."""
    ms = lambda v: int(round(v * sr / 1000.0))
    rng = np.random.default_rng(0)
    res = {"note": "No classifier is trained. Separability = pseudo-F (between-key / within-key variance of 32 log-band spectral features), compared with a label-permutation null; >1 x null means key-dependent information."}
    e1, floor1, dt = env["e1"], env["floor1"], env["dt"]
    pw = env["fast"] ** 2                              # 0.5 ms band (1.5-10 kHz) power, full rate
    noise_pw = float(np.median(pw))                    # most of the recording is background noise
    x = pcm.astype(np.float64) / 32768.0
    plt.figure(figsize=(16, 9))
    for pi, cls in enumerate(("press", "release")):
        typ = "keydown" if cls == "press" else "keyup"
        R = {int(PRE_MS + p): {r["event_index"]: r for r in rows if r["dataset_class"] == cls and r["window_ms"] == int(PRE_MS + p)} for p in POST_MS}
        C = {"yield": {f"w{W}": sum(r["clean"] for r in R[W].values()) for W in R}}
        conf = [e for e in ev if e["event_type"] == typ and e["alignment"] in ("detected", "detected_resolved")]
        C["confident_events"] = len(conf)
        C["clean_fraction_of_confident"] = {f"w{W}": (sum(R[W][e["event_index"]]["clean"] for e in conf) / len(conf) if conf else None) for W in R}
        # energy of the OWN sound per segment, on samples clean at 80 ms (nothing else present up to +75 ms)
        c80 = [e for e in conf if R[80][e["event_index"]]["clean"]]
        segs = {"0-25ms": (0, 25), "25-45ms": (25, 45), "45-75ms": (45, 75)}
        energy = {k: [] for k in segs}; excess = {k: [] for k in segs}
        t_axis = np.arange(-5, 100)
        envs = []
        for e in c80:
            A = e["anchor"] / sr
            for k, (s, t) in segs.items():
                seg = pw[e["anchor"] + ms(s): e["anchor"] + ms(t)]
                energy[k].append(10 * np.log10(seg.mean() / noise_pw))
                excess[k].append(np.sum(np.maximum(seg - noise_pw, 0)))
            idx = (int(A / dt) + t_axis).clip(0, len(e1) - 1)
            envs.append(20 * np.log10(e1[idx] / floor1[idx]))
        if c80:
            tot = np.array(excess["0-25ms"]) + np.array(excess["25-45ms"]) + np.array(excess["45-75ms"])
            C["own_sound_by_segment_clean_at_80ms"] = {
                "n": len(c80),
                "median_level_db_over_floor": {k: float(np.median(v)) for k, v in energy.items()},
                "median_share_of_excess_energy": {k: float(np.median(np.array(excess[k]) / np.maximum(tot, 1e-12))) for k in segs},
                "note": "mean band (1.5-10 kHz) power of each segment relative to the recording's median noise power; ~0 dB = indistinguishable from background noise"}
        # separability on a FIXED sample set (clean at 80 ms) truncated to each length, plus tail-only segments
        sep = sep_block(c80, x, sr, rng, ms)
        C["separability_fixed_set_excluding_Space"] = sep_block([e for e in c80 if e["code"] != "Space"], x, sr, rng, ms)
        y = np.array([e["code"] for e in c80]); keep = np.isin(y, [k for k, n in Counter(y).items() if n >= 5])
        if False:
            cut = {"w30 (-5..25)": (-5, 25), "w50 (-5..45)": (-5, 45), "w80 (-5..75)": (-5, 75), "tail 25..45": (25, 45), "tail 45..75": (45, 75), "pre-onset noise -25..-5": (-25, -5)}
            sel = [e for e, k in zip(c80, keep) if k]
            for name, (s, t) in cut.items():
                X = np.array([features(x[e["anchor"] + ms(s): e["anchor"] + ms(t)], sr) for e in sel])
                Xs = X - X.mean(1, keepdims=True)      # spectral shape only (level removed)
                sep[name] = {"with_level": separability(X, y[keep], rng), "shape_only": separability(Xs, y[keep], rng)}
            sep["_samples"] = int(keep.sum()); sep["_keys"] = sorted(set(y[keep]))
        C["separability_fixed_set"] = sep
        # separability on each length's OWN clean set (more samples at shorter windows)
        own = {}
        for W in R:
            ids = [e for e in conf if R[W][e["event_index"]]["clean"]]
            yy = np.array([e["code"] for e in ids]); kk = np.isin(yy, [k for k, n in Counter(yy).items() if n >= 5])
            if kk.sum() >= 20 and len(set(yy[kk])) >= 3:
                X = np.array([features(x[e["anchor"] - ms(PRE_MS): e["anchor"] + ms(W - PRE_MS)], sr) for e, k in zip(ids, kk) if k])
                own[f"w{W}"] = {"n": int(kk.sum()), "keys": len(set(yy[kk])), "shape_only": separability(X - X.mean(1, keepdims=True), yy[kk], rng)}
        C["separability_own_clean_set"] = own
        fa = np.array([e["free_after_ms"] for e in conf])
        C["fraction_of_confident_with_contamination_by"] = {f"{t}ms": float(np.mean(fa < t)) for t in (25, 45, 75)} if len(fa) else {}
        res[cls] = C

        # plots: envelope, hazard, separability
        ax = plt.subplot(2, 3, pi * 3 + 1)
        if envs:
            E = np.array(envs)
            ax.fill_between(t_axis, np.percentile(E, 25, 0), np.percentile(E, 75, 0), alpha=0.3, color="#c0392b")
            ax.plot(t_axis, np.median(E, 0), color="#c0392b", label=f"median (n={len(E)} clean@80)")
        for v in (25, 45, 75): ax.axvline(v, color="k", ls=":", lw=0.8)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{cls}: own key sound, dB over noise floor", fontsize=10); ax.set_xlabel("ms after onset"); ax.legend(fontsize=8)
        ax = plt.subplot(2, 3, pi * 3 + 2)
        if len(fa):
            tt = np.arange(0, 151)
            ax.plot(tt, [np.mean(fa < t) for t in tt], color="#2980b9")
            for v in (25, 45, 75): ax.axvline(v, color="k", ls=":", lw=0.8)
            ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_title(f"{cls}: fraction of confident events already\ncontaminated by another key's sound", fontsize=10); ax.set_xlabel("ms after onset")
        ax = plt.subplot(2, 3, pi * 3 + 3)
        if sep:
            names = [n for n in sep if not n.startswith("_")]
            ax.bar(np.arange(len(names)) - 0.2, [sep[n]["with_level"]["ratio_to_null"] for n in names], 0.4, label="with level")
            ax.bar(np.arange(len(names)) + 0.2, [sep[n]["shape_only"]["ratio_to_null"] for n in names], 0.4, label="shape only")
            ax.axhline(1, color="k", lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.legend(fontsize=8)
        ax.set_title(f"{cls}: key separability (pseudo-F / permutation null)\nsame clean@80 samples, n={sep.get('_samples', 0)}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "length_comparison.png"), dpi=120); plt.close()
    return res


# ====================================================================================================== plots
def plot_counts(rows, out):
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    for ax, cls in zip(axs, ("press", "release")):
        R = [r for r in rows if r["dataset_class"] == cls]
        keys = [k for k, _ in Counter(r["label"] for r in R if r["window_ms"] == 30).most_common()]
        xs = np.arange(len(keys))
        tot = [sum(1 for r in R if r["label"] == k and r["window_ms"] == 30) for k in keys]
        ax.bar(xs, tot, color="#ddd", label="all events")
        for off, W, col in ((-0.27, 30, "#27ae60"), (0, 50, "#2980b9"), (0.27, 80, "#8e44ad")):
            ax.bar(xs + off, [sum(r["clean"] for r in R if r["label"] == k and r["window_ms"] == W) for k in keys], 0.25, color=col, label=f"clean w{W}")
        ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=8); ax.set_title(f"{cls}: samples per physical key"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out, "counts_per_key.png"), dpi=120); plt.close()


def montage(rows, pcm, sr, out, cls, per_key):
    R = [r for r in rows if r["dataset_class"] == cls and r["window_ms"] == 80 and r["clean"]]
    if not R:
        R = [r for r in rows if r["dataset_class"] == cls and r["window_ms"] == 30 and r["clean"]]
    if not R: return
    by = defaultdict(list)
    for r in R: by[r["label"]].append(r)
    keys = sorted(by, key=lambda k: -len(by[k]))
    rng = np.random.default_rng(1)
    specs = {}
    for k in keys:
        pick = by[k] if len(by[k]) <= per_key else [by[k][i] for i in sorted(rng.choice(len(by[k]), per_key, replace=False))]
        for r in pick:
            seg = pcm[r["clip_start_sample"]: r["clip_end_sample"]].astype(np.float64) / 32768
            f, t, S = signal.spectrogram(seg, sr, nperseg=128, noverlap=120, window="hann")
            specs[(k, r["event_index"])] = (f, (t * 1000 - PRE_MS), 10 * np.log10(S + 1e-14), r)
    allv = np.concatenate([v[2].ravel() for v in specs.values()])
    vmin, vmax = np.percentile(allv, 20), np.percentile(allv, 99.5)
    fig, axs = plt.subplots(len(keys), per_key, figsize=(per_key * 2.1, len(keys) * 1.15 + 0.6), squeeze=False)
    for i, k in enumerate(keys):
        items = [v for (kk, _), v in specs.items() if kk == k]
        for j in range(per_key):
            ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if j >= len(items): ax.axis("off"); continue
            f, t, S, r = items[j]
            ax.pcolormesh(t, f / 1000, S, vmin=vmin, vmax=vmax, cmap="magma", shading="auto")
            for v in (25, 45): ax.axvline(v, color="w", lw=0.5, ls=":")
            ax.set_ylim(0, 20)
            ax.text(0.02, 0.97, f"ev{r['event_index']} {r['key'] if r['key'] != ' ' else '␣'}", transform=ax.transAxes, fontsize=6, color="w", va="top")
            if j == 0: ax.set_ylabel(f"{k}\n(n={len(by[k])})", fontsize=7, rotation=0, ha="right", va="center")
    W = R[0]["window_ms"]
    fig.suptitle(f"{cls}: clean w{W} clips, 0-20 kHz, common colour scale; dotted lines at +25 / +45 ms (ends of the 30 / 50 ms windows)", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (len(keys) * 1.15 + 0.6))); plt.savefig(os.path.join(out, f"montage_{cls}.png"), dpi=110); plt.close()


def print_summary(S):
    print(f"\nlags used: {S['method']['lag_ms_used']}  tolerances: {S['method']['alignment_tolerance_ms']}")
    for cls in ("press", "release"):
        c = S[cls]
        print(f"\n[{cls}] {c['events']} events, alignment {c['alignment']}")
        for W in (30, 50, 80):
            w = c[f"w{W}"]
            print(f"  w{W}: clean {w['clean']}, excluded {w['excluded']}, overlapping(confident) {w['overlapping_among_confident']}, top reasons {dict(list(w['exclusion_reasons'].items())[:5])}")
        if "free_after_onset_ms" in c: print("  free time after onset (ms) quantiles:", {k: round(v, 1) for k, v in c["free_after_onset_ms"]["quantiles"].items()})
        L = S["length_comparison"][cls]
        if "own_sound_by_segment_clean_at_80ms" in L: print("  own sound by segment:", json.dumps(L["own_sound_by_segment_clean_at_80ms"]["median_level_db_over_floor"]), "share:", json.dumps({k: round(v, 3) for k, v in L["own_sound_by_segment_clean_at_80ms"]["median_share_of_excess_energy"].items()}))
        for n, v in L.get("separability_fixed_set", {}).items():
            if not n.startswith("_"): print(f"  sep {n:24s} level {v['with_level']['ratio_to_null']:.2f}x (p={v['with_level']['p_perm']:.3f})  shape {v['shape_only']['ratio_to_null']:.2f}x (p={v['shape_only']['p_perm']:.3f})")
        for n, v in L.get("separability_fixed_set_excluding_Space", {}).items():
            if not n.startswith("_"): print(f"  sep-noSpace {n:18s} level {v['with_level']['ratio_to_null']:.2f}x (p={v['with_level']['p_perm']:.3f})  shape {v['shape_only']['ratio_to_null']:.2f}x (p={v['shape_only']['p_perm']:.3f})")
        print("  inseparable breakdown:", S[cls].get("inseparable_breakdown"))
        for n, v in L.get("separability_own_clean_set", {}).items():
            print(f"  own-set {n}: n={v['n']} keys={v['keys']} shape {v['shape_only']['ratio_to_null']:.2f}x (p={v['shape_only']['p_perm']:.3f})")
    print("\nquestionable labels:", S["questionable_labels"])


if __name__ == "__main__":
    main()
