"""Validates analysis/sync_analysis.py against synthetic sessions with KNOWN ground truth.
Run: .venv/bin/python -m unittest tests/test_sync_analysis.py -v
"""
import csv, json, os, subprocess, sys, tempfile, unittest, wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "analysis", "sync_analysis.py")
SR = 48000


def click(rng, amp, tau=0.004, n=int(0.05 * SR)):
    t = np.arange(n) / SR
    return amp * rng.standard_normal(n) * np.exp(-t / tau)


def make_session(d, name, *, dur=120.0, lag_down=0.038, lag_up=0.041, drift_ppm=0.0, jitter=0.003, miss=0.05,
                 unrelated=False, seed=1, with_csv=True):
    """Write <name>_audio.wav / _keystrokes.csv. est timestamps are the 'claimed' positions; true clicks sit lag later."""
    rng = np.random.default_rng(seed)
    x = 0.003 * rng.standard_normal(int(dur * SR)) + 0.002 * np.sin(2 * np.pi * 50 * np.arange(int(dur * SR)) / SR)
    rows, T, idx, truth = [], 2.0, 0, []
    while T < dur - 3:
        hold = max(0.05, rng.normal(0.09, 0.02))
        key = rng.choice(list("abcdefghijklmnopqrstuvwxyz "))
        is_miss = rng.random() < miss
        shift = (T * drift_ppm * 1e-6)
        on_d = T + lag_down + shift + rng.normal(0, jitter)
        on_u = T + hold + lag_up + shift + rng.normal(0, jitter)
        if unrelated:
            on_d, on_u = rng.uniform(1, dur - 1), rng.uniform(1, dur - 1)
        for on, amp in ((on_d, 0.0 if is_miss else rng.uniform(0.25, 0.6)), (on_u, rng.uniform(0.08, 0.2))):
            i0 = int(on * SR); c = click(rng, amp); x[i0:i0 + len(c)] += c[: len(x) - i0]
        for typ, tt in (("keydown", T), ("keyup", T + hold)):
            rows.append(dict(event_index=idx, event_type=typ, key=key, code="Key" + key.upper(), location=0,
                             time_ms=100000 + tt * 1000, event_time_ms=100000 + tt * 1000, t_rel_ms=tt * 1000,
                             sample_index_est=int(round(tt * SR)), repeat=0, shift=0, ctrl=0, alt=0, meta=0,
                             action="char" if typ == "keydown" else "", target_char=key, typed_char=key if typ == "keydown" else "", correct=1,
                             cursor_before=0, cursor_after=0))
            idx += 1
        truth.append(dict(T=T, missed=is_miss))
        T += rng.lognormal(np.log(0.19), 0.35)
    # a held key: auto-repeat keydowns that make NO sound must be excluded, not counted as misses
    for k in range(5):
        tt = 30 + k * 0.03
        rows.append(dict(event_index=idx, event_type="keydown", key="x", code="KeyX", location=0, time_ms=100000 + tt * 1000, event_time_ms=0,
                         t_rel_ms=tt * 1000, sample_index_est=int(tt * SR), repeat=1, shift=0, ctrl=0, alt=0, meta=0, action="char",
                         target_char="", typed_char="x", correct=0, cursor_before=0, cursor_after=0)); idx += 1
    rows.sort(key=lambda r: r["sample_index_est"])
    for i, r in enumerate(rows): r["event_index"] = i
    base = os.path.join(d, name)
    with wave.open(base + "_audio.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes())
    if with_csv:
        with open(base + "_keystrokes.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return base, truth


def run(base, out, *extra):
    r = subprocess.run([sys.executable, SCRIPT, base + "_audio.wav", "--out", out, *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(os.path.join(out, "sync_stats.json")) as fh: return json.load(fh), r.stdout


class SyncAnalysis(unittest.TestCase):
    def test_recovers_known_latency_and_flags_misses(self):
        with tempfile.TemporaryDirectory() as d:
            base, truth = make_session(d, "session_a", drift_ppm=0.0)
            st, _ = run(base, os.path.join(d, "o"))
            dn = st["offset_stats_ms"]["down/char/matched"]
            self.assertAlmostEqual(dn["median"], 38.0, delta=1.5)
            self.assertLess(dn["std"], 5.0)                                     # injected jitter 3 ms
            up = st["offset_stats_ms"]["up/char/matched"]
            self.assertAlmostEqual(up["median"], 41.0, delta=2.5)               # release classed separately
            self.assertAlmostEqual(st["press_vs_release"]["up_minus_down_ms"], 3.0, delta=3.0)
            self.assertTrue(st["lag_search"]["down"]["significant"])
            self.assertTrue(st["lag_search"]["down"]["global_significant"])
            ms = st["match_summary"]["down/char"]
            n_true_miss = sum(t["missed"] for t in truth)
            self.assertGreaterEqual(ms["missed"], n_true_miss * 0.7)            # injected silent presses show up as missed
            self.assertLessEqual(ms["missed"], n_true_miss + 8)
            self.assertEqual(ms["eligible"], len(truth))                        # repeats excluded
            self.assertLess(abs(st["decomposition"]["drift_ppm"]), 150)
            self.assertTrue(st["verdict"]["fixed_offset_ok"])
            self.assertFalse(st["decomposition"]["drift_significant"])
            self.assertTrue(os.path.exists(os.path.join(d, "o", "sync_report.md")))

    def test_detects_clock_drift_separately_from_constant_latency(self):
        with tempfile.TemporaryDirectory() as d:
            base, _ = make_session(d, "session_b", drift_ppm=500.0, seed=2)     # +60 ms accumulated over 120 s
            st, _ = run(base, os.path.join(d, "o"))
            dec = st["decomposition"]
            ls = st["lag_search"]["down"]
            self.assertTrue(ls["significant"])                                   # tracked windows find it even though drift smears the global peak
            lo, hi = ls["tracked_lag_range_ms"]
            self.assertAlmostEqual(lo, 38 + 500e-6 * 15 * 1e3, delta=8); self.assertAlmostEqual(hi, 38 + 500e-6 * 105 * 1e3, delta=8)
            self.assertAlmostEqual(dec["drift_ppm"], 500.0, delta=120)
            self.assertGreater(dec["drift_ci95_ppm"][0], 0)                      # CI excludes zero
            self.assertGreater(dec["accumulated_drift_over_events_ms"], 40)
            self.assertGreater(dec["std_slow_ms"], 5)                            # slow component is visible
            self.assertLess(dec["std_fast_ms"], 6)                               # fast jitter still ~injected 3 ms
            self.assertFalse(st["verdict"]["fixed_offset_ok"])                   # constant offset alone no longer suffices
            self.assertTrue(dec["drift_significant"])
            self.assertAlmostEqual(dec["offset_at_start_ms"], 38 + 500e-6 * dec["t_start_s"] * 1e3, delta=3)
            self.assertTrue(st["verdict"]["linear_corrected_ok"])                # offset + drift line IS good enough
            self.assertLess(dec["p95_abs_residual_after_linear_ms"], 8)

    def test_no_relationship_is_reported_not_forced(self):
        with tempfile.TemporaryDirectory() as d:
            base, _ = make_session(d, "session_c", unrelated=True, seed=3)
            st, _ = run(base, os.path.join(d, "o"))
            self.assertFalse(st["lag_search"]["down"]["significant"])
            self.assertFalse(st["lag_search"]["down"]["global_significant"])
            self.assertFalse(st["verdict"]["snapped_ok"])
            self.assertFalse(st["verdict"]["fixed_offset_ok"])

    def test_audio_only_when_csv_missing_and_inputs_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            base, _ = make_session(d, "session_d", with_csv=False)
            st, out = run(base, os.path.join(d, "o"))
            self.assertIn("cannot be computed", out)
            self.assertNotIn("offset_stats_ms", st)
            self.assertGreater(st["audio"]["n_onsets"], 300)

    def test_corrected_csv_is_opt_in_copy(self):
        with tempfile.TemporaryDirectory() as d:
            base, _ = make_session(d, "session_e", seed=4)
            with open(base + "_keystrokes.csv") as fh: before = fh.read()
            run(base, os.path.join(d, "o1"))
            self.assertFalse(os.path.exists(os.path.join(d, "o1", "keystrokes_corrected.csv")))   # never automatic
            run(base, os.path.join(d, "o2"), "--write-corrected-csv", "38")
            with open(base + "_keystrokes.csv") as fh: self.assertEqual(before, fh.read())                        # original untouched
            with open(os.path.join(d, "o2", "keystrokes_corrected.csv")) as fh: r = next(csv.DictReader(fh))
            self.assertEqual(int(r["sample_index_corrected"]) - int(r["sample_index_est"]), round(0.038 * SR))
            run(base, os.path.join(d, "o3"), "--write-corrected-csv", "40", "--corrected-drift-ppm", "500", "--corrected-ref-s", "10")
            with open(os.path.join(d, "o3", "keystrokes_corrected.csv")) as fh:
                rows = list(csv.DictReader(fh))
            r10 = min(rows, key=lambda q: abs(int(q["sample_index_est"]) - 10 * SR))
            r70 = min(rows, key=lambda q: abs(int(q["sample_index_est"]) - 70 * SR))
            sh = lambda q: (int(q["sample_index_corrected"]) - int(q["sample_index_est"])) / SR * 1e3
            self.assertAlmostEqual(sh(r10), 40.0 + 500e-3 * (int(r10["sample_index_est"]) / SR - 10), delta=0.05)
            self.assertAlmostEqual(sh(r70) - sh(r10), 500e-3 * (int(r70["sample_index_est"]) - int(r10["sample_index_est"])) / SR, delta=0.05)


if __name__ == "__main__":
    unittest.main()
