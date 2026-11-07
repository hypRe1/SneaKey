"""Validates analysis/extract_keystroke_segments.py on a synthetic session with KNOWN click positions.
Run: .venv/bin/python -m unittest tests/test_extract_segments.py -v
"""
import csv, hashlib, json, os, shutil, subprocess, sys, tempfile, unittest, wave
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_sync_analysis import make_session, SR   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LAG_DOWN, LAG_UP = 0.038, 0.041                   # make_session defaults: true click = estimate + lag (+3 ms jitter)


def sha(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()


class ExtractSegments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="sneakey-seg-")
        base, _ = make_session(cls.d, "syn", seed=7)
        sess = os.path.join(cls.d, "session"); os.makedirs(sess)
        for suf in ("audio.wav", "keystrokes.csv"): shutil.move(f"{base}_{suf}", os.path.join(sess, suf))
        with wave.open(os.path.join(sess, "audio.wav")) as w: frames = w.getnframes()
        json.dump({"folder": "session_syn", "session_id": "syn-1", "audio": {"sample_rate_hz": SR, "frames": frames}},
                  open(os.path.join(sess, "metadata.json"), "w"))
        cls.ana = os.path.join(cls.d, "analysis")
        for cmd in ([PY, os.path.join(ROOT, "analysis", "sync_analysis.py"), sess, "--out", cls.ana],
                    [PY, os.path.join(ROOT, "analysis", "extract_keystroke_segments.py"), "--session", sess, "--analysis", cls.ana,
                     "--out", os.path.join(cls.d, "dataset")]):
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=cls.d)
            assert r.returncode == 0, r.stdout + r.stderr
        cls.sess = sess
        cls.out = os.path.join(cls.d, "dataset")
        cls.rows = list(csv.DictReader(open(os.path.join(cls.out, "manifest.csv"))))
        cls.summary = json.load(open(os.path.join(cls.out, "dataset_summary.json")))
        cls.ks = list(csv.DictReader(open(os.path.join(sess, "keystrokes.csv"))))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_layout_labels_and_bit_exact_clips(self):
        v = self.summary["verification"]
        self.assertEqual(v["errors"], []); self.assertTrue(v["inputs_unchanged"])
        self.assertEqual(v["clips_checked"], v["bit_exact_and_pcm16"])
        with wave.open(os.path.join(self.sess, "audio.wav")) as w: src = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        clean = [r for r in self.rows if r["clean"] == "1"]
        self.assertGreater(len(clean), 200)
        for r in clean[:50]:
            parts = r["clip_path"].split(os.sep)
            self.assertEqual(parts[:2], [r["dataset_class"], r["label"]])               # dataset/press/<code>/...
            self.assertEqual(r["label"], r["code"])                                     # physical key, not character
            with wave.open(os.path.join(self.out, r["clip_path"])) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, SR))
                d = np.frombuffer(w.readframes(w.getnframes()), "<i2")
            self.assertTrue(np.array_equal(d, src[int(r["clip_start_sample"]): int(r["clip_end_sample"])]))
            self.assertEqual(len(d), round((float(r["pre_ms"]) + float(r["post_ms"])) * SR / 1000))

    def test_every_event_preserved_with_ids_and_timestamps(self):
        self.assertEqual(len(self.rows), 3 * len(self.ks))
        by = {(r["event_index"], r["window_ms"]): r for r in self.rows}
        for k in self.ks:
            for W in ("30", "50", "80"):
                r = by[(k["event_index"], W)]
                for col in ("time_ms", "t_rel_ms", "sample_index_est", "key", "code", "event_type"): self.assertEqual(r[col], k[col])
        rep = [r for r in self.rows if r["repeat"] == "1"]
        self.assertTrue(rep and all(r["clean"] == "0" and "repeat" in r["exclusion_reasons"] for r in rep))

    def test_anchors_hit_the_true_clicks(self):
        for r in self.rows:
            if r["clean"] != "1": continue
            lag = LAG_DOWN if r["event_type"] == "keydown" else LAG_UP
            err_ms = (int(r["anchor_sample"]) - int(r["sample_index_est"]) - lag * SR) / SR * 1000
            self.assertLess(abs(err_ms), 12, r["sample_id"])                            # injected jitter is 3 ms sd

    def test_clean_clips_contain_no_other_key_sound(self):
        """Ground truth: every press clicks at est+38 ms and every release at est+41 ms (except injected silent presses).
        No clip marked clean may contain another event's click in its window (or ringing in from just before)."""
        truth = sorted((int(k["sample_index_est"]) + round((LAG_DOWN if k["event_type"] == "keydown" else LAG_UP) * SR), k["event_index"])
                       for k in self.ks if k["repeat"] == "0")
        t = np.array([a for a, _ in truth]); ids = [b for _, b in truth]
        bad = []
        for r in self.rows:
            if r["clean"] != "1": continue
            A = int(r["anchor_sample"]); lo = A - int(0.020 * SR); hi = int(r["clip_end_sample"])
            for j in range(*np.searchsorted(t, [lo, hi])):
                if ids[j] != r["event_index"]: bad.append((r["sample_id"], ids[j], (t[j] - A) / SR * 1000))
        # injected silent presses still have a 'true' time here, so allow a handful of hits that are in fact silent
        self.assertLess(len(bad), 0.02 * sum(r["clean"] == "1" for r in self.rows), bad[:10])

    def test_overlaps_are_flagged_not_dropped(self):
        ov = [r for r in self.rows if r["contaminating_events"]]
        self.assertTrue(ov)
        self.assertTrue(all(r["clean"] == "0" for r in ov))
        self.assertTrue(all(r["clip_path"].startswith("excluded") for r in ov if r["clip_path"]))
        w80 = self.summary["press"]["w80"]["clean"]; w30 = self.summary["press"]["w30"]["clean"]
        self.assertLessEqual(w80, w30)                                                  # longer windows can only lose samples
        for k in ("press", "release"):
            for W in ("w30", "w50", "w80"): self.assertIn("per_key", self.summary[k][W])
        self.assertIn("length_comparison", self.summary)


if __name__ == "__main__":
    unittest.main()
