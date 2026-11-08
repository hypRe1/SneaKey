"""
Phone recording: clock mapping across devices, and labelling of phone audio.

    python -m unittest tests.test_phone -v
"""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SNEAKEY_DATA", tempfile.mkdtemp(prefix="sneakey-test-"))
sys.path.insert(0, str(ROOT))

from sneakey import phone  # noqa: E402
from sneakey.groundtruth import label_events  # noqa: E402

OLD_SESSION = Path(os.environ.get("SNEAKEY_ORIGINAL", ROOT.parent / "SneaKey")) / "recordings" / "session_20261107_182727"


def simulated_devices(seed=0, seconds=120):
    """Laptop and phone clocks related to the server clock, pings with random asymmetric Wi-Fi delays, and the
    true audio sample of each laptop event. Returns (recording, laptop_pings, event_times, true_samples)."""
    rng = np.random.default_rng(seed)
    laptop_to_server = lambda t: t + 1_234_000.0                       # laptop page clock: fixed offset
    phone_to_server = lambda t: t * (1 - 40e-6) - 55_000.0              # phone page clock: offset, runs 40 ppm slow
    server_to_phone = lambda s: (s + 55_000.0) / (1 - 40e-6)
    frames_per_ms = 48 * (1 + 250e-6)                                   # phone audio clock runs 250 ppm fast
    frame_at = lambda phone_t: 1_000_000 + frames_per_ms * phone_t

    def pings(to_server, start, end, every=1000):
        out = []
        for t0 in np.arange(start, end, every):
            up, down = rng.uniform(2, 60), rng.uniform(2, 20)
            out.append({"t0": float(t0), "server": float(to_server(t0 + up)), "t1": float(t0 + up + down)})
        return out

    phone_start = 10_000.0
    rec = phone.Recording()
    first_frame = int(frame_at(phone_start))
    total = int(seconds * 48_000 * (1 + 250e-6))
    for start in range(first_frame, first_frame + total, 24_000):
        rec.chunks[start] = np.zeros(min(24_000, first_frame + total - start), np.int16)
    perf = np.arange(phone_start, phone_start + seconds * 1000, 250.0)
    rec.sync = {"sample_rate": 48000, "pings": pings(lambda t: phone_to_server(t), phone_start, phone_start + seconds * 1000),
                "clock_pairs": [{"perf": float(t), "frame": float(frame_at(t) + rng.normal(0, 20))} for t in perf]}
    rec.state = "done"
    # Laptop events during the recording, in laptop page time.
    server_start = phone_to_server(phone_start)
    laptop_t0 = server_start - 1_234_000.0
    event_times = laptop_t0 + np.sort(rng.uniform(2000, seconds * 1000 - 2000, 300))
    laptop_pings = pings(laptop_to_server, laptop_t0, laptop_t0 + seconds * 1000, every=1000)
    for p in laptop_pings:                                           # same machine: sub-millisecond round trips
        p["t1"] = p["t0"] + 0.4
        p["server"] = laptop_to_server(p["t0"] + 0.2)
    true_samples = frame_at(server_to_phone(laptop_to_server(event_times))) - first_frame
    return rec, laptop_pings, event_times, true_samples


class ClockMapping(unittest.TestCase):
    def test_events_land_on_the_right_phone_samples(self):
        for seed in range(3):
            rec, laptop_pings, times, truth = simulated_devices(seed)
            audio, samples, details = phone.assemble(rec, laptop_pings, times)
            error_ms = np.abs(samples - truth) / 48
            self.assertLess(error_ms.max(), 15, details)        # within half the fastest round trip, plus jitter
            self.assertLess(np.median(error_ms), 8, details)
            self.assertAlmostEqual(details["phone_clock"]["rate_ppm"], -40, delta=25)

    def test_gaps_become_silence_without_shifting_audio(self):
        rec, laptop_pings, times, _ = simulated_devices()
        starts = sorted(rec.chunks)
        for start in starts:
            rec.chunks[start] = np.full(len(rec.chunks[start]), 7, np.int16)
        del rec.chunks[starts[3]]                                # a lost upload
        audio, _, details = phone.assemble(rec, laptop_pings, times)
        gap = slice(starts[3] - starts[0], starts[4] - starts[0])
        self.assertTrue((audio[gap] == 0).all() and audio[starts[4] - starts[0]] == 7)
        self.assertGreater(details["audio_gaps_ms"], 400)

    def test_resamples_other_rates_to_48k(self):
        rec, laptop_pings, times, truth = simulated_devices()
        rec.sync["sample_rate"] = 44100
        audio, samples, _ = phone.assemble(rec, laptop_pings, times)
        total = sum(len(c) for c in rec.chunks.values())
        self.assertAlmostEqual(len(audio), total * 48000 / 44100, delta=2)


@unittest.skipUnless(OLD_SESSION.exists(), "needs a real recording from the original project")
class PhoneLabelling(unittest.TestCase):
    """A real laptop recording turned into a simulated phone recording: extra microphone delay, an audio clock
    that drifts, a different level and added noise, and event positions with a sync error and no drift tracking."""

    @classmethod
    def setUpClass(cls):
        from sneakey.audio import read_wav
        cls.audio = read_wav((OLD_SESSION / "audio.wav").read_bytes())
        cls.events = [{"type": r["event_type"], "sample": int(float(r["sample_index_est"]))}
                      for r in csv.DictReader(open(OLD_SESSION / "keystrokes.csv"))]
        cls.truth = label_events(cls.audio, cls.events)

    def simulate(self, delay_ms, drift_ppm, sync_error_ms):
        rng = np.random.default_rng(1)
        r, d = 1 + drift_ppm * 1e-6, delay_ms * 48
        n = int((len(self.audio) + d) * r)
        audio = np.interp(np.arange(n) / r - d, np.arange(len(self.audio)), self.audio, left=0, right=0) * 0.4
        audio += rng.normal(0, 0.0004, n)
        events = [{**e, "sample": int(e["sample"] + sync_error_ms * 48)} for e in self.events]
        expected = [None if c is None else (c + d) * r for c in self.truth["centers"]]
        return audio, events, expected

    def test_wide_labelling_recovers_delay_and_drift(self):
        for delay, drift, sync_error in [(230, 300, 25), (120, -400, -60)]:
            audio, events, expected = self.simulate(delay, drift, sync_error)
            labels = label_events(audio, events, wide=True)
            errors = np.array([abs(c - x) / 48 for c, x, e in zip(labels["centers"], expected, events)
                               if e["type"] == "keydown" and c is not None and x is not None])
            self.assertGreater((errors <= 2).mean(), 0.8, (delay, drift))
            self.assertAlmostEqual(labels["drift_ppm"], drift, delta=60)
            self.assertEqual(label_events(audio, events)["press_lag_ms"] <= 100, True)   # laptop method can't reach it


if __name__ == "__main__":
    unittest.main()
