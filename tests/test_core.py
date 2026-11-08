"""
Unit tests for the pipeline, ground truth, evaluation and sentence generator.

    python -m unittest discover -s tests -v

The parity tests compare against the original project's code and one of its recordings in ../SneaKey, and are
skipped when that folder is absent. Everything else uses synthetic audio.
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SNEAKEY_DATA", tempfile.mkdtemp(prefix="sneakey-test-"))
sys.path.insert(0, str(ROOT))

from sneakey import detection, evaluation, groundtruth, quotes, sentences  # noqa: E402
from sneakey.audio import read_wav, wav_bytes  # noqa: E402
from sneakey.pipeline import Pipeline, render_text  # noqa: E402

OLD = Path(os.environ.get("SNEAKEY_ORIGINAL", ROOT.parent / "SneaKey"))
OLD_SESSION = OLD / "recordings" / "session_20261107_182727"
pipeline = Pipeline()


def synthetic_typing(times_ms, seconds=6.0, seed=0):
    """Background noise with a short decaying broadband click at each time."""
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 0.002, int(seconds * 48000))
    click = rng.normal(0, 1, 480) * np.exp(-np.arange(480) / 60) * 0.3
    for t in times_ms:
        start = int(t * 48)
        audio[start:start + 480] += click
    return audio


def _has_pandas():
    import importlib.util
    return importlib.util.find_spec("pandas") is not None   # the original project's code needs it; SneaKey2 doesn't


@unittest.skipUnless(OLD_SESSION.exists() and (OLD / "webapp").exists() and _has_pandas(),
                     "original project (and pandas, which it needs) not available")
class ParityWithOriginal(unittest.TestCase):
    """The restructured pipeline must give exactly the original's output on a real recording."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(OLD / "webapp"))
        from server.model import V6Model, _load_presses
        from server.words import WordDecoder, load_tuned_settings
        cls.wav = (OLD_SESSION / "audio.wav").read_bytes()
        old = V6Model()
        cls.old = old.predict_wav(cls.wav)
        cls.old_words = WordDecoder(old.key_names, load_tuned_settings()).decode(cls.old["key_scores"])
        cls.old_presses = _load_presses(OLD_SESSION)
        cls.samples = read_wav(cls.wav)
        cls.new = pipeline.predict(cls.samples)

    def test_detection_and_keys(self):
        self.assertEqual([e["time_ms"] for e in self.old["events"]], [p["time_ms"] for p in self.new["presses"]])
        self.assertEqual([e["key"] for e in self.old["events"]], [p["key"] for p in self.new["presses"]])
        np.testing.assert_allclose(np.exp(self.old["key_scores"]), self.new["probabilities"] + 1e-9, atol=1e-6)

    def test_text_and_words(self):
        self.assertEqual(self.old["text"], self.new["text"])
        self.assertEqual(self.old_words["text"], self.new["words"]["text"])

    def test_ground_truth(self):
        events = [{"type": r["event_type"], "sample": int(float(r["sample_index_est"]))}
                  for r in __import__("csv").DictReader(open(OLD_SESSION / "keystrokes.csv"))]
        labels = groundtruth.label_events(self.samples, events)
        self.assertEqual(labels["press_lag_ms"], int(self.old_presses.offset.iloc[0]))
        centres = sorted(c for c, e in zip(labels["centers"], events) if e["type"] == "keydown")
        self.assertEqual(centres, sorted(self.old_presses.center.tolist()))

    def test_published_metrics(self):
        """v6's audio-only numbers and the CNN's published logged-timing result (cnn_final.json)."""
        session = {"events": [], "typed_text": json.loads((OLD_SESSION / "metadata.json").read_text())["typed_text"]}
        import csv
        for r in csv.DictReader(open(OLD_SESSION / "keystrokes.csv")):
            session["events"].append({"type": r["event_type"], "code": r["code"], "key": r["key"],
                                      "sample": int(float(r["sample_index_est"])), "repeat": r["repeat"] == "1"})
        presses, _ = groundtruth.true_presses(session, self.samples)
        logged = pipeline.classify(self.samples, [p["sample"] for p in presses])
        e = evaluation.evaluate(self.new, presses, session["typed_text"], logged, pipeline.key_names)
        self.assertEqual((e["detection"]["matched"], e["detection"]["missed"], e["detection"]["extra"]), (805, 240, 142))
        self.assertAlmostEqual(e["audio_only"]["key_accuracy_matched"], 0.8385, places=4)
        self.assertAlmostEqual(e["text"]["raw_rates"]["cer"], 0.3870, places=4)
        self.assertAlmostEqual(e["text"]["dictionary_rates"]["cer"], 0.2948, places=4)
        self.assertAlmostEqual(e["logged_timing"]["accuracy"], 0.7742, places=4)
        self.assertAlmostEqual(e["logged_timing"]["top3"], 0.8995, places=4)


class Pipeline_(unittest.TestCase):
    def test_silence_gives_no_presses(self):
        result = pipeline.predict(np.zeros(48000))
        self.assertEqual(result["presses"], [])
        self.assertEqual(result["text"], "")
        e = evaluation.evaluate(result, [], "", np.zeros((0, len(pipeline.key_names))), pipeline.key_names)
        self.assertEqual(e["detection"]["detected_presses"], 0)

    def test_candidates_found_at_clicks(self):
        times = np.arange(500, 5500, 250)
        audio = synthetic_typing(times)
        found = detection.find_candidates(audio, pipeline.detector.flux_threshold) / 48
        nearest = np.array([np.min(np.abs(found - t)) for t in times])
        self.assertTrue((nearest < 5).all(), nearest)

    def test_probabilities_are_distributions(self):
        audio = synthetic_typing([1000, 2000, 3000])
        p = pipeline.classify(audio, [48000, 96000, 144000])
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)

    def test_decoder_reports_press_spans(self):
        rng = np.random.default_rng(1)
        probabilities = rng.dirichlet(np.ones(len(pipeline.key_names)) * 0.1, size=40)
        decoded = pipeline.decoder.decode(np.log(probabilities + 1e-9))
        used = [i for piece in decoded["words"] for i in piece["presses"]]
        self.assertEqual(used, sorted(used))                     # in time order, never repeated
        self.assertTrue(all(0 <= i < 40 for i in used))
        for piece in decoded["words"]:
            if not piece["separator"]:
                self.assertEqual(len(piece["raw"]), len(piece["presses"]))

    def test_render_text(self):
        self.assertEqual(render_text(["KeyH", "KeyI", "KeyX", "Backspace", "Space", "Comma", "Digit1"]), "hi ,1")

    def test_wav_round_trip(self):
        audio = synthetic_typing([100], seconds=0.5)
        np.testing.assert_allclose(read_wav(wav_bytes(audio)), np.clip(audio, -1, 1), atol=1 / 32768)


class GroundTruth(unittest.TestCase):
    def test_recovers_input_delay(self):
        logged = np.arange(600, 5400, 180).astype(float)
        delay = 43
        audio = synthetic_typing(logged + delay)
        events = [{"type": "keydown", "sample": int(t * 48)} for t in logged]
        labels = groundtruth.label_events(audio, events)
        self.assertAlmostEqual(labels["press_lag_ms"], delay, delta=2)
        heard = np.array(labels["centers"]) / 48
        self.assertTrue((np.abs(heard - (logged + delay)) <= 3).all())


class Evaluation(unittest.TestCase):
    def test_edit_distance_and_alignment(self):
        self.assertEqual(evaluation.edit_distance("kitten", "sitting"), 3)
        ops = evaluation.align("the cat sat".split(), "the bat sat on".split())
        self.assertEqual([op for op, _, _ in ops], ["equal", "substitute", "equal", "insert"])

    def test_match_is_one_to_one_within_tolerance(self):
        m = evaluation.match([100, 200, 300], [101, 105, 330], 20)
        self.assertEqual(m.tolist(), [0, -1, -1])

    def test_word_alignment_ignores_punctuation(self):
        decoded = {"words": [{"text": "hello", "separator": False}, {"text": ",", "separator": True},
                             {"text": " ", "separator": True}, {"text": "word", "separator": False}]}
        rows = evaluation.word_alignment("Hello, world.", decoded)
        self.assertEqual([(r["op"], r["typed"], r["predicted"]) for r in rows],
                         [("equal", "hello", "hello"), ("substitute", "world", "word")])


class Sentences(unittest.TestCase):
    def test_deterministic_by_seed(self):
        self.assertEqual(sentences.generate_test(300, seed=5)["text"], sentences.generate_test(300, seed=5)["text"])
        self.assertNotEqual(sentences.generate_test(300, seed=5)["text"], sentences.generate_test(300, seed=6)["text"])

    def test_typeable_and_well_formed(self):
        for seed in range(20):
            result = sentences.generate_test(400, seed=seed)
            self.assertRegex(result["text"], r"^[A-Za-z ,.?]+$")
            self.assertGreaterEqual(len(result["text"]), 400)
            for s in result["sentences"]:
                self.assertTrue(s["text"][0].isupper() and s["text"][-1] in ".?", s["text"])
                self.assertNotIn("  ", s["text"])

    def test_varied_frames_and_low_repetition(self):
        history, frames, repeats = [], set(), []
        for seed in range(25):
            result = sentences.generate_test(400, seed=seed, previous_texts=history)
            repeats.append(result["repeated_trigrams"])
            frames |= {s["frame"] for s in result["sentences"]}
            consecutive = [a["frame"] == b["frame"] for a, b in zip(result["sentences"], result["sentences"][1:])]
            self.assertFalse(any(consecutive), "a frame repeated back to back")
            history.append(result["text"])
        self.assertEqual(frames, set(sentences.FRAMES))
        self.assertLessEqual(sum(repeats), 5)

    def test_coverage_mode_favours_missing_letters(self):
        counts = {f"Key{c}": 1000 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        counts["KeyZ"] = 0
        def z_rate(mode):
            text = " ".join(sentences.generate_test(3000, seed=s, mode=mode, key_counts=counts)["text"] for s in range(5)).lower()
            return text.count("z") / len(re.sub(r"[^a-z]", "", text))
        self.assertGreater(z_rate("coverage"), 1.3 * z_rate("natural"))


class Quotes(unittest.TestCase):
    QUOTES = [{"id": i, "text": f"Quote number {i} is about the {word} and nothing else.", "author": f"Author {i}"}
              for i, word in enumerate(["sea", "sky", "fox", "jazz", "quiz", "moon", "tree", "road", "wind", "rain",
                                        "lamp", "door", "fire", "snow", "hill", "lake", "bird", "song", "book", "star"])]

    def test_normalises_and_filters(self):
        self.assertEqual(quotes.normalise("\u201cIt\u2019s  fine\u201d \u2014 ok\u2026"), '"It\'s fine" - ok...')
        self.assertTrue(quotes.usable("Be like a flower that gives its fragrance even to the hand that crushed it."))
        self.assertFalse(quotes.usable("I Have Held Many Things In My Hands, And I Have Lost Them All."))
        self.assertFalse(quotes.usable("Success is 99% perspiration, and that is all there is to it."))

    def test_deterministic_and_avoids_previous(self):
        a = quotes.generate_test(150, seed=4, quotes=self.QUOTES)
        self.assertEqual(a["text"], quotes.generate_test(150, seed=4, quotes=self.QUOTES)["text"])
        self.assertGreaterEqual(len(a["text"]), 150)
        self.assertEqual(a["source"], "quotes")
        self.assertTrue(all(s["author"] and s["frame"] == "quote" for s in a["sentences"]))
        b = quotes.generate_test(150, seed=4, quotes=self.QUOTES, previous_texts=[a["text"]])
        self.assertFalse({s["id"] for s in a["sentences"]} & {s["id"] for s in b["sentences"]})


if __name__ == "__main__":
    unittest.main()
