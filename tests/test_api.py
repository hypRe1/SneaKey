"""
API tests against a real server process and a temporary data folder holding one synthetic session.

    python -m unittest tests.test_api -v
"""
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import synthetic_typing  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(tempfile.mkdtemp(prefix="sneakey-api-"))
        port = free_port()
        cls.base = f"http://127.0.0.1:{port}/api/"
        cls.server = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--port", str(port)], cwd=ROOT,
                                      env=dict(os.environ, SNEAKEY_DATA=str(cls.data)),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                urllib.request.urlopen(cls.base + "health", timeout=2)
                break
            except OSError:
                time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(10)
        shutil.rmtree(cls.data, ignore_errors=True)

    def call(self, path, method="GET", body=None, headers=None):
        request = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
            return json.loads(data) if response.headers.get_content_type() == "application/json" else data

    def status(self, *args, **kwargs):
        try:
            self.call(*args, **kwargs)
            return 200
        except urllib.error.HTTPError as error:
            return error.code

    def make_session(self, delay_ms=40):
        logged = np.arange(800, 5200, 200).astype(float)
        audio = synthetic_typing(logged + delay_ms, seconds=6)
        codes = ["KeyH", "KeyE", "KeyL", "KeyL", "KeyO", "Space"] * 4
        events = []
        for t, code in zip(logged, codes):
            for kind, offset in (("keydown", 0), ("keyup", 80)):
                events.append({"event_type": kind, "key": code[-1].lower() if code.startswith("Key") else " ", "code": code,
                               "sample_index_est": int((t + offset) * 48), "repeat": 0, "action": "char" if kind == "keydown" else ""})
        metadata = {"name": "P-api", "participant": {"id": "P-api"}, "target_text": "hello " * 4, "typed_text": "hello " * 4}
        session_id = self.call("sessions", "POST", json.dumps({"metadata": metadata, "events": events}).encode(),
                               {"Content-Type": "application/json"})["id"]
        pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype("<i2").tobytes()
        self.call(f"sessions/{session_id}/audio", "PUT", pcm)
        return session_id

    def test_session_lifecycle(self):
        health = self.call("health")
        self.assertEqual(health["sample_rate"], 48000)
        self.assertIn("KeyA", health["keys"])

        session_id = self.make_session()
        listed = {s["id"]: s for s in self.call("sessions")}
        self.assertEqual(listed[session_id]["role"], "unseen")
        self.assertEqual(listed[session_id]["participant"], "P-api")

        signals = self.call(f"sessions/{session_id}/signals")
        self.assertAlmostEqual(signals["press_lag_ms"], 40, delta=2)
        self.assertEqual(len(signals["novelty"]), 6001)

        rate, filtered = wavfile.read(io.BytesIO(self.call(f"sessions/{session_id}/audio.wav?filter=model")))
        self.assertEqual((rate, len(filtered)), (48000, 6 * 48000))

        detail = self.call(f"sessions/{session_id}/keystroke?time_ms=840")
        self.assertEqual(len(detail["input_view"]["matrix"]), 60)       # the shipped CNN: 60 ms x 64 mel bands
        self.assertEqual(len(detail["input_view"]["matrix"][0]), 64)
        self.assertAlmostEqual(sum(t["probability"] for t in detail["top"]), 1, delta=0.5)

        result = self.call(f"sessions/{session_id}/predict", "POST")
        self.assertEqual(result["evaluation"]["detection"]["typed_presses"], len(np.arange(800, 5200, 200)))
        self.assertEqual(len(result["probabilities"]), len(result["presses"]))
        self.assertIn("unseen", self.call("summary"))
        self.assertTrue(any(s["has_prediction"] for s in self.call("sessions")))

        # Locked sessions can be analysed but not predicted.
        (self.data / "splits.json").write_text(json.dumps({"locked": [session_id]}))
        self.assertEqual(self.status(f"sessions/{session_id}/predict", "POST"), 403)
        self.assertEqual(self.status(f"sessions/{session_id}/signals"), 200)
        (self.data / "splits.json").unlink()

        self.call(f"sessions/{session_id}", "DELETE")
        self.assertEqual(self.status(f"sessions/{session_id}"), 404)

    def test_training_lifecycle(self):
        train_ids = [self.make_session(), self.make_session_after()]
        validation_id = self.make_session_after()
        self.assertEqual(self.status("training", "POST", json.dumps({"kind": "bogus", "train": train_ids}).encode(),
                                     {"Content-Type": "application/json"}), 400)
        self.assertEqual(self.status("training", "POST", json.dumps({"kind": "cnn", "train": train_ids}).encode(),
                                     {"Content-Type": "application/json"}), 400)    # CNN training disabled for now
        self.assertEqual(self.status("training", "POST", json.dumps({"kind": "lda", "train": [validation_id], "validation": [validation_id]}).encode(),
                                     {"Content-Type": "application/json"}), 400)    # same session in both sets
        body = {"kind": "lda", "train": train_ids, "validation": [validation_id], "name": "api test", "end_to_end": True,
                "min_presses": 1, "retrain_detector": True}
        self.call("training", "POST", json.dumps(body).encode(), {"Content-Type": "application/json"})
        for _ in range(300):
            status = self.call("training")
            if status["state"] != "running":
                break
            time.sleep(0.5)
        self.assertEqual(status["state"], "done", status.get("log"))
        evaluation = status["result"]["evaluation"]
        self.assertEqual(evaluation["presses"], 22)
        self.assertIn("end_to_end", evaluation["candidate"])
        model_id = status["result"]["model"]["id"]
        self.assertEqual(status["result"]["model"]["microphone"], "laptop")
        detector = status["result"]["model"]["detector"]
        self.assertIn(detector["max_leaf_nodes"], (7, 15, 31))
        self.assertIsNotNone(evaluation["detection"])                        # new vs active detector compared

        listed = {m["id"]: m for m in self.call("models")}
        self.assertTrue(listed["original-cnn"]["active"])
        self.assertEqual(listed[model_id]["trained_on"], train_ids)

        self.call(f"models/{model_id}/activate", "POST")
        roles = {s["id"]: s["role"] for s in self.call("sessions")}
        self.assertEqual([roles[i] for i in train_ids], ["trained", "trained"])
        self.assertEqual(roles[validation_id], "validation")
        result = self.call(f"sessions/{validation_id}/predict", "POST")
        self.assertEqual(result["model"]["id"], model_id)
        self.assertEqual(self.status(f"models/{model_id}", "DELETE"), 400)      # can't delete the active model

        self.call("models/original-cnn/activate", "POST")
        self.call(f"models/{model_id}", "DELETE")
        self.assertNotIn(model_id, {m["id"] for m in self.call("models")})
        self.assertEqual(self.status("models/original-cnn", "DELETE"), 400)
        self.assertEqual(self.status("models/nope/activate", "POST"), 404)

    def make_session_after(self):
        time.sleep(1.1)      # session ids are per second
        return self.make_session()

    @unittest.skipIf(os.environ.get("DEEPSEEK_API_KEY"), "a DeepSeek key is set: this test must not send data")
    def test_llm_decoder_needs_a_key(self):
        self.assertFalse(self.call("llm")["configured"])
        session_id = self.make_session()
        self.assertEqual(self.status(f"sessions/{session_id}/llm-decode", "POST"), 400)   # no key: nothing is sent
        self.assertEqual(self.status(f"sessions/{session_id}/llm-decode?style=candidates"), 404)   # nothing stored
        self.assertEqual(self.status(f"sessions/{session_id}/llm-decode?style=bogus", "POST"), 400)

    def test_rejects_bad_input(self):
        self.assertEqual(self.status("sessions/..%2F..%2Fetc"), 404)
        self.assertEqual(self.status("sessions/session_20990101_000000/predict", "POST"), 404)
        self.assertEqual(self.status("sessions", "POST", b"{}", {"Content-Type": "application/json"}), 400)
        session_id = self.call("sessions", "POST", json.dumps({"metadata": {}, "events": []}).encode(), {"Content-Type": "application/json"})["id"]
        self.assertEqual(self.status(f"sessions/{session_id}/audio", "PUT", b"\0" * 1000), 400)   # under half a second
        self.assertEqual(self.status("sentences?mode=bogus"), 400)
        test = self.call("sentences?chars=200&seed=3")
        self.assertEqual(test["seed"], 3)
        self.assertGreaterEqual(len(test["text"]), 200)


if __name__ == "__main__":
    unittest.main()
