"""
Tune the dictionary word decoder (sneakey/decoder.py) on the validation sessions.

Unchanged method from the original analysis/tune_word_decoder.py: the audio-only prediction for each validation
session is computed once; settings are then tuned one at a time (two rounds) by lowercase character error rate,
then word error rate, against the typed text. Test and locked sessions are never used.

    python training/tune_decoder.py            # report the best settings
    python training/tune_decoder.py --save     # ... and write them to models/decoder_settings.json
    python training/tune_decoder.py --test     # score the saved settings once on the test sessions
"""
import argparse
import json
from dataclasses import asdict

import numpy as np

from common import OUT, split_sessions
from sneakey import sessions
from sneakey.audio import read_wav
from sneakey.decoder import TUNED_SETTINGS, DecoderSettings, WordDecoder, load_tuned_settings
from sneakey.evaluation import text_error_rates
from sneakey.pipeline import Pipeline

CHOICES = {
    "temperature": [1, 2, 3, 4, 6, 8],
    "space_threshold": [0.2, 0.35, 0.5, 0.7],
    "frequency_weight": [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
    "missed_key_cost": [2, 3, 4, 6, 8],
    "extra_press_cost": [1, 2, 3, 4, 6],
    "split_cost": [0, 2, 4, 8, 16],
    "raw_word_cost": [4, 8, 12, 20, 40],
    "min_zipf": [2.5, 3.0, 3.5],
}


def session_scores(pipeline, session_ids):
    """Per-session (log key probabilities, raw text, typed text), cached per model version."""
    out = {}
    for session_id in session_ids:
        path = OUT / f"decoder_{session_id}.{pipeline.version}.npz"
        if not path.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            prediction = pipeline.predict(read_wav(sessions.audio_path(session_id)))
            np.savez(path, scores=np.log(prediction["probabilities"] + 1e-9), raw=prediction["text"])
        data = np.load(path)
        out[session_id] = (data["scores"], str(data["raw"]), sessions.load_session(session_id)["typed_text"])
    return out


def evaluate(decoder, scored):
    rates = [text_error_rates(typed, decoder.decode(scores)["text"]) for scores, _, typed in scored.values()]
    return {"cer": float(np.mean([r["cer"] for r in rates])), "wer": float(np.mean([r["wer"] for r in rates]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    splits = split_sessions()
    pipeline = Pipeline()

    if args.test:
        decoder = WordDecoder(pipeline.key_names, load_tuned_settings())
        for session_id, (scores, raw, typed) in session_scores(pipeline, splits["test"]).items():
            print(session_id, "keys only", text_error_rates(typed, raw), "dictionary", text_error_rates(typed, decoder.decode(scores)["text"]))
        return

    if not splits["validation"]:
        raise SystemExit("No validation sessions: list some under \"validation\" in data/splits.json.")
    scored = session_scores(pipeline, splits["validation"])
    print("keys only:", {s: text_error_rates(typed, raw) for s, (_, raw, typed) in scored.items()})
    best = asdict(DecoderSettings())
    best_result = evaluate(WordDecoder(pipeline.key_names, DecoderSettings(**best)), scored)
    rank = lambda r: (r["cer"], r["wer"])
    for round_number in range(2):
        for name, values in CHOICES.items():
            for value in values:
                if value == best[name]:
                    continue
                trial = dict(best, **{name: value})
                result = evaluate(WordDecoder(pipeline.key_names, DecoderSettings(**trial)), scored)
                if rank(result) < rank(best_result):
                    best, best_result = trial, result
            print(f"round {round_number + 1}, {name} = {best[name]}: CER {best_result['cer']:.3f} WER {best_result['wer']:.3f}", flush=True)
    print("BEST", json.dumps({"settings": best, **best_result}))
    if args.save:
        TUNED_SETTINGS.write_text(json.dumps(best, indent=1))
        print(f"saved -> {TUNED_SETTINGS}")


if __name__ == "__main__":
    main()
