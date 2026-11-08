"""
Build the labelled key-press dataset from data/recordings (locked sessions excluded).

Each logged keydown (auto-repeats excluded) is placed on its sound by the ground-truth labelling
(sneakey.groundtruth, v6's method) and a 100 ms slice of 700 Hz high-passed audio centred on it is stored. As in
the original analysis/key_model_data.py, the high-pass is applied to the whole recording first.

    python training/build_dataset.py        # -> data/training/presses.npz
"""
import json

import numpy as np

from common import OUT, split_sessions
from sneakey import sessions
from sneakey.audio import highpass, read_wav
from sneakey.groundtruth import true_presses

HALF = 2400   # samples either side of the press: 50 ms


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = split_sessions()
    slices, codes, names, centres = [], [], [], []
    for session_id in splits["train"] + splits["validation"] + splits["test"]:
        session = sessions.load_session(session_id)
        audio = read_wav(sessions.audio_path(session_id))
        presses = [p for p in true_presses(session, audio)[0] if not p["repeat"]]
        padded = np.pad(highpass(audio, 700), HALF)
        for p in presses:
            slices.append(padded[p["sample"]:p["sample"] + 2 * HALF].astype(np.float32))
            codes.append(p["code"])
            names.append(session_id)
            centres.append(p["sample"])
        print(f"{session_id}: {len(presses)} presses", flush=True)
    if not slices:
        raise SystemExit("No recordings with audio to build a dataset from.")
    np.savez(OUT / "presses.npz", slices=np.stack(slices), codes=np.array(codes), sessions=np.array(names), centers=np.array(centres))
    counts = {code: codes.count(code) for code in sorted(set(codes))}
    (OUT / "dataset.json").write_text(json.dumps({"presses": len(codes), "splits": splits, "keys": counts}, indent=1))
    print(f"{len(codes)} presses, {len(counts)} keys -> {OUT / 'presses.npz'}")


if __name__ == "__main__":
    main()
