"""Session splits for training, shared by the training scripts.

Roles come from data/splits.json ({"validation": [...], "test": [...], "locked": [...]}); every other recording
with audio is training data. Locked sessions are never loaded by any training script.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sneakey import config, sessions  # noqa: E402

OUT = config.DATA_DIR / "training"


def split_sessions():
    splits = config.load_splits()
    available = [s["id"] for s in sessions.list_sessions() if s["has_audio"]]
    locked = set(splits["locked"])
    test = [s for s in available if s in splits["test"]]
    validation = [s for s in available if s in splits["validation"]]
    train = [s for s in available if s not in locked and s not in test and s not in validation]
    return {"train": sorted(train), "validation": sorted(validation), "test": sorted(test), "locked": sorted(locked)}
