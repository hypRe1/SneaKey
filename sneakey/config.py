"""Paths and constants shared by the whole package.

Everything is local: recordings live under data/ (override with SNEAKEY_DATA), trained models under models/.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = Path(os.environ.get("SNEAKEY_DATA", ROOT / "data"))
RECORDINGS_DIR = DATA_DIR / "recordings"
CACHE_DIR = DATA_DIR / "cache"            # prediction results, keyed by model version
SPLITS_FILE = DATA_DIR / "splits.json"   # optional: {"validation": [...], "test": [...], "locked": [...]}

SAMPLE_RATE = 48000                      # every model was trained on 48 kHz mono 16-bit audio
SAMPLES_PER_MS = SAMPLE_RATE // 1000
MATCH_TOLERANCE_MS = 20                  # a detected press matches a typed press within this distance (v6's rule)

# Phone recording (run.py --phone): the HTTPS port the phone page is served on, and this laptop's LAN addresses.
PHONE_PORT = int(os.environ.get("SNEAKEY_PHONE_PORT") or 0)
PHONE_ADDRESSES = [a for a in os.environ.get("SNEAKEY_PHONE_ADDRESSES", "").split(",") if a]


def load_splits():
    """User-assigned evaluation roles for sessions. Sessions not listed are ordinary data."""
    splits = {"validation": [], "test": [], "locked": []}
    if SPLITS_FILE.exists():
        splits.update(json.loads(SPLITS_FILE.read_text()))
    return splits
