"""
Reading and writing recorded typing sessions.

Each session is a folder data/recordings/session_YYYYMMDD_HHMMSS/ holding audio.wav (mono 16-bit), keystrokes.csv
(one row per key event) and metadata.json. The format is unchanged from the original recorder, which the
training scripts and the v6 labelling read, so the column names and meanings must stay as they are.
"""
import csv
import json
import re
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config

SESSION_NAME = re.compile(r"^session_\d{8}_\d{6}$")

KEYSTROKE_COLUMNS = [
    "event_index",      # position in the file
    "event_type",       # keydown or keyup
    "key",              # KeyboardEvent.key, e.g. "a", "A", "Shift"
    "code",             # KeyboardEvent.code, the physical key, e.g. "KeyA". Used as the label.
    "location",
    "time_ms",          # performance.now() when the handler ran
    "event_time_ms",    # the event's own timestamp
    "t_rel_ms",         # event time relative to the first recorded audio sample
    "sample_index_est", # estimated audio sample of the event (before the microphone's input delay)
    "repeat", "shift", "ctrl", "alt", "meta",
    "action",           # what the typing test did: char, backspace, ignored, ...
    "target_char", "typed_char", "correct", "cursor_before", "cursor_after",
]


class NotFound(Exception):
    pass


def list_sessions():
    sessions = []
    if not config.RECORDINGS_DIR.exists():
        return sessions
    for folder in sorted(config.RECORDINGS_DIR.iterdir(), reverse=True):
        if not (SESSION_NAME.match(folder.name) and (folder / "metadata.json").exists()):
            continue
        metadata = json.loads((folder / "metadata.json").read_text())
        audio = metadata.get("audio", {})
        sessions.append({
            "id": folder.name,
            "name": metadata.get("name", ""),
            "participant": metadata.get("participant", {}).get("id", metadata.get("name", "")),
            "started": metadata.get("session", {}).get("started_iso"),
            "duration_s": audio.get("duration_s"),
            "keydowns": metadata.get("events", {}).get("keydown"),
            "has_audio": (folder / "audio.wav").exists(),
            "wpm": metadata.get("stats", {}).get("wpm"),
        })
    return sessions


def session_folder(session_id):
    if not SESSION_NAME.match(session_id):
        raise NotFound(session_id)
    folder = config.RECORDINGS_DIR / session_id
    if not (folder / "metadata.json").exists():
        raise NotFound(session_id)
    return folder


def load_session(session_id):
    folder = session_folder(session_id)
    metadata = json.loads((folder / "metadata.json").read_text())
    events = []
    with open(folder / "keystrokes.csv", newline="") as f:
        for row in csv.DictReader(f):
            events.append({
                "type": row["event_type"],
                "key": row["key"],
                "code": row["code"],
                "sample": int(float(row["sample_index_est"])),
                "repeat": row.get("repeat") == "1",
                "action": row.get("action", ""),
                "correct": row.get("correct") == "1",
                "typed_char": row.get("typed_char", ""),
                "target_char": row.get("target_char", ""),
            })
    return {
        "id": session_id,
        "name": metadata.get("name", ""),
        "sample_rate": metadata.get("audio", {}).get("sample_rate_hz", 48000),
        "target_text": metadata.get("target_text", ""),
        "typed_text": metadata.get("typed_text", ""),
        "stats": metadata.get("stats", {}),
        "participant": metadata.get("participant", {}),
        "conditions": metadata.get("conditions", {}),
        "generator": metadata.get("generator", {}),
        "started": metadata.get("session", {}).get("started_iso"),
        "events": events,
    }


def create_session(metadata, events):
    """Write keystrokes.csv and metadata.json for a new session. The audio follows in save_audio."""
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    while True:
        session_id = started.strftime("session_%Y%m%d_%H%M%S")
        folder = config.RECORDINGS_DIR / session_id
        if not folder.exists():
            break
        started += timedelta(seconds=1)
    folder.mkdir()

    with open(folder / "keystrokes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEYSTROKE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for index, event in enumerate(events):
            writer.writerow({**event, "event_index": index})

    metadata = dict(metadata)
    metadata["folder"] = session_id
    metadata["files"] = {"audio": "audio.wav", "keystrokes": "keystrokes.csv", "metadata": "metadata.json"}
    metadata["events"] = {
        "total": len(events),
        "keydown": sum(e.get("event_type") == "keydown" for e in events),
        "keyup": sum(e.get("event_type") == "keyup" for e in events),
        "repeated_keydown": sum(e.get("event_type") == "keydown" and e.get("repeat") == 1 for e in events),
    }
    _write_metadata(folder, metadata)
    return session_id


def save_audio(session_id, pcm_bytes, sample_rate):
    """Store raw little-endian int16 mono samples as the session's audio.wav."""
    folder = session_folder(session_id)
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    with wave.open(str(folder / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())

    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["audio"] = {
        **metadata.get("audio", {}),
        "file": "audio.wav", "format": "PCM", "bit_depth": 16, "channels": 1,
        "sample_rate_hz": sample_rate,
        "frames": len(samples),
        "duration_s": round(len(samples) / sample_rate, 3),
        "peak_amplitude_float": round(float(np.abs(samples.astype(np.int32)).max(initial=0)) / 32768, 6),
    }
    _write_metadata(folder, metadata)


def delete_session(session_id):
    """Remove a session folder (used when a recording is discarded after its metadata was written)."""
    import shutil
    shutil.rmtree(session_folder(session_id))


def audio_path(session_id):
    path = session_folder(session_id) / "audio.wav"
    if not path.exists():
        raise NotFound(session_id)
    return path


def _write_metadata(folder, metadata):
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
