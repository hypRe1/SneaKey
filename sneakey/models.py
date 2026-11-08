"""Key classifier registry: the shipped CNN plus models trained on the Train page.

Every classifier has the same interface:
    key_names            keys it can output (column order of its probabilities)
    classify(audio, c)   (presses, keys) probabilities for presses at sample indices c
    input_view(audio, c) what the classifier looks at for one press, for the keystroke inspector
    id, name, kind, version, trained_on, validation_sessions, manifest

Trained models live in data/models/<id>/ (manifest.json + model.pt or model.joblib), next to the recordings
they were trained on. data/active_model.json names the model the pipeline uses; the default is the shipped CNN.
"""
import hashlib
import json
import shutil

import joblib
import numpy as np

from . import config
from .classifier import CROP_MS, MEL_BANK, N_FFT, KeyCNN, slices_around
from .features import band_mel_features, mel_patch

ORIGINAL_ID = "original-cnn"
MODELS_DIR = config.DATA_DIR / "models"
ACTIVE_FILE = config.DATA_DIR / "active_model.json"


class FeatureModel:
    """An SVM or LDA on the band64+mel features. `estimator` is a fitted scikit-learn pipeline with predict_proba."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.key_names = [str(k) for k in estimator.classes_]

    def classify(self, audio, centers):
        if len(centers) == 0:
            return np.zeros((0, len(self.key_names)))
        return self.estimator.predict_proba(band_mel_features(slices_around(audio, centers)))

    def input_view(self, audio, center):
        patch = mel_patch(slices_around(audio, [center]))[0]               # (10 time steps, 40 mels)
        patch = patch - patch.mean()
        return {"matrix": np.round(patch, 3).tolist(), "time_ms": [-10, 30], "bands_hz": [300, 20000],
                "description": "Part of the SVM/LDA input: log-mel energies of the 700 Hz high-passed audio, 40 bands "
                               "from 10 ms before to 30 ms after the press in 4 ms steps (plus 64 band energies, not shown)."}


def _cnn_input_view(self, audio, center):
    freqs = np.fft.rfftfreq(N_FFT, 1 / config.SAMPLE_RATE)
    centres = (MEL_BANK * freqs[:, None]).sum(0) / np.maximum(MEL_BANK.sum(0), 1e-12)
    return {"matrix": np.round(self.network_input(audio, center), 3).tolist(), "time_ms": list(CROP_MS),
            "bands_hz": [round(float(centres[0])), round(float(centres[-1]))],
            "description": "What the CNN sees: log-mel spectrogram of the 700 Hz high-passed audio, 64 bands, "
                           f"{CROP_MS[0]} to +{CROP_MS[1]} ms around the press in 1 ms steps, normalised per press."}


KeyCNN.input_view = _cnn_input_view


def _attach(model, manifest, version):
    model.manifest = manifest
    model.id = manifest["id"]
    model.name = manifest.get("name", manifest["id"])
    model.kind = manifest["kind"]
    model.version = version
    model.trained_on = set(manifest.get("trained_on", []))
    model.validation_sessions = set(manifest.get("validation_sessions", []))
    return model


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def original_manifest():
    manifest = json.loads((config.MODELS_DIR / "key_cnn.json").read_text())
    return {"id": ORIGINAL_ID, "name": "Original CNN (shipped)", "kind": "cnn", "created": None, **manifest}


def list_models():
    """Manifests of every available model, the shipped one first."""
    out = [original_manifest()]
    if MODELS_DIR.exists():
        for folder in sorted(MODELS_DIR.iterdir(), reverse=True):
            if (folder / "manifest.json").exists():
                out.append(json.loads((folder / "manifest.json").read_text()))
    return out


def load(model_id):
    if model_id == ORIGINAL_ID:
        model = KeyCNN()
        return _attach(model, original_manifest(), f"cnn-{model.version}")
    folder = _folder(model_id)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["kind"] == "cnn":
        model = KeyCNN(folder / "model.pt")
        return _attach(model, manifest, f"cnn-{model.version}")
    path = folder / "model.joblib"
    model = FeatureModel(joblib.load(path))      # our own file, written by save() below
    return _attach(model, manifest, f"{manifest['kind']}-{_file_hash(path)}")


def save(manifest, kind, network=None, key_names=None, estimator=None):
    """Store a newly trained model; returns its folder."""
    import torch
    folder = _folder(manifest["id"], must_exist=False)
    folder.mkdir(parents=True)
    if kind == "cnn":
        torch.save({"keys": [str(k) for k in key_names], "state": network.state_dict()}, folder / "model.pt")
    else:
        joblib.dump(estimator, folder / "model.joblib", compress=3)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return folder


def delete(model_id):
    if model_id == ORIGINAL_ID:
        raise ValueError("The shipped model can't be deleted.")
    if model_id == active_id():
        raise ValueError("Activate another model before deleting this one.")
    shutil.rmtree(_folder(model_id))


def active_id():
    if ACTIVE_FILE.exists():
        model_id = json.loads(ACTIVE_FILE.read_text()).get("id")
        if model_id == ORIGINAL_ID or (model_id and (MODELS_DIR / model_id / "manifest.json").exists()):
            return model_id
    return ORIGINAL_ID


def set_active(model_id):
    if model_id != ORIGINAL_ID:
        _folder(model_id)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(json.dumps({"id": model_id}))


def _folder(model_id, must_exist=True):
    if not model_id.replace("-", "").replace("_", "").isalnum():
        raise KeyError(model_id)
    folder = MODELS_DIR / model_id
    if must_exist and not (folder / "manifest.json").exists():
        raise KeyError(model_id)
    return folder
