"""
The CNN key classifier: names the key for each press found by v6's detector.

It looks at a log-mel spectrogram from 15 ms before to 45 ms after the press (64 mel bands, one frame per
millisecond) of the 700 Hz high-passed audio, and outputs a probability for each key. Trained by
training/train_cnn.py (originally analysis/key_model_cnn.py, unchanged); in the original project it named 60% of
validation presses and 77% of test-session presses correctly with logged timing, against 45% / 56% for v6's SVM.

The audio around each press is filtered locally (a quarter of a second either side), so whole recordings and the
live stream give exactly the same answer for the same press.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch import nn

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "key_cnn.pt"
MANIFEST_FILE = MODEL_FILE.with_suffix(".json")     # which sessions the network was trained on
SAMPLE_RATE = 48000
SLICE = 2400                         # samples either side of the press kept for the spectrogram (50 ms)
FILTER_CONTEXT = 12000               # samples either side filtered together (250 ms)
HIGHPASS = signal.butter(3, 700, "highpass", fs=SAMPLE_RATE, output="sos")

START_MS, END_MS = -18, 48           # spectrogram range computed around each press
CROP_MS = (-15, 45)                  # range the network sees
N_FFT, HOP, N_MELS = 256, 48, 64     # 5.3 ms frames every 1 ms
MAX_SHIFT_MS = 3                     # training cuts each example up to this many ms off centre
TEST_SHIFTS_MS = (-2, -1, 0, 1, 2)   # the prediction is averaged over these small shifts


def mel_filterbank(n_fft, n_mels, low_hz=300, high_hz=20000):
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    edges = to_hz(np.linspace(to_mel(low_hz), to_mel(high_hz), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    bank = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        left, center, right = edges[i:i + 3]
        bank[i] = np.clip(np.minimum((freqs - left) / (center - left), (right - freqs) / (right - center)), 0, None)
    return bank


MEL_BANK = mel_filterbank(N_FFT, N_MELS).T


def spectrograms(slices):
    """Log-mel spectrogram for each 100 ms slice centred on a press: (presses, frames, mels)."""
    first = SLICE + START_MS * 48 - N_FFT // 2
    frames = END_MS - START_MS
    index = first + np.arange(frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
    window = np.hanning(N_FFT)
    out = np.empty((len(slices), frames, N_MELS), np.float32)
    for start in range(0, len(slices), 1000):
        power = np.abs(np.fft.rfft(slices[start:start + 1000][:, index] * window, axis=-1)) ** 2
        out[start:start + 1000] = np.log(power @ MEL_BANK + 1e-10)
    return out


def crop(batch, shift_ms):
    """The network's input window, moved by shift_ms."""
    begin = CROP_MS[0] - START_MS + shift_ms
    return batch[:, begin:begin + CROP_MS[1] - CROP_MS[0]]


def normalise(x):
    """Remove each press's overall level and scale, so loudness differences between sessions matter less."""
    return (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)


class KeyNet(nn.Module):
    def __init__(self, n_keys):
        super().__init__()

        def block(inputs, outputs):
            return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(),
                                 nn.Conv2d(outputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_keys))

    def forward(self, x):                      # x: (batch, time, mels)
        x = self.features(x.unsqueeze(1))
        return self.classifier(x.mean(dim=(2, 3)))


def predict_probabilities(network, spectra, shifts=TEST_SHIFTS_MS):
    """Mean softmax over the given time shifts."""
    network.eval()
    total = 0
    with torch.no_grad():
        for shift in shifts:
            parts = [torch.softmax(network(torch.from_numpy(normalise(crop(spectra[i:i + 512], shift)))), 1).numpy()
                     for i in range(0, len(spectra), 512)]
            total = total + np.concatenate(parts)
    return total / len(shifts)


def slices_around(audio, centers):
    """700 Hz high-passed 100 ms slices centred on each press. audio: float samples; centers: sample indices.

    Each press is filtered within its own 250 ms-either-side context, so the result does not depend on how much
    audio surrounds it (a whole recording and a live 2 s window give the same slice for the same press). All
    presses are filtered together in one zero-phase pass over the stacked windows."""
    centers = np.asarray(centers, dtype=int)
    if len(centers) == 0:
        return np.zeros((0, 2 * SLICE), np.float32)
    padded = np.pad(np.asarray(audio, dtype=float), FILTER_CONTEXT)
    windows = padded[centers[:, None] + np.arange(2 * FILTER_CONTEXT)[None, :]]
    filtered = signal.sosfiltfilt(HIGHPASS, windows, axis=1)
    return filtered[:, FILTER_CONTEXT - SLICE:FILTER_CONTEXT + SLICE].astype(np.float32)


class KeyCNN:
    def __init__(self, model_file=MODEL_FILE):
        saved = torch.load(model_file, map_location="cpu", weights_only=True)
        self.key_names = list(saved["keys"])
        self.network = KeyNet(len(self.key_names))
        self.network.load_state_dict(saved["state"])
        self.network.eval()
        self.version = hashlib.sha256(Path(model_file).read_bytes()).hexdigest()[:12]
        manifest = Path(model_file).with_suffix(".json")
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        self.trained_on = set(self.manifest.get("trained_on", []))

    def network_input(self, audio, center):
        """The normalised log-mel patch the network sees for one press (no shift): (60 frames of 1 ms, 64 mels)."""
        spectra = spectrograms(slices_around(audio, [center]))
        return normalise(crop(spectra, 0))[0]

    def classify(self, audio, centers):
        """Probability of each key (columns in self.key_names order) for each press."""
        if len(centers) == 0:
            return np.zeros((0, len(self.key_names)))
        return predict_probabilities(self.network, spectrograms(slices_around(audio, centers)))