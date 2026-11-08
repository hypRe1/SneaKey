"""Hand-made features for the SVM and LDA key classifiers.

Ported unchanged from the original project's analysis/key_model_experiments.py, where "band64+mel" was the best
feature set for the SVM (53.2% validation key accuracy vs 50.2% for v6's spectrum features; the CNN reached 60.3%
on the same split):

  * band64: log energy in 64 log-spaced bands (500 Hz - 20 kHz) over 30 ms around the press,
  * mel40x10: log-mel spectrogram (40 bands) from 10 ms before to 30 ms after the press, averaged to 10 time steps,

each with its mean subtracted per press, so loudness differences between presses and sessions matter less.
Input: 100 ms slices of 700 Hz high-passed audio centred on each press (classifier.slices_around).
"""
import numpy as np

from .classifier import SLICE as HALF
from .classifier import mel_filterbank
from .config import SAMPLE_RATE


def mel_patch(slices, start_ms=-10, end_ms=30, n_fft=256, hop=48, n_mels=40, time_bins=10):
    """(presses, time_bins, n_mels) log-mel energies."""
    first = HALF + start_ms * 48 - n_fft // 2
    frames = (end_ms - start_ms) * 48 // hop
    index = first + np.arange(frames)[:, None] * hop + np.arange(n_fft)[None, :]
    bank = mel_filterbank(n_fft, n_mels)
    out = np.empty((len(slices), time_bins, n_mels), np.float32)
    for start in range(0, len(slices), 2000):
        chunk = slices[start:start + 2000][:, index] * np.hanning(n_fft)
        power = np.abs(np.fft.rfft(chunk, axis=-1)) ** 2
        mel = np.log(power @ bank.T + 1e-10)
        out[start:start + 2000] = mel.reshape(len(chunk), time_bins, frames // time_bins, n_mels).mean(axis=2)
    return out


def band_energies(slices, start_ms=-8, end_ms=22, bands=64, n_fft=4096):
    """(presses, bands) log energies in log-spaced frequency bands."""
    clip = slices[:, HALF + start_ms * 48:HALF + end_ms * 48]
    power = np.abs(np.fft.rfft(clip * np.hanning(clip.shape[1]), n=n_fft, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    edges = np.geomspace(500, 20000, bands + 1)
    return np.stack([np.log(power[:, (freqs >= a) & (freqs < b)].sum(axis=1) + 1e-10)
                     for a, b in zip(edges[:-1], edges[1:])], axis=1).astype(np.float32)


def band_mel_features(slices):
    """The "band64+mel" feature vector: 64 + 400 values per press."""
    level = lambda f: f - f.mean(axis=1, keepdims=True)
    mel = mel_patch(slices)
    return np.hstack([level(band_energies(slices)), level(mel.reshape(len(mel), -1))])
