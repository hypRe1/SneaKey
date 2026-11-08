"""WAV input/output and the filters the pipeline uses."""
import io
import wave
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import SAMPLE_RATE


class AudioFormatError(ValueError):
    pass


def read_wav(source):
    """Read a mono 16-bit 48 kHz WAV (path or bytes). Returns float samples in [-1, 1), as v6 does."""
    data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    rate, raw = wavfile.read(data)
    if rate != SAMPLE_RATE or raw.ndim != 1 or raw.dtype != np.int16:
        raise AudioFormatError("Expected mono 16-bit PCM WAV at 48 kHz.")
    if len(raw) < 1536:
        raise AudioFormatError("Recording must be at least 32 ms long.")
    return raw.astype(float) / 32768


def wav_bytes(samples, sample_rate=SAMPLE_RATE):
    """Encode float samples (or int16) as a mono 16-bit WAV."""
    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        samples = np.clip(np.round(samples * 32768), -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


@lru_cache(maxsize=None)
def highpass_sos(cutoff_hz, order=3):
    """Butterworth high-pass, as used throughout v6 (always order 3, applied zero-phase with sosfiltfilt)."""
    return signal.butter(order, cutoff_hz, "highpass", fs=SAMPLE_RATE, output="sos")


def highpass(samples, cutoff_hz):
    return signal.sosfiltfilt(highpass_sos(cutoff_hz), samples)
