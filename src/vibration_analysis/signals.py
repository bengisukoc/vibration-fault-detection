"""Simulate and analyse representative machinery vibration signals."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis


def generate_signal(condition: str, fs: int = 12_000, duration: float = 2.0,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate normal, imbalance or bearing-fault acceleration data."""
    if fs <= 0 or duration <= 0:
        raise ValueError("fs and duration must be positive")
    condition = condition.lower()
    if condition not in {"normal", "imbalance", "bearing_fault"}:
        raise ValueError("Unknown condition")
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    shaft_frequency = 30.0
    signal = 0.18 * np.sin(2 * np.pi * shaft_frequency * t)
    signal += 0.04 * np.sin(2 * np.pi * 2 * shaft_frequency * t + 0.4)

    if condition == "imbalance":
        signal += 0.78 * np.sin(2 * np.pi * shaft_frequency * t + 0.2)
    elif condition == "bearing_fault":
        impact_frequency = 120.0
        resonance = 2_500.0
        decay = 1_200.0
        samples_per_impact = max(1, int(fs / impact_frequency))
        response_length = int(0.006 * fs)
        response_time = np.arange(response_length) / fs
        impulse_response = np.exp(-decay * response_time) * np.sin(2 * np.pi * resonance * response_time)
        impacts = np.zeros_like(t)
        impacts[::samples_per_impact] = 1.2
        signal += np.convolve(impacts, impulse_response, mode="full")[:len(t)]

    signal += rng.normal(0, 0.06, size=len(t))
    return t, signal


def _spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(signal))
    amplitude = 2 * np.abs(np.fft.rfft((signal - signal.mean()) * window)) / window.sum()
    frequency = np.fft.rfftfreq(len(signal), d=1 / fs)
    return frequency, amplitude


def analyse_signal(signal: np.ndarray, fs: int) -> dict:
    """Return time features, FFT and envelope spectrum for one signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 32:
        raise ValueError("signal must be a one-dimensional array with at least 32 samples")
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    frequency, amplitude = _spectrum(signal, fs)
    nonzero = frequency > 1
    dominant = float(frequency[nonzero][np.argmax(amplitude[nonzero])])

    # Isolate the resonance band, then demodulate its amplitude with the
    # analytic signal. Repeating impacts appear in the envelope spectrum.
    sos = butter(4, [1500, 3500], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(filtered))
    env_frequency, env_amplitude = _spectrum(envelope, fs)
    env_band = (env_frequency >= 20) & (env_frequency <= 500)
    envelope_peak = float(env_frequency[env_band][np.argmax(env_amplitude[env_band])])
    resonance_band = (frequency >= 1500) & (frequency <= 3500)
    resonance_energy = float(np.trapezoid(amplitude[resonance_band] ** 2,
                                          frequency[resonance_band]))
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / rms,
        "kurtosis": float(kurtosis(signal, fisher=False, bias=False)),
        "dominant_frequency_hz": dominant,
        "resonance_energy": resonance_energy,
        "envelope_peak_hz": envelope_peak,
        "frequency": frequency,
        "amplitude": amplitude,
        "envelope_frequency": env_frequency,
        "envelope_amplitude": env_amplitude,
    }


def classify_condition(features: dict) -> str:
    """Apply transparent demonstration thresholds to extracted features."""
    if features["resonance_energy"] > 0.01 or features["kurtosis"] > 4.5:
        return "bearing_fault"
    if features["rms"] > 0.45 and 25 <= features["dominant_frequency_hz"] <= 35:
        return "imbalance"
    return "normal"

