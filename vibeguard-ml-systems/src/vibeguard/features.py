"""Numerically guarded time- and frequency-domain feature extraction."""

from __future__ import annotations

import numpy as np

from vibeguard.constants import FEATURE_NAMES, SAMPLE_RATE_HZ


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator / max(float(denominator), eps))


def extract_features(
    signal: np.ndarray,
    fundamental_hz: float,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> dict[str, float]:
    """Extract a stable, fixed-schema feature vector from one vibration window.

    A Hann window reduces spectral leakage. Spectral powers are normalized by total
    power so amplitude changes do not dominate every frequency feature.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 32:
        raise ValueError("A signal window must contain at least 32 samples.")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or infinite values.")
    if sample_rate_hz <= 0 or not np.isfinite(fundamental_hz) or fundamental_hz <= 0:
        raise ValueError("Sample rate and fundamental frequency must be positive.")

    centered = x - np.mean(x)
    rms = float(np.sqrt(np.mean(np.square(centered))))
    std = float(np.std(centered))
    abs_values = np.abs(centered)
    peak_abs = float(np.max(abs_values))
    mean_abs = float(np.mean(abs_values))
    eps = 1e-12
    skewness = float(np.mean(centered**3) / max(std**3, eps))
    kurtosis = float(np.mean(centered**4) / max(std**4, eps))
    zero_crossing_rate = float(np.mean(np.diff(np.signbit(centered))))

    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(centered * window)) ** 2
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    total_power = float(np.sum(spectrum))
    normalized_power = spectrum / max(total_power, eps)
    spectral_centroid = float(np.sum(frequencies * normalized_power))
    positive = normalized_power[1:]
    spectral_entropy = float(
        -np.sum(positive * np.log(positive + eps)) / np.log(max(positive.size, 2))
    )
    dominant_frequency = float(frequencies[int(np.argmax(spectrum[1:]) + 1)])
    cumulative = np.cumsum(normalized_power)
    rolloff_index = min(int(np.searchsorted(cumulative, 0.85)), frequencies.size - 1)
    rolloff = float(frequencies[rolloff_index])

    def band_power(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        return float(np.sum(normalized_power[mask]))

    def harmonic_power(target: float) -> float:
        width = max(3.0, sample_rate_hz / x.size * 2.0)
        mask = np.abs(frequencies - target) <= width
        return float(np.sum(normalized_power[mask]))

    features = {
        "rms": rms,
        "std": std,
        "mean_absolute": mean_abs,
        "peak_to_peak": float(np.ptp(centered)),
        "peak_absolute": peak_abs,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": _safe_ratio(peak_abs, rms),
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "dominant_frequency_hz": dominant_frequency,
        "spectral_rolloff_85_hz": rolloff,
        "band_power_0_60": band_power(0.0, 60.0),
        "band_power_60_140": band_power(60.0, 140.0),
        "band_power_140_260": band_power(140.0, 260.0),
        "band_power_260_nyquist": band_power(260.0, sample_rate_hz / 2.0 + 1.0),
        "fundamental_ratio": harmonic_power(fundamental_hz),
        "second_harmonic_ratio": harmonic_power(2.0 * fundamental_hz),
        "third_harmonic_ratio": harmonic_power(3.0 * fundamental_hz),
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("Feature implementation and schema constants are out of sync.")
    if not np.isfinite(np.asarray(list(features.values()), dtype=np.float64)).all():
        raise FloatingPointError("Feature extraction produced a non-finite value.")
    return features
