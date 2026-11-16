import numpy as np
import pytest

from vibeguard.constants import FEATURE_NAMES
from vibeguard.features import extract_features


def test_feature_extractor_has_fixed_finite_schema() -> None:
    sample_rate = 1024
    time = np.arange(512) / sample_rate
    signal = np.sin(2 * np.pi * 25 * time)
    features = extract_features(signal, fundamental_hz=25.0, sample_rate_hz=sample_rate)

    assert tuple(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())
    assert 0.0 <= features["spectral_entropy"] <= 1.0
    assert 0.0 <= features["fundamental_ratio"] <= 1.0


def test_feature_extractor_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        extract_features(np.ones(10), 25.0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        extract_features(np.full(64, np.nan), 25.0)
    with pytest.raises(ValueError, match="must be positive"):
        extract_features(np.ones(64), 0.0)


def test_constant_signal_does_not_divide_by_zero() -> None:
    features = extract_features(np.ones(128), fundamental_hz=20.0)
    assert all(np.isfinite(value) for value in features.values())
    assert features["crest_factor"] == 0.0
