"""Shared project constants and the versioned feature contract."""

SAMPLE_RATE_HZ = 1_024
WINDOW_SIZE = 512
CLASS_NAMES = ("healthy", "imbalance", "misalignment", "bearing_fault")
FEATURE_NAMES = (
    "rms",
    "std",
    "mean_absolute",
    "peak_to_peak",
    "peak_absolute",
    "skewness",
    "kurtosis",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_entropy",
    "dominant_frequency_hz",
    "spectral_rolloff_85_hz",
    "band_power_0_60",
    "band_power_60_140",
    "band_power_140_260",
    "band_power_260_nyquist",
    "fundamental_ratio",
    "second_harmonic_ratio",
    "third_harmonic_ratio",
)
SCHEMA_VERSION = "1.0"
