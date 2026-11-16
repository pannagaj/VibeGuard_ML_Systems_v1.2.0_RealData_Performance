"""Synthetic motor-vibration data generation, validation, and splitting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import train_test_split

from vibeguard.constants import (
    CLASS_NAMES,
    FEATURE_NAMES,
    SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
    WINDOW_SIZE,
)
from vibeguard.features import extract_features


class Dataset(NamedTuple):
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray


class DatasetSplits(NamedTuple):
    train: Dataset
    validation: Dataset
    test: Dataset


def _make_signal(class_name: str, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Create one synthetic sensor window with class-specific fault signatures."""
    fs = SAMPLE_RATE_HZ
    time = np.arange(WINDOW_SIZE, dtype=np.float64) / fs
    rpm = float(rng.uniform(1_080.0, 2_100.0))
    fundamental = rpm / 60.0
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase3 = float(rng.uniform(0.0, 2.0 * np.pi))

    if class_name == "healthy":
        amp1, amp2, amp3, noise = 0.22, 0.025, 0.010, 0.055
    elif class_name == "imbalance":
        amp1 = float(rng.uniform(0.75, 1.05))
        amp2, amp3, noise = 0.045, 0.012, 0.060
    elif class_name == "misalignment":
        amp1 = float(rng.uniform(0.25, 0.42))
        amp2 = float(rng.uniform(0.62, 0.85))
        amp3, noise = 0.20, 0.065
    elif class_name == "bearing_fault":
        amp1, amp2, amp3, noise = 0.31, 0.075, 0.025, 0.075
    else:
        raise ValueError(f"Unknown class: {class_name}")

    signal = (
        amp1 * np.sin(2.0 * np.pi * fundamental * time + phase)
        + amp2 * np.sin(2.0 * np.pi * 2.0 * fundamental * time + phase2)
        + amp3 * np.sin(2.0 * np.pi * 3.0 * fundamental * time + phase3)
    )
    signal += 0.025 * np.sin(2.0 * np.pi * 0.5 * fundamental * time + phase / 2.0)

    if class_name == "bearing_fault":
        # Randomly timed decaying resonant impulses mimic repetitive bearing impacts.
        event_count = int(rng.integers(5, 10))
        positions = rng.integers(8, WINDOW_SIZE - 24, size=event_count)
        resonance = float(rng.uniform(150.0, 215.0))
        decay = np.exp(-np.arange(24, dtype=np.float64) / rng.uniform(3.0, 6.0))
        for position in positions:
            stop = min(position + decay.size, WINDOW_SIZE)
            pulse_time = np.arange(stop - position, dtype=np.float64) / fs
            pulse = rng.uniform(0.35, 0.75) * decay[: stop - position] * np.sin(
                2.0 * np.pi * resonance * pulse_time + rng.uniform(0.0, 2.0 * np.pi)
            )
            signal[position:stop] += pulse

    signal += rng.normal(0.0, noise, size=WINDOW_SIZE)
    signal += rng.normal(0.0, 0.008)  # Sensor offset.
    return signal.astype(np.float64), fundamental, rpm


def generate_signal_window(
    class_name: str, seed: int = 42
) -> tuple[np.ndarray, float, float]:
    """Generate one deterministic synthetic sensor window for examples and API demos."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"class_name must be one of {CLASS_NAMES}.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return _make_signal(class_name, np.random.default_rng(seed))


def generate_dataset(
    output_path: str | Path,
    samples_per_class: int = 1_000,
    seed: int = 42,
) -> Path:
    """Generate a deterministic, labeled synthetic dataset and write a CSV file."""
    if samples_per_class < 30:
        raise ValueError("samples_per_class must be at least 30 for train/validation/test splits.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fieldnames = ["sample_id", "rpm", "sample_rate_hz", "schema_version", *FEATURE_NAMES, "label"]
    rows: list[dict[str, str | int | float]] = []
    sample_id = 0
    for class_name in CLASS_NAMES:
        for _ in range(samples_per_class):
            signal, fundamental, rpm = _make_signal(class_name, rng)
            features = extract_features(signal, fundamental)
            rows.append(
                {
                    "sample_id": sample_id,
                    "rpm": round(rpm, 5),
                    "sample_rate_hz": SAMPLE_RATE_HZ,
                    "schema_version": SCHEMA_VERSION,
                    **features,
                    "label": class_name,
                }
            )
            sample_id += 1
    rng.shuffle(rows)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def load_dataset(path: str | Path) -> Dataset:
    """Load and validate the versioned feature CSV; reject malformed data early."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    with source.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {"sample_id", *FEATURE_NAMES, "label"}
        actual = set(reader.fieldnames or [])
        missing = expected - actual
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        feature_rows: list[list[float]] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        for line_number, row in enumerate(reader, start=2):
            version = row.get("schema_version")
            if version and version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported feature schema {version!r} on CSV line {line_number}; "
                    f"expected {SCHEMA_VERSION}."
                )
            label = row.get("label", "")
            if label not in class_to_index:
                raise ValueError(f"Unknown label {label!r} on CSV line {line_number}.")
            try:
                vector = [float(row[name]) for name in FEATURE_NAMES]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric feature on CSV line {line_number}.") from exc
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite feature on CSV line {line_number}.")
            feature_rows.append(vector)
            labels.append(class_to_index[label])
            sample_ids.append(row["sample_id"])

    if not feature_rows:
        raise ValueError("Dataset contains no rows.")
    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(sample_ids, dtype=str)
    class_counts = np.bincount(y, minlength=len(CLASS_NAMES))
    if np.any(class_counts < 5):
        raise ValueError("Every class must contain at least five samples for stratified splitting.")
    if np.unique(ids).size != ids.size:
        raise ValueError(
            "sample_id values must be unique; duplicate records could leak across splits."
        )
    return Dataset(X, y, ids)


def split_dataset(dataset: Dataset, seed: int = 42) -> DatasetSplits:
    """Return a stratified 64/16/20 split with no shared sample IDs."""
    indices = np.arange(dataset.labels.size)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=seed, stratify=dataset.labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=seed,
        stratify=dataset.labels[train_val_idx],
    )

    def take(index: np.ndarray) -> Dataset:
        return Dataset(dataset.features[index], dataset.labels[index], dataset.sample_ids[index])

    splits = DatasetSplits(take(train_idx), take(val_idx), take(test_idx))
    all_ids = np.concatenate([split.sample_ids for split in splits])
    if np.unique(all_ids).size != all_ids.size:
        raise RuntimeError("Data leakage detected: a sample appears in multiple splits.")
    return splits
