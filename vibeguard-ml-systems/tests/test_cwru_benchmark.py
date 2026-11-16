from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from experiments.cwru_benchmark.data import (
    CLASS_NAMES,
    RECORDINGS,
    _load_drive_end_signal,
    load_prepared_dataset,
    prepare_dataset,
    split_for_load,
    windows_from_signal,
)
from vibeguard.constants import FEATURE_NAMES


def test_load_split_is_fixed_and_separates_motor_conditions() -> None:
    assert [split_for_load(load) for load in range(4)] == [
        "train", "train", "validation", "test"
    ]
    with pytest.raises(ValueError):
        split_for_load(4)


def test_recording_manifest_has_four_classes_and_four_loads() -> None:
    assert len(RECORDINGS) == 16
    assert {row.class_name for row in RECORDINGS} == set(CLASS_NAMES)
    assert {row.load_hp for row in RECORDINGS} == {0, 1, 2, 3}


def test_windowing_is_contiguous_and_rejects_bad_values() -> None:
    signal = np.arange(10_240, dtype=np.float64)
    windows = windows_from_signal(signal)
    assert windows.shape == (4, 4_096)
    assert np.array_equal(windows[1], signal[2_048:6_144])
    with pytest.raises(ValueError):
        windows_from_signal(np.ones(100))
    bad = np.ones(5_000)
    bad[10] = np.nan
    with pytest.raises(ValueError):
        windows_from_signal(bad)


def test_loadmat_reader_finds_drive_end_channel(tmp_path) -> None:
    target = tmp_path / "97.mat"
    signal = np.linspace(-1, 1, 5_000)
    savemat(target, {"X097_DE_time": signal.reshape(-1, 1)})
    loaded = _load_drive_end_signal(target)
    assert loaded.shape == signal.shape
    assert np.allclose(loaded, signal)


def test_prepare_dataset_keeps_recordings_in_one_split(tmp_path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "out"
    raw.mkdir()
    t = np.arange(8_192, dtype=np.float64) / 12_000
    for recording in RECORDINGS:
        class_idx = CLASS_NAMES.index(recording.class_name)
        frequency = 30.0 + class_idx * 80.0
        signal = np.sin(2 * np.pi * frequency * t) + 0.05 * np.sin(2 * np.pi * 700 * t)
        savemat(raw / f"{recording.file_number}.mat", {
            f"X{recording.file_number:03d}_DE_time": signal.reshape(-1, 1)
        })
    summary = prepare_dataset(raw, output, download=False)
    dataset = load_prepared_dataset(output / "cwru_features.npz")
    assert dataset["features"].shape[1] == len(FEATURE_NAMES)
    assert dataset["labels"].size == 16 * 3
    for recording_id in np.unique(dataset["recording_ids"]):
        assert np.unique(dataset["splits"][dataset["recording_ids"] == recording_id]).size == 1
    assert summary["split_counts"] == {"train": 24, "validation": 12, "test": 12}
    assert summary["raw_files_redistributed"] is False


def test_prepared_dataset_loader_rejects_wrong_feature_dimension(tmp_path) -> None:
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path,
        features=np.zeros((2, 3), dtype=np.float32),
        labels=np.zeros(2, dtype=np.int64),
        splits=np.asarray(["train", "test"]),
        recording_ids=np.asarray(["a", "b"]),
        load_hp=np.asarray([0, 3]),
        window_indices=np.asarray([0, 0]),
    )
    with pytest.raises(ValueError, match="feature contract"):
        load_prepared_dataset(path)


def test_training_writes_weights_only_reloadable_checkpoint(tmp_path) -> None:
    from experiments.cwru_benchmark.train import train_experiment
    import torch

    rng = np.random.default_rng(42)
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    split_names: list[str] = []
    recording_ids: list[str] = []
    load_values: list[int] = []
    window_indices: list[int] = []
    for split_name, loads, rows_per_recording in (
        ("train", (0, 1), 6),
        ("validation", (2,), 8),
        ("test", (3,), 8),
    ):
        for class_index in range(len(CLASS_NAMES)):
            for load in loads:
                recording_id = f"class{class_index}-load{load}"
                rows = rng.normal(
                    loc=float(class_index) * 1.5,
                    scale=0.2,
                    size=(rows_per_recording, len(FEATURE_NAMES)),
                ).astype(np.float32)
                feature_rows.extend(rows)
                labels.extend([class_index] * rows_per_recording)
                split_names.extend([split_name] * rows_per_recording)
                recording_ids.extend([recording_id] * rows_per_recording)
                load_values.extend([load] * rows_per_recording)
                window_indices.extend(range(rows_per_recording))

    dataset_path = tmp_path / "training_smoke.npz"
    np.savez_compressed(
        dataset_path,
        features=np.asarray(feature_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        splits=np.asarray(split_names),
        recording_ids=np.asarray(recording_ids),
        load_hp=np.asarray(load_values, dtype=np.int64),
        window_indices=np.asarray(window_indices, dtype=np.int64),
    )
    output = tmp_path / "artifacts"
    result = train_experiment(
        dataset_path,
        output,
        seed=42,
        epochs=2,
        batch_size=32,
        patience=2,
    )
    payload = torch.load(output / "cwru_mlp.pt", map_location="cpu", weights_only=True)
    assert result["rows"] == {"train": 48, "validation": 32, "test": 32}
    assert payload["metadata"]["split_protocol"]["test_load_hp"] == [3]
    assert isinstance(payload["scaler_mean"], list)
    assert isinstance(payload["scaler_scale"], list)
    assert (output / "cwru_evaluation.json").is_file()
