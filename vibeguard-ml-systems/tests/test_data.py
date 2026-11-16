from pathlib import Path

import numpy as np
import pytest

from vibeguard.constants import CLASS_NAMES, FEATURE_NAMES
from vibeguard.data import generate_dataset, load_dataset, split_dataset


def test_dataset_generation_is_reproducible_and_balanced(tmp_path: Path) -> None:
    first = generate_dataset(tmp_path / "first.csv", samples_per_class=35, seed=17)
    second = generate_dataset(tmp_path / "second.csv", samples_per_class=35, seed=17)
    assert first.read_bytes() == second.read_bytes()

    dataset = load_dataset(first)
    assert dataset.features.shape == (35 * len(CLASS_NAMES), len(FEATURE_NAMES))
    assert np.isfinite(dataset.features).all()
    assert all(np.sum(dataset.labels == index) == 35 for index in range(len(CLASS_NAMES)))


def test_split_is_stratified_and_has_no_leakage(tmp_path: Path) -> None:
    path = generate_dataset(tmp_path / "dataset.csv", samples_per_class=40, seed=3)
    dataset = load_dataset(path)
    splits = split_dataset(dataset, seed=9)
    ids = [set(split.sample_ids.tolist()) for split in splits]

    assert len(ids[0] & ids[1]) == 0
    assert len(ids[0] & ids[2]) == 0
    assert len(ids[1] & ids[2]) == 0
    assert sum(map(len, ids)) == dataset.labels.size
    for split in splits:
        assert set(np.unique(split.labels)) == set(range(len(CLASS_NAMES)))


def test_dataset_rejects_too_few_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 30"):
        generate_dataset(tmp_path / "invalid.csv", samples_per_class=10)
