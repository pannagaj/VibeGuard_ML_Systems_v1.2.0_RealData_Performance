import csv
from pathlib import Path

import numpy as np

from vibeguard.constants import FEATURE_NAMES
from vibeguard.data import generate_dataset, load_dataset
from vibeguard.inference import Predictor
from vibeguard.training import train_project


def test_end_to_end_training_and_prediction(tmp_path: Path) -> None:
    data_path = generate_dataset(tmp_path / "motor.csv", samples_per_class=45, seed=11)
    output_dir = tmp_path / "artifacts"
    result = train_project(
        data_path,
        output_dir,
        epochs=5,
        batch_size=64,
        patience=3,
        seed=11,
        device_name="cpu",
        with_baseline=False,
    )

    assert (output_dir / "model.pt").is_file()
    assert (output_dir / "test_metrics.json").is_file()
    assert (output_dir / "training_history.csv").is_file()
    assert 0.0 <= result["summary"]["test_accuracy"] <= 1.0

    dataset = load_dataset(data_path)
    predictor = Predictor(output_dir / "model.pt")
    probabilities = predictor.predict_proba(dataset.features[:7])
    predictions = predictor.predict(dataset.features[:2])
    assert probabilities.shape == (7, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert len(predictions) == 2
    assert set(predictions[0]["probabilities"]) == {
        "healthy", "imbalance", "misalignment", "bearing_fault"
    }


def test_inference_rejects_wrong_feature_dimension(tmp_path: Path) -> None:
    data_path = generate_dataset(tmp_path / "motor.csv", samples_per_class=35, seed=12)
    output_dir = tmp_path / "artifacts"
    train_project(data_path, output_dir, epochs=2, batch_size=64, patience=2, with_baseline=False)
    predictor = Predictor(output_dir / "model.pt")

    try:
        predictor.predict_proba(np.zeros((1, len(FEATURE_NAMES) - 1), dtype=np.float32))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected a feature-dimension validation error.")
