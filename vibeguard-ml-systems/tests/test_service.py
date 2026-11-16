from pathlib import Path

from fastapi.testclient import TestClient

from vibeguard.constants import FEATURE_NAMES
from vibeguard.data import generate_dataset, load_dataset
from vibeguard.service import create_app
from vibeguard.training import train_project


def test_health_metadata_and_prediction_endpoints(tmp_path: Path) -> None:
    data_path = generate_dataset(tmp_path / "motor.csv", samples_per_class=35, seed=22)
    output_dir = tmp_path / "artifacts"
    train_project(data_path, output_dir, epochs=2, batch_size=64, patience=2, with_baseline=False)
    app = create_app(output_dir / "model.pt")
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ready"
    metadata = client.get("/metadata")
    assert metadata.status_code == 200
    assert len(metadata.json()["feature_names"]) == len(FEATURE_NAMES)

    dataset = load_dataset(data_path)
    sample = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, dataset.features[0], strict=True)
    }
    response = client.post("/predict", json={"samples": [sample]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1

    invalid = client.post("/predict", json={"samples": [{"rms": 0.1}]})
    assert invalid.status_code == 422


def test_raw_window_prediction_endpoint(tmp_path: Path) -> None:
    import numpy as np

    from vibeguard.data import generate_signal_window

    data_path = generate_dataset(tmp_path / "motor.csv", samples_per_class=35, seed=23)
    output_dir = tmp_path / "artifacts"
    train_project(data_path, output_dir, epochs=2, batch_size=64, patience=2, with_baseline=False)
    signal, fundamental, _ = generate_signal_window("bearing_fault", seed=99)
    client = TestClient(create_app(output_dir / "model.pt"))
    response = client.post(
        "/predict-window",
        json={
            "signal": np.asarray(signal).tolist(),
            "fundamental_hz": fundamental,
            "sample_rate_hz": 1024,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["features"]) == set(FEATURE_NAMES)
    assert payload["prediction"]["predicted_label"] in {
        "healthy", "imbalance", "misalignment", "bearing_fault"
    }

    invalid = client.post(
        "/predict-window",
        json={"signal": [0.0] * 31, "fundamental_hz": fundamental},
    )
    assert invalid.status_code == 422
