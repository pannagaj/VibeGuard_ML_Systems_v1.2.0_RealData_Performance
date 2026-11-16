from pathlib import Path

from vibeguard.data import generate_dataset
from vibeguard.optimization import run_optimization_experiment
from vibeguard.training import train_project


def test_optimization_exports_compatible_models_and_reports_parity(tmp_path: Path) -> None:
    dataset_path = generate_dataset(tmp_path / "motor.csv", samples_per_class=35, seed=31)
    artifact_dir = tmp_path / "artifacts"
    train_project(
        dataset_path,
        artifact_dir,
        epochs=2,
        batch_size=64,
        patience=2,
        with_baseline=False,
    )
    output_dir = tmp_path / "optimization"
    report = run_optimization_experiment(
        artifact_dir / "model.pt",
        output_dir,
        runs=10,
        warmup=1,
        thread_count=1,
        data_path=dataset_path,
    )

    for filename in (
        "model_exported_fp32.pt2",
        "model_dynamic_int8.pt",
        "optimization_report.json",
    ):
        assert (output_dir / filename).is_file()
    assert report["held_out_test_evaluation"]["test_samples"] > 0
    assert "macro_f1" in report["held_out_test_evaluation"]["pytorch_dynamic_int8"]
    representations = report["representations"]
    assert set(representations) == {
        "pytorch_eager_fp32", "torch_export_fp32", "pytorch_dynamic_int8"
    }
    for item in representations.values():
        assert len(item["measurements"]) == 3
        assert item["measurements"][0]["p50_ms_per_batch"] >= 0
    quantized_drift = representations["pytorch_dynamic_int8"]["numerical_drift_vs_eager"]
    assert quantized_drift["max_absolute_probability_delta"] < 0.02
    assert quantized_drift["prediction_agreement"] >= 0.95
