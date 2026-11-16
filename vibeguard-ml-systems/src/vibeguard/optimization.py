"""CPU inference optimization experiments: eager, torch.export and dynamic INT8.

The experiment is comparative and measurement-driven. It reports the actual latency
and numerical drift observed on the current host; it never assumes that an optimized
representation must be faster for this small model.
"""

from __future__ import annotations

import copy
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from vibeguard.artifacts import atomic_torch_save, load_model_artifact, write_json
from vibeguard.data import load_dataset, split_dataset
from vibeguard.metrics import classification_metrics


class StandardizedInferenceModel(nn.Module):
    """Wrap the network and training-fitted scaler into one raw-feature model."""

    def __init__(self, network: nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("scaler_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.scaler_mean) / self.scaler_scale
        return self.network(standardized)


def _quantize_network(network: nn.Module) -> tuple[nn.Module, str]:
    """Prefer TorchAO when installed; fall back to PyTorch eager dynamic quantization.

    The fallback is kept for environments that have PyTorch but have not installed
    the optional TorchAO package. New deployments should use a TorchAO/PyTorch pair
    that is validated for their target hardware.
    """
    candidate = copy.deepcopy(network).eval()
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        quantize_(candidate, Int8DynamicActivationInt8WeightConfig(version=2))
        return candidate, "torchao.Int8DynamicActivationInt8WeightConfig(v2)"
    except (ImportError, AttributeError, RuntimeError, ValueError, NotImplementedError, TypeError):
        legacy_api = getattr(getattr(torch.ao, "quantization", None), "quantize_dynamic", None)
        if legacy_api is None:
            raise RuntimeError(
                "TorchAO dynamic INT8 is unavailable and this PyTorch build no longer exposes "
                "the legacy quantize_dynamic API. Install a compatible project extra with "
                "`python -m pip install -e '.[quantization]'`."
            )
        fallback = copy.deepcopy(network).eval()
        quantized = legacy_api(fallback, {nn.Linear}, dtype=torch.qint8)
        return quantized, "torch.ao.quantization.quantize_dynamic (compatibility fallback)"


def load_dynamic_int8_artifact(
    training_checkpoint: str | Path,
    optimized_checkpoint: str | Path,
) -> StandardizedInferenceModel:
    """Reconstruct the dynamic-INT8 model from a safe weights-only state artifact."""
    source_model, metadata = load_model_artifact(training_checkpoint, device="cpu")
    quantized_network, quantization_backend = _quantize_network(source_model)
    pipeline = StandardizedInferenceModel(
        quantized_network, metadata["scaler_mean"], metadata["scaler_scale"]
    ).eval()
    payload = torch.load(optimized_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "vibeguard-dynamic-int8-v1":
        raise ValueError("Unsupported dynamic INT8 artifact format.")
    if payload.get("source_schema_version") != metadata["schema_version"]:
        raise ValueError("Optimized model and source checkpoint schema versions differ.")
    if payload.get("quantization_backend") != quantization_backend:
        raise ValueError(
            "Current quantization backend differs from the backend used to export this artifact."
        )
    pipeline.load_state_dict(payload["state_dict"], strict=True)
    return pipeline


def _predict_probabilities_in_chunks(
    model: nn.Module, features: torch.Tensor, chunk_size: int = 256
) -> np.ndarray:
    """Run an inference module without exceeding exported dynamic-batch bounds."""
    if features.ndim != 2 or features.shape[0] == 0 or chunk_size < 1:
        raise ValueError("features must be a non-empty matrix and chunk_size must be positive.")
    probability_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], chunk_size):
            logits = model(features[start : start + chunk_size])
            probability_chunks.append(logits.softmax(dim=1).cpu())
    return torch.cat(probability_chunks, dim=0).numpy()


def _measure(
    forward: Callable[[torch.Tensor], torch.Tensor],
    batches: dict[int, torch.Tensor],
    *,
    runs: int,
    warmup: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size, batch in batches.items():
        with torch.inference_mode():
            for _ in range(warmup):
                forward(batch)
            samples: list[float] = []
            started = time.perf_counter()
            for _ in range(runs):
                tick = time.perf_counter()
                forward(batch)
                samples.append((time.perf_counter() - tick) * 1_000.0)
            elapsed = time.perf_counter() - started
        values = np.asarray(samples, dtype=np.float64)
        rows.append(
            {
                "batch_size": batch_size,
                "runs": runs,
                "p50_ms_per_batch": float(np.percentile(values, 50)),
                "p95_ms_per_batch": float(np.percentile(values, 95)),
                "mean_ms_per_batch": float(np.mean(values)),
                "throughput_samples_per_second": float(batch_size * runs / max(elapsed, 1e-12)),
            }
        )
    return rows


def run_optimization_experiment(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    runs: int = 300,
    warmup: int = 40,
    seed: int = 42,
    thread_count: int = 1,
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export and benchmark CPU inference representations with parity checks."""
    if runs < 10 or warmup < 1 or thread_count < 1:
        raise ValueError("runs must be >=10; warmup and thread_count must be positive.")

    torch.set_num_threads(thread_count)
    model, metadata = load_model_artifact(checkpoint, device="cpu")
    model.eval()
    dimension = int(metadata["input_dim"])
    rng = np.random.default_rng(seed)
    mean = np.asarray(metadata["scaler_mean"], dtype=np.float32)
    scale = np.asarray(metadata["scaler_scale"], dtype=np.float32)
    batches = {
        size: torch.from_numpy(
            (
                rng.normal(size=(size, dimension)).astype(np.float32) * scale + mean
            ).astype(np.float32)
        )
        for size in (1, 32, 128)
    }
    parity_input = batches[32]

    # Keep preprocessing inside the exported model: all variants accept the same
    # raw engineered features as Predictor and the REST API.
    eager_pipeline = StandardizedInferenceModel(model, mean, scale).eval()
    example = torch.zeros(2, dimension, dtype=torch.float32)
    dynamic_batch = torch.export.Dim("batch", min=1, max=256)
    exported_program = torch.export.export(
        eager_pipeline,
        (example,),
        dynamic_shapes=({0: dynamic_batch},),
    )
    exported_path = Path(output_dir) / "model_exported_fp32.pt2"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, exported_path)
    exported_pipeline = torch.export.load(exported_path).module()

    # Dynamic weight-only INT8 quantization is CPU-only. The quantized state dict is
    # saved with its source schema so it can be reconstructed without pickle loading.
    quantized_network, quantization_backend = _quantize_network(model)
    quantized_pipeline = StandardizedInferenceModel(quantized_network, mean, scale).eval()

    with torch.inference_mode():
        reference = eager_pipeline(parity_input).softmax(dim=1).cpu().numpy()
        exported_probabilities = exported_pipeline(parity_input).softmax(dim=1).cpu().numpy()
        int8_probabilities = quantized_pipeline(parity_input).softmax(dim=1).cpu().numpy()

    def drift(candidate: np.ndarray) -> dict[str, float | bool]:
        difference = np.abs(reference - candidate)
        return {
            "max_absolute_probability_delta": float(difference.max()),
            "mean_absolute_probability_delta": float(difference.mean()),
            "prediction_agreement": float(
                np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
            ),
        }

    int8_path = Path(output_dir) / "model_dynamic_int8.pt"
    atomic_torch_save(
        {
            "format": "vibeguard-dynamic-int8-v1",
            "source_schema_version": metadata["schema_version"],
            "quantization_backend": quantization_backend,
            "state_dict": quantized_pipeline.state_dict(),
        },
        int8_path,
    )
    # Reload the persisted artifact and verify that the saved representation behaves
    # the same as the in-memory candidate before declaring the export complete.
    reloaded_int8 = load_dynamic_int8_artifact(checkpoint, int8_path)
    with torch.inference_mode():
        persisted_probabilities = reloaded_int8(parity_input).softmax(dim=1).cpu().numpy()
    persisted_delta = float(np.max(np.abs(int8_probabilities - persisted_probabilities)))
    if persisted_delta > 1e-7:
        raise RuntimeError(
            f"Persisted INT8 artifact changed outputs (max delta {persisted_delta})."
        )

    quality_evaluation: dict[str, Any] | None = None
    if data_path is not None:
        dataset = load_dataset(data_path)
        test_split = split_dataset(dataset, seed=int(metadata["seed"])).test
        test_tensor = torch.from_numpy(np.ascontiguousarray(test_split.features, dtype=np.float32))
        eager_test_probabilities = _predict_probabilities_in_chunks(eager_pipeline, test_tensor)
        export_test_probabilities = _predict_probabilities_in_chunks(exported_pipeline, test_tensor)
        int8_test_probabilities = _predict_probabilities_in_chunks(reloaded_int8, test_tensor)
        quality_evaluation = {
            "evaluation_split": "held-out test split; split seed from source checkpoint",
            "data_path": str(data_path),
            "test_samples": int(test_split.labels.size),
            "pytorch_eager_fp32": classification_metrics(
                test_split.labels, eager_test_probabilities
            ),
            "torch_export_fp32": classification_metrics(
                test_split.labels, export_test_probabilities
            ),
            "pytorch_dynamic_int8": classification_metrics(
                test_split.labels, int8_test_probabilities
            ),
            "max_absolute_probability_delta_vs_eager": {
                "torch_export_fp32": float(
                    np.max(np.abs(eager_test_probabilities - export_test_probabilities))
                ),
                "pytorch_dynamic_int8": float(
                    np.max(np.abs(eager_test_probabilities - int8_test_probabilities))
                ),
            },
            "prediction_agreement_vs_eager": {
                "torch_export_fp32": float(
                    np.mean(
                        np.argmax(eager_test_probabilities, axis=1)
                        == np.argmax(export_test_probabilities, axis=1)
                    )
                ),
                "pytorch_dynamic_int8": float(
                    np.mean(
                        np.argmax(eager_test_probabilities, axis=1)
                        == np.argmax(int8_test_probabilities, axis=1)
                    )
                ),
            },
        }

    output = Path(output_dir)
    eager_state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in eager_pipeline.state_dict().values()
    )
    experiment = {
        "experiment": "CPU inference representation comparison",
        "seed": seed,
        "device": "cpu",
        "torch_threads": thread_count,
        "quantization_backend": quantization_backend,
        "runs_per_batch_size": runs,
        "warmup_iterations_per_batch_size": warmup,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported by host",
            "logical_cpu_count": __import__("os").cpu_count(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        },
        "held_out_test_evaluation": quality_evaluation,
        "representations": {
            "pytorch_eager_fp32": {
                "measurements": _measure(eager_pipeline, batches, runs=runs, warmup=warmup),
                "parameter_state_bytes": int(eager_state_bytes),
                "numerical_drift_vs_eager": {
                    "max_absolute_probability_delta": 0.0,
                    "mean_absolute_probability_delta": 0.0,
                    "prediction_agreement": 1.0,
                },
            },
            "torch_export_fp32": {
                "measurements": _measure(exported_pipeline, batches, runs=runs, warmup=warmup),
                "artifact_bytes": exported_path.stat().st_size,
                "numerical_drift_vs_eager": drift(exported_probabilities),
            },
            "pytorch_dynamic_int8": {
                "measurements": _measure(quantized_pipeline, batches, runs=runs, warmup=warmup),
                "artifact_bytes": int8_path.stat().st_size,
                "numerical_drift_vs_eager": drift(int8_probabilities),
                "persisted_artifact_max_probability_delta": persisted_delta,
            },
        },
        "artifacts": {
            "torch_export_fp32": str(exported_path),
            "pytorch_dynamic_int8": str(int8_path),
        },
        "interpretation": (
            "All representations accept raw engineered feature vectors and include the "
            "training-fitted standardization step. Compare measured p50/p95 latency and "
            "throughput on the same machine. Small models can be dominated by dispatch "
            "overhead; export or INT8 may not be faster for every batch size."
        ),
    }
    write_json(output / "optimization_report.json", experiment)
    return experiment
