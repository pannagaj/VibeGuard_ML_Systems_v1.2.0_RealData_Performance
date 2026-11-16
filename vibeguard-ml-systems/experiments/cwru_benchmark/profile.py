"""Profile and compare eager FP32 / dynamic INT8 CWRU inference paths."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn

from experiments.cwru_benchmark.data import load_prepared_dataset
from vibeguard.model import VibrationMLP
from vibeguard.optimization import _quantize_network


def _load_checkpoint(path: str | Path) -> tuple[nn.Module, dict[str, Any], np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    model = VibrationMLP(
        int(metadata["input_dim"]),
        len(metadata["class_names"]),
        hidden_dims=tuple(metadata["hidden_dims"]),
        dropout=float(metadata["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
    if mean.shape != (int(metadata["input_dim"]),) or scale.shape != mean.shape:
        raise ValueError("Checkpoint scaler dimensions do not match the model input dimension.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Checkpoint scaler contains invalid values.")
    return model, metadata, mean, scale


def _measure(
    model: nn.Module,
    batches: dict[int, torch.Tensor],
    runs: int,
    warmup: int,
) -> list[dict[str, Any]]:
    results = []
    model.eval()
    with torch.inference_mode():
        for batch_size, batch in batches.items():
            for _ in range(warmup):
                model(batch)
            durations = []
            start_all = time.perf_counter()
            for _ in range(runs):
                start = time.perf_counter()
                model(batch)
                durations.append((time.perf_counter() - start) * 1_000.0)
            total_seconds = time.perf_counter() - start_all
            results.append(
                {
                "batch_size": batch_size,
                "runs": runs,
                "latency_p50_ms_per_batch": float(np.percentile(durations, 50)),
                "latency_p95_ms_per_batch": float(np.percentile(durations, 95)),
                "latency_mean_ms_per_batch": float(statistics.fmean(durations)),
                "throughput_samples_per_second": float(
                    batch_size * runs / max(total_seconds, 1e-12)
                ),
                }
            )
    return results


def profile_experiment(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    runs: int = 300,
    warmup: int = 40,
    threads: int = 1,
) -> dict[str, Any]:
    if runs < 20 or warmup < 1 or threads < 1:
        raise ValueError("runs must be at least 20; warmup and threads must be positive.")
    torch.set_num_threads(threads)
    model, metadata, mean, scale = _load_checkpoint(checkpoint_path)
    dataset = load_prepared_dataset(dataset_path)
    test_mask = dataset["splits"] == "test"
    X_test = dataset["features"][test_mask]
    if X_test.size == 0:
        raise ValueError("Test split is empty; train the CWRU experiment first.")
    standardized = ((X_test - mean) / scale).astype(np.float32)
    batches = {
        size: torch.from_numpy(
            standardized[np.arange(size) % standardized.shape[0]].copy()
        )
        for size in (1, 32, 128)
    }

    eager = copy.deepcopy(model).eval()
    int8, quantization_backend = _quantize_network(model)
    with torch.inference_mode():
        eager_probs = torch.softmax(eager(torch.from_numpy(standardized)), dim=1).numpy()
        int8_probs = torch.softmax(int8(torch.from_numpy(standardized)), dim=1).numpy()
    drift = np.abs(eager_probs - int8_probs)
    parity = float(np.mean(np.argmax(eager_probs, axis=1) == np.argmax(int8_probs, axis=1)))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    int8_path = output / "cwru_mlp_dynamic_int8_state.pt"
    torch.save(int8.state_dict(), int8_path)
    persisted_state = torch.load(int8_path, map_location="cpu", weights_only=True)
    reloaded_int8, reload_backend = _quantize_network(model)
    if reload_backend != quantization_backend:
        raise RuntimeError("Quantization backend changed while reloading the INT8 state dict.")
    reloaded_int8.load_state_dict(persisted_state, strict=True)
    with torch.inference_mode():
        reloaded_probs = torch.softmax(
            reloaded_int8(torch.from_numpy(standardized)), dim=1
        ).numpy()
    reload_delta = float(np.max(np.abs(int8_probs - reloaded_probs)))
    if reload_delta > 1e-7:
        raise RuntimeError(f"Reloaded INT8 artifact changed probabilities by {reload_delta}.")

    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    # Generate a short operator trace for diagnosis. The trace is not used for timing.
    trace_path = output / "torch_profiler_trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(10):
                eager(batches[32])
    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    (output / "torch_profiler_summary.txt").write_text(table + "\n", encoding="utf-8")
    rss_after = int(process.memory_info().rss)

    state_bytes = sum(t.numel() * t.element_size() for t in eager.state_dict().values())
    int8_state_bytes = int8_path.stat().st_size
    report = {
        "experiment": "CWRU CPU inference profiling and dynamic INT8 comparison",
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
            "logical_cpu_count": __import__("os").cpu_count(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torch_threads": threads,
        },
        "quantization_backend": quantization_backend,
        "benchmark": {
            "runs_per_batch": runs,
            "warmup_per_batch": warmup,
            "input": (
                "Actual held-out CWRU test features after the train-fitted scaler; "
                "repeated only if needed to fill a batch."
            ),
            "representations": {
                "eager_fp32": _measure(eager, batches, runs, warmup),
                "dynamic_int8": _measure(reloaded_int8, batches, runs, warmup),
            },
            "numerical_parity_on_held_out_cwru_test": {
                "prediction_agreement": parity,
                "max_absolute_probability_delta": float(drift.max()),
                "mean_absolute_probability_delta": float(drift.mean()),
            },
            "state_size_bytes": {
                "eager_fp32": int(state_bytes),
                "dynamic_int8_file": int8_state_bytes,
            },
            "int8_state_reload_max_probability_delta": reload_delta,
        },
        "memory_observation": {
            "process_rss_before_profiler_bytes": rss_before,
            "process_rss_after_profiler_bytes": rss_after,
            "rss_delta_bytes": rss_after - rss_before,
            "note": (
                "Process RSS delta is a coarse whole-process observation, "
                "not isolated model memory."
            ),
        },
        "profiler": {
            "trace": str(trace_path),
            "summary": str(output / "torch_profiler_summary.txt"),
            "activities": ["CPU"],
            "records_shapes": True,
            "profile_memory": True,
        },
        "limitations": [
            "Benchmark uses CPU and must be repeated on the target deployment hardware.",
            (
                "The timings cover model inference only, not sensor acquisition "
                "or FFT feature extraction."
            ),
            "INT8 may be slower for small networks; retain FP32 unless measurements justify it.",
            "The saved INT8 state dict was reloaded into the same quantized architecture.",
        ],
    }
    (output / "cwru_profile_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/cwru/cwru_mlp.pt")
    parser.add_argument("--dataset", default="data/cwru/cwru_features.npz")
    parser.add_argument("--output-dir", default="artifacts/cwru_profile")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    report = profile_experiment(
        args.checkpoint,
        args.dataset,
        args.output_dir,
        runs=args.runs,
        warmup=args.warmup,
        threads=args.threads,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
