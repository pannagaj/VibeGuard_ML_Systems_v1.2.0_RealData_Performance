"""Inference micro-benchmarks with warm-up and percentile latency reporting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vibeguard.artifacts import write_json
from vibeguard.inference import Predictor


def benchmark_inference(
    checkpoint: str | Path,
    *,
    runs: int = 250,
    warmup: int = 40,
    seed: int = 42,
    thread_count: int = 1,
) -> dict[str, Any]:
    if runs < 10 or warmup < 1 or thread_count < 1:
        raise ValueError("runs must be >=10; warmup and thread_count must be positive.")
    torch.set_num_threads(thread_count)
    predictor = Predictor(checkpoint, device="cpu")
    dimension = len(predictor.metadata["feature_names"])
    rng = np.random.default_rng(seed)
    batches = {size: rng.normal(size=(size, dimension)).astype(np.float32) for size in (1, 32, 128)}
    results: list[dict[str, float | int]] = []

    for batch_size, batch in batches.items():
        for _ in range(warmup):
            predictor.predict_proba(batch)
        latencies_ms: list[float] = []
        total_start = time.perf_counter()
        for _ in range(runs):
            start = time.perf_counter()
            predictor.predict_proba(batch)
            latencies_ms.append((time.perf_counter() - start) * 1_000.0)
        elapsed = time.perf_counter() - total_start
        values = np.asarray(latencies_ms, dtype=np.float64)
        results.append(
            {
                "batch_size": batch_size,
                "runs": runs,
                "latency_p50_ms_per_batch": float(np.percentile(values, 50)),
                "latency_p95_ms_per_batch": float(np.percentile(values, 95)),
                "latency_mean_ms_per_batch": float(np.mean(values)),
                "throughput_samples_per_second": float((batch_size * runs) / max(elapsed, 1e-12)),
            }
        )
    return {
        "model": "VibrationMLP",
        "device": "cpu",
        "torch_threads": thread_count,
        "warmup_iterations_per_batch": warmup,
        "measurement_iterations_per_batch": runs,
        "results": results,
        "notes": "Synthetic standardized inputs; includes Python and tensor-conversion overhead.",
    }


def save_benchmark_report(report: dict[str, Any], output_path: str | Path) -> Path:
    return write_json(output_path, report)
