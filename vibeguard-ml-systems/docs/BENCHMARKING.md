# Benchmarking and performance optimization protocol

## Questions measured

1. What is eager FP32 inference latency at batch sizes 1, 32 and 128?
2. Does the `torch.export` representation change latency or throughput?
3. What latency, model-size and probability-drift trade-off is observed for dynamic INT8 on this host?
4. Do optimized models preserve top-class predictions and close numerical outputs?

## Method

- Run on CPU with a configurable PyTorch thread count.
- Use a fixed RNG seed to generate deterministic representative feature values from the training scaler's feature-wise means and scales.
- Warm up each batch shape before collecting measurements.
- Record p50, p95 and mean batch latency and samples/second.
- Compare probabilities to eager FP32 on the same 32-row parity batch.
- When a dataset is supplied, evaluate each representation on the same untouched test split (using the split seed stored in the training checkpoint); report accuracy, macro-F1, log loss, probability drift and top-class agreement.
- Record Python, PyTorch, host platform, processor string, logical CPU count, batch sizes, warmup and measured iteration counts.
- Save exported artifacts and reload the INT8 artifact using its tagged backend; check that persisted outputs equal the in-memory quantized output within a strict tolerance.

## Run

```bash
vibeguard optimize --checkpoint artifacts/model.pt --data data/motor_vibration.csv --output-dir artifacts/optimization --runs 300 --warmup 40 --threads 1
```

Use at least several hundred measured runs for presentation-quality results and repeat on the deployment host. Close other CPU-heavy applications and document any power/thermal constraints.

## Interpretation rules

- p50 summarizes typical latency; p95 exposes tail behavior.
- Throughput depends on batch size and batching overhead; do not compare a batch's latency to a per-sample latency without converting units carefully.
- Small MLPs may be too small for export or quantization overhead to pay off. It is acceptable—and valuable—to report that a candidate is slower.
- Dynamic quantization is not assumed to reduce memory or latency by a fixed percentage. Check actual artifact size and measurements.
- Probability drift alone is insufficient: inspect top-class agreement and re-evaluate class metrics on the held-out dataset before promoting a quantized model.
- Benchmark numbers are tied to the recorded environment and must not be generalized to other hardware.

## Future extensions

- Add warm-up and latency tests for the FastAPI request path, not just model forward passes.
- Evaluate CPU thread scaling and pinned-memory/transfer overhead on CUDA.
- Benchmark `torch.compile` where compilation time can be amortized over a known serving volume.
- Add memory/RSS and energy measurements on target edge hardware.
- Add real-data input windows, data-loader throughput and end-to-end preprocessing cost.
