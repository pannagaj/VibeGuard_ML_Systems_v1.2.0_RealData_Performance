# Architecture and system design

## Runtime paths

### Training path

1. `generate_dataset` creates labelled sensor windows with a fixed sample rate, motor fundamental frequency, harmonics, class-specific spectral/impulse signatures, phase variation and sensor noise.
2. `extract_features` validates each window and calculates 20 features from time and frequency domains.
3. `load_dataset` validates the CSV, schema and label contract. Sample IDs must be unique.
4. `split_dataset` performs stratified 64/16/20 splitting and checks that IDs do not overlap.
5. `StandardScaler` is fitted only on training features. The same mean/scale are persisted with the checkpoint.
6. The MLP is trained with AdamW and cross-entropy. Every epoch checks finite loss and clips gradients; validation loss controls early stopping and checkpoint selection.
7. ExtraTrees is trained on the same training split as a non-neural baseline.
8. The untouched test split is evaluated once for final comparison artifacts.

### Inference path

- `Predictor` checks the checkpoint schema, validates input matrix shape/finiteness, applies the stored scaler, and returns labels plus class scores.
- `extract_features` is also used for raw windows, so the API and command line do not duplicate feature engineering.
- The FastAPI app loads one checkpoint per configured path, exposes readiness and metadata, and validates each request before inference.

### Optimization path

- FP32 eager is the correctness and performance reference.
- `torch.export` serializes an exported FP32 program with a dynamic batch dimension. The scaler is embedded into this exported module, so its input contract remains raw engineered features.
- Dynamic INT8 is compared on CPU. TorchAO is preferred when present and compatible; otherwise a PyTorch compatibility path is used. The chosen backend is stored in the artifact and must match when reloading.
- Probability drift, top-class agreement, artifact size, batch p50/p95 and throughput are reported. Nothing assumes the smaller/quantized model is necessarily faster.

## Design trade-offs

- A compact MLP is intentionally small enough for CPU and edge-style experiments; the point is to demonstrate system methodology, not maximize model capacity.
- The ExtraTrees comparator is useful for small tabular feature sets and prevents the project from assuming a neural network is automatically the best model.
- Engineered features make the pipeline inspectable and inexpensive, but can omit temporal relationships. A later raw-window 1D CNN or temporal model should be compared against this baseline under the same split protocol.
- Synthetic signal patterns make a deterministic runnable example possible. They do not replace vibration datasets collected from real equipment under representative loads and fault severities.

## Key contracts

| Contract | Source of truth | Failure behavior |
|---|---|---|
| Feature order | `constants.FEATURE_NAMES` | reject wrong width or key set |
| Class order | `constants.CLASS_NAMES` | reject unknown class / artifact mismatch |
| Feature version | `constants.SCHEMA_VERSION` | reject incompatible source/checkpoint |
| Preprocessing | `scaler_mean`, `scaler_scale` in checkpoint | verify dimensions, finiteness and positive scales |
| Raw window | finite 1-D signal, at least 32 samples, positive rate/fundamental | explicit validation error |
| Optimized INT8 model | training checkpoint plus backend-tagged optimized state | reject backend/schema mismatch |
