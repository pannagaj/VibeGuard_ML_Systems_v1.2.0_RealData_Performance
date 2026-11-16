# VibeGuard contributor and coding-agent instructions

These instructions apply to human contributors and coding agents. Read this file before changing model, data, inference, or benchmark code.

## Goal

Maintain a reproducible, inspectable ML engineering project for motor-vibration fault classification. Prefer correct, testable implementations over impressive-looking claims. The included dataset is synthetic and must always be described as synthetic.

## Architecture contract

- `constants.py` is the source of truth for the ordered 20-feature schema, class order, sample rate, window length, and artifact schema version.
- `features.py` converts one finite raw waveform into exactly `FEATURE_NAMES`, in the same order.
- `data.py` owns the deterministic simulator, CSV validation, and stratified data split. Keep unique sample IDs and reject incompatible schema versions.
- `training.py` owns seeds, train-only preprocessing, validation-driven model selection, gradient clipping, finite-loss checks, model comparisons, and metrics artifacts.
- `artifacts.py` owns checkpoint serialization, schema validation, and safe state-dict loading.
- `inference.py` and `optimization.py` must apply the exact scaler fitted during training. Exported variants must accept the same raw feature vectors as `Predictor`.
- `service.py` exposes validated prediction APIs; do not add hidden execution, shell access, or unsafe deserialization.
- `cli.py` is the user-facing command-line interface. Add a parser command and a test for every new user-facing workflow.
- `experiments/cwru_benchmark/` is the measured-lab-data extension. Keep its recording-level motor-load split fixed unless a new protocol is explicitly versioned; do not mix windows from one source recording across splits.

## Non-negotiable ML rules

1. Never fit a scaler, selector, calibrator, or other data-derived preprocessing on validation or test data.
2. The test set is a final report only. Do not use it to select epochs, models, thresholds, or hyperparameters.
3. Keep class and feature ordering explicit. Never depend on incidental dictionary, CSV, or estimator ordering.
4. Validate array shapes, class ranges, finiteness, and probability sums at system boundaries.
5. Report actual measured metrics. Never hardcode accuracy, speed-up, memory reduction, or latency in source or docs.
6. Performance claims require reproducible measurements, warm-up, batch size, host/library context, and parity/error checks.
7. Preserve a working CPU path. CUDA must remain optional and must fail clearly if requested but unavailable.
8. Do not present synthetic performance as real-world industrial generalization.
9. For CWRU, report the exact loads, 0.007-inch seeded-fault subset, windowing, per-recording metrics and single-rig limitations. Do not call CWRU field-factory data.
10. Benchmark INT8/export variants against eager FP32 on the same host, preserve numerical parity reporting, and do not claim a speed-up unless measurements prove it.

## Python and implementation style

- Support Python 3.11+ and use type hints for public functions.
- Keep modules focused; avoid giant CLI or training functions when a tested helper is clearer.
- Prefer deterministic behavior and explicit parameters over global hidden configuration.
- Use `torch.inference_mode()` for inference and avoid keeping computation graphs in metric/benchmark code.
- Save PyTorch artifacts as state dictionaries plus primitive metadata. Load with `weights_only=True`; never load untrusted pickle/joblib files.
- Write files atomically when partial files could be mistaken for valid artifacts.
- Return clear `ValueError`/`RuntimeError` messages for invalid user inputs and unavailable backends.
- Do not add dependencies unless the feature requires them; explain optional dependencies in README.

## Required checks before finishing a change

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests experiments
```

For changes to the full workflow, also run:

```bash
vibeguard generate-data --output /tmp/vibeguard-data.csv --samples-per-class 40 --seed 42
vibeguard train --data /tmp/vibeguard-data.csv --config configs/default.json --epochs 3 --output-dir /tmp/vibeguard-artifacts --no-baseline
vibeguard validate --data /tmp/vibeguard-data.csv --checkpoint /tmp/vibeguard-artifacts/model.pt
vibeguard optimize --checkpoint /tmp/vibeguard-artifacts/model.pt --output-dir /tmp/vibeguard-optimization --runs 10 --warmup 1
```

Run API tests when changing request/response behavior. Update docs and tests in the same change. Never claim a check passed unless it was executed.

## Coding-agent task checklist

Before editing: inspect the module, its callers, related tests, feature/artifact contracts, and docs. State the intended change briefly. Make the smallest coherent patch. Add tests for success, invalid inputs, and boundary cases. Run checks, inspect `git diff`, and summarize changed behavior plus commands/results. Ask the human before changing the feature schema or artifact schema version.

For the measured CWRU extension (after downloading the official files):

```bash
python -m experiments.cwru_benchmark.prepare --raw-dir data/cwru/raw --output-dir data/cwru
python -m experiments.cwru_benchmark.train --dataset data/cwru/cwru_features.npz --output-dir artifacts/cwru
python -m experiments.cwru_benchmark.profile --checkpoint artifacts/cwru/cwru_mlp.pt --dataset data/cwru/cwru_features.npz --output-dir artifacts/cwru_profile
```
