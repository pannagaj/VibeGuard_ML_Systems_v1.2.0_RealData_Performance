# Reproducibility guide

## Reproduce the reference workflow

```bash
python -m pip install -e ".[dev]"
vibeguard generate-data --output data/motor_vibration.csv --samples-per-class 1000 --seed 42
vibeguard train --data data/motor_vibration.csv --config configs/default.json --output-dir artifacts
vibeguard validate --data data/motor_vibration.csv --checkpoint artifacts/model.pt --seed 42
vibeguard evaluate --data data/motor_vibration.csv --checkpoint artifacts/model.pt --seed 42
vibeguard optimize --checkpoint artifacts/model.pt --data data/motor_vibration.csv --output-dir artifacts/optimization --runs 300 --warmup 40 --threads 1
```

The bundled `data/motor_vibration.csv` and `artifacts/` are a convenience reference run. Re-run the commands to regenerate them. The feature CSV is synthetic.

## What is deterministic

- Generator uses NumPy `default_rng(seed)` and deterministic row shuffling.
- Train/validation/test split uses a fixed random state and stratification.
- Python, NumPy and PyTorch random seeds are set during training.
- The selected checkpoint is the epoch with the lowest validation loss.
- Class and feature orders are explicit constants.

## What can vary

Different PyTorch, BLAS, scikit-learn, CPU, CUDA, and operating-system versions can change floating-point execution and timing. PyTorch deterministic mode is requested where available, but exact bitwise identity is not guaranteed across all hardware/library versions. Performance timings are inherently host-dependent.

Record package versions and hardware context when presenting benchmark results. The reports include the core host and PyTorch version fields.
