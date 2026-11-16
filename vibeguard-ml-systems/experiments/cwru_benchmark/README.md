# CWRU measured-signal experiment

This experiment extends VibeGuard beyond its bundled synthetic simulator using measured
laboratory vibration recordings from the **Case Western Reserve University (CWRU) Bearing
Data Center**. Original MATLAB files are downloaded from the university site at run time;
they are not checked into or redistributed by this repository.

Official source: <https://engineering.case.edu/bearingdatacenter/download-data-file>

## Dataset scope

- 12 kHz drive-end accelerometer measurements.
- Healthy baseline recordings and 0.007-inch seeded inner-race, ball, and outer-race faults.
- Four motor loads (0, 1, 2, 3 HP), with RPM metadata taken from the CWRU data tables.
- 4,096-sample windows with 2,048-sample step (50% overlap).
- The existing ordered 20-feature contract is reused.

This is **measured lab-rig data**, not a field deployment dataset. It comes from seeded faults
on a documented motor test rig, so it does not establish performance on other machines,
naturally developing faults, sensors, mountings, or factory environments.

## Leakage-aware protocol

| Split | Load condition | Use |
|---|---|---|
| Train | 0 and 1 HP | Fit scaler and model weights |
| Validation | 2 HP | Early stopping and model selection |
| Final test | 3 HP | One final cross-load report |

All windows from a source recording are assigned by motor load and remain in one split.
Overlapping windows are therefore never split randomly across train and test. The scaler is fit
on training rows only. The test split is not used for model selection.

This evaluates cross-load transfer on the CWRU rig, **not cross-machine generalization**. A
strong result must be reported with the per-recording metrics and the full confusion matrix.

## Install and run

From the repository root:

```bash
python -m pip install -e '.[dev,realdata]'
python -m experiments.cwru_benchmark.prepare \
  --raw-dir data/cwru/raw --output-dir data/cwru
python -m experiments.cwru_benchmark.train \
  --dataset data/cwru/cwru_features.npz --output-dir artifacts/cwru \
  --seed 42 --epochs 80 --device cpu
python -m experiments.cwru_benchmark.profile \
  --checkpoint artifacts/cwru/cwru_mlp.pt \
  --dataset data/cwru/cwru_features.npz \
  --output-dir artifacts/cwru_profile --runs 300 --warmup 40 --threads 1
```

The downloader accesses the official CWRU host. If you download the files manually, put the
16 numbered `.mat` files in `data/cwru/raw/` and call prepare with `--no-download`.

## Main outputs

- `data/cwru/dataset_manifest.json`: provenance, recording IDs, sample counts and split counts.
- `data/cwru/cwru_features.npz`: feature matrix, labels, recording IDs, load values and split tags.
- `artifacts/cwru/cwru_evaluation.json`: PyTorch MLP vs ExtraTrees, validation and held-out test.
- `artifacts/cwru/cwru_mlp.pt`: safe state-dict and primitive metadata checkpoint.
- `artifacts/cwru_profile/cwru_profile_report.json`: p50/p95, throughput, state size, probability
  drift, test-set prediction agreement and coarse process RSS observation.
- `artifacts/cwru_profile/torch_profiler_trace.json`: CPU trace for Chrome trace viewer.
- `artifacts/cwru_profile/torch_profiler_summary.txt`: top CPU operators and memory events.

## Interpret performance honestly

The profile compares eager FP32 with dynamic INT8. It reports the actual result on the current
host; no speed-up is assumed. A lower-precision model is only useful if the target hardware
shows an improvement at an acceptable numerical and prediction-parity level. The profiler's
process RSS change is a coarse whole-process observation, not isolated model memory.

## Attribution

The CWRU Bearing Data Center describes the test rig, fault conditions, sample rates, and the
meaning of drive-end/fan-end/base channels. Cite the official data center in reports and research
notes. This software only downloads, segments, and analyzes the original source recordings.
