# Changelog

## 1.1.0

- Added raw waveform prediction through the CLI and FastAPI.
- Added an exported FP32 inference artifact with dynamic batch support.
- Added dynamic INT8 experiment, artifact reload validation, numerical drift and throughput reporting.
- Made training hyperparameters configurable through JSON.
- Added coding-agent instructions for AGENTS.md, Claude Code, Cursor and GitHub Copilot.
- Expanded tests, API documentation, model card, benchmarking protocol and reproducibility notes.

## 1.0.0

- Initial reproducible vibration feature generation, PyTorch training, ExtraTrees baseline, metrics, CLI, versioned checkpoint and FastAPI batch inference.


## 1.2.0 — measured-data and performance engineering extension

- Added optional CWRU 12 kHz drive-end data download and feature preparation.
- Added recording-grouped cross-load train/validation/test protocol (0/1 HP train, 2 HP validation, 3 HP test).
- Added PyTorch MLP and ExtraTrees comparison with validation-only selection and per-recording test metrics.
- Added CPU p50/p95, throughput, dynamic INT8 parity, process RSS observation, and torch.profiler trace generation.
- Added bounded coding-agent tasks and a manual run-log template; no personal agent-use claim is implied.
- No measured CWRU metrics are bundled because original data download could not be performed in the packaging environment.
