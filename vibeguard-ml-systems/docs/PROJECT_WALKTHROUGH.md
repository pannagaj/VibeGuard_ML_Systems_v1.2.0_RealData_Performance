# Project walkthrough and interview preparation

This guide explains how to present VibeGuard accurately and how to verify that you
understand the implementation before discussing it in an interview.

## 60-second project explanation

> VibeGuard is a reproducible machine-learning systems project for classifying four
> motor conditions from vibration windows. The repository includes a deterministic
> signal simulator and a versioned 20-feature contract, plus a PyTorch MLP compared
> with an ExtraTrees baseline using stratified train/validation/test splits. The scaler is
> fitted on training data only, checkpoint selection uses validation loss, and the
> held-out test set is reserved for final evaluation. I added batch and raw-waveform
> inference, a FastAPI service, artifact validation, automated tests, and a benchmark
> comparing eager FP32, `torch.export`, and dynamic INT8. The benchmark is measurement-
> driven: on the included small model, eager FP32 was faster on this host, while INT8
> reduced artifact size. The bundled dataset is synthetic, so the metrics demonstrate
> pipeline correctness rather than industrial fault-detection performance.

Use this as an architecture walkthrough, not an automatic claim of personal authorship.
Before presenting it as your own project work, read the implementation, run it yourself,
make and test meaningful changes, and be transparent about AI assistance. Do not claim
production deployment, real-sensor validation, or prior professional ML experience based
only on this repository.

## Be ready to explain these design decisions

1. **Why synthetic data?** It makes the complete workflow runnable without private sensor
   data. It is a controlled engineering fixture, not a substitute for validation on real,
   correctly labelled machines and operating conditions.
2. **Why fixed feature names and order?** Training, inference, API clients and optimized
   artifacts need the same numerical input contract. A changed feature definition should
   have a schema/version change, not a silent reorder.
3. **Why fit the scaler on the train split only?** Fitting preprocessing on validation or
   test data leaks information from evaluation data into model development.
4. **Why use validation loss for checkpoint selection?** It supports model selection
   without using the final held-out test set to choose an epoch.
5. **Why compare with ExtraTrees?** A classical baseline helps determine whether a neural
   network adds value for this tabular feature representation.
6. **Why report p50 and p95?** p50 describes typical latency; p95 shows tail behavior.
   Include batch size, warm-up, run count, device, thread count and software versions.
7. **Why can INT8 be slower?** For a tiny MLP, quantization and dispatch overhead can be
   larger than the compute saved. Measure on the intended deployment machine.
8. **What does probability drift mean?** Quantized outputs need not be numerically
   identical. Check probability deltas, top-class agreement, and held-out metrics together.

## Hands-on verification checklist

Before describing this as a personal portfolio project, run each command yourself and
inspect the corresponding source and output:

```bash
python -m pip install -e ".[dev]"
pytest -q
vibeguard validate --data data/motor_vibration.csv --checkpoint artifacts/model.pt
vibeguard evaluate --data data/motor_vibration.csv --checkpoint artifacts/model.pt
vibeguard predict-window --input examples/sample_vibration_window.json
vibeguard optimize --data data/motor_vibration.csv --runs 300 --warmup 40 --threads 1
```

Then make at least one meaningful change yourself, such as adding a new feature, changing
an MLP configuration, or measuring a different batch-size range. Add tests, rerun the
workflow, and inspect the diff and generated report.

## Coding-agent exercise

Open the repository in Cursor, Claude Code, or GitHub Copilot Chat. Ask the agent to read
`AGENTS.md` and `docs/AGENT_WORKFLOW.md` before making a bounded change. For example:

```text
Read AGENTS.md, docs/ARCHITECTURE.md and docs/BENCHMARKING.md. Add an ablation that
compares two hidden-layer configurations using the same deterministic split and
train-only preprocessing. Select the checkpoint using validation data only. Preserve
the held-out test set for final evaluation. Add tests, write a reproducible report,
run the required checks, and summarize the exact files changed. Do not invent a
performance improvement if the measurements do not show one.
```

Treat generated code as a proposal: review the diff, run the tests, and verify the
agent's claims. Mention coding-agent use in an application only after you have actually
used an agent in this repository and can explain how you checked its output.
