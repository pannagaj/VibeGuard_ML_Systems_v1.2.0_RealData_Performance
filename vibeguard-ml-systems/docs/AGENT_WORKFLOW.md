# Coding-agent workflow

VibeGuard includes shared instructions for common coding agents:

- `AGENTS.md` — canonical repository-wide rules and verification checklist.
- `CLAUDE.md` — Claude Code entry point.
- `.cursor/rules/ml-engineering.mdc` — Cursor project rule.
- `.github/copilot-instructions.md` — GitHub Copilot guidance.

The files are instructions for an agent working inside this repository; their presence does not claim that a particular agent has already been used.

## Recommended workflow

1. Open the repository in Cursor, Claude Code, or another coding agent with Python and terminal access.
2. Ask the agent to read `AGENTS.md` before editing.
3. Give it one bounded task (for example, add an ablation for hidden-layer width or add a raw-data adapter).
4. Require it to inspect callers and tests before changing implementation.
5. Require a test for the new behavior and for invalid/boundary inputs.
6. Ask it to run the required checks, show the diff, and summarize only verified outcomes.

## Example prompt: model ablation

```text
Read AGENTS.md and inspect the current trainer, model, config, and tests.
Add a controlled ablation comparing two MLP hidden-dimension configurations.
Use the same deterministic train/validation/test split and training-only scaler.
Select models using validation data only; keep the test set untouched until final
reporting. Save per-run config, seed, best epoch, test macro-F1, accuracy, and
parameter count. Add tests and update the experiment documentation. Do not invent
performance claims. Run the full required checks and show the changed files.
```

## Example prompt: inference optimization

```text
Read AGENTS.md and docs/BENCHMARKING.md. Inspect the current optimize command.
Propose one additional CPU inference optimization and implement it behind a clear
CLI option. Preserve the raw-feature input and scaler contract. Compare against eager
FP32 using warm-up, p50/p95, throughput, artifact size, probability drift and
prediction agreement. Add a reload test for the saved artifact and update docs.
Do not claim a speed-up unless the measured report demonstrates it.
```

## Example prompt: real-sensor ingestion

```text
Read AGENTS.md and docs/ARCHITECTURE.md. Add a validated CSV adapter for windows
of real vibration samples with explicit sampling-rate and label metadata. Reject
unknown schema versions, missing samples, non-finite values, and duplicate IDs.
Never change the feature names/order or artifact schema silently. Add fixture-based
tests and explain data provenance and the real/synthetic distinction in docs.
```

## Good evaluation behavior

An engineering coding agent should be asked to explain design trade-offs, edge cases, and test evidence—not merely produce code. Review generated code and dependencies yourself. Do not paste private datasets, credentials, proprietary source, or sensitive information into third-party coding agents unless you are authorized to do so.
