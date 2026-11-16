# Coding-agent hands-on evidence checklist

The repository contains persistent instructions for Cursor, Claude Code and GitHub Copilot.
Configuration files alone do **not** establish hands-on coding-agent proficiency. Complete the
following work yourself in one of those tools before describing that experience on a resume.

## Required practical run

1. Open the repository root in Cursor or Claude Code.
2. Start a new agent session and ask it to read `AGENTS.md` and `docs/CODING_AGENT_EVIDENCE.md`.
3. Ask it to complete one bounded task from the prompts below. Do not accept a large rewrite.
4. Review every changed line. Ask the agent to explain any new API, tensor shape, or data split.
5. Run tests, Ruff, formatting, and the exact experiment commands yourself.
6. Inspect `git diff`, make a commit, and record the actual agent name, task, date, tests and commit hash in `docs/AGENT_RUN_LOG.md`.

## Task A — data correctness

```text
Read AGENTS.md and inspect experiments/cwru_benchmark/data.py plus its tests.
Add a metadata validation step that checks every manifest file has a known class, load,
RPM and official CWRU file number. Add tests for malformed metadata and a duplicate source
file. Do not change the official split protocol. Run the CWRU tests and Ruff; show the diff.
```

## Task B — numerical and model ablation

```text
Read AGENTS.md and inspect the CWRU trainer, model, and evaluation report.
Add a controlled ablation for two MLP widths using the exact existing recording-level split.
Fit preprocessing on train loads only; choose the checkpoint from validation load only; report
final test load metrics once, including per-recording macro-F1. Save configs, seeds, epoch,
parameter count and training duration. Add tests and do not invent results.
```

## Task C — performance optimization

```text
Read AGENTS.md and docs/BENCHMARKING.md. Profile the CWRU MLP with torch.profiler and identify
the top measured CPU operators. Implement one optimization only if the trace suggests a real
bottleneck. Compare before and after on identical held-out inputs with warm-up, p50/p95 latency,
throughput, process memory observation, parameter/artifact size, probability drift and prediction
agreement. Retain FP32 if INT8 or another path is not measurably better. Add reload tests and show
all measured reports and the git diff.
```

## Agent run log — fill only after your own session

Copy the template in `AGENT_RUN_LOG_TEMPLATE.md` to `AGENT_RUN_LOG.md` and enter factual details.
Do not pre-fill or claim an agent run that you did not perform.

## Resume wording after evidence exists

Only after completing and understanding the tasks above, consider wording such as:

> Used [actual coding agent] to implement and refactor Python ML workflows; reviewed generated
> changes, added regression tests, inspected diffs, and independently ran reproducibility and
> performance checks.

Name only the agent(s) you personally used. Be prepared to walk through one agent-generated change.
