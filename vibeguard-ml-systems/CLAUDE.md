# Claude Code project entry point

Read [`AGENTS.md`](AGENTS.md) first. It is the shared source of project rules, ML leakage constraints, artifact contracts, and required verification commands.

For every change:
1. Inspect the relevant implementation, call sites, and tests before editing.
2. Preserve the ordered feature schema and versioned artifact contract.
3. Add or update automated tests and documentation.
4. Run `pytest -q`, `ruff check .`, `ruff format --check .`, and `python -m compileall -q src tests`.
5. Report only commands that actually passed; do not invent model metrics or performance gains.
