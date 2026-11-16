# GitHub Copilot instructions

Follow [`AGENTS.md`](../AGENTS.md) for architecture and contribution rules. In particular:

- Preserve the ordered feature and class schemas.
- Do not leak validation/test information into preprocessing or model selection.
- Validate shapes, finiteness, class ranges and serialized artifact versions.
- Keep CPU inference as the baseline; CUDA is optional.
- Report benchmark measurements, host context and numerical drift honestly.
- Keep synthetic-data limitations explicit.
- Add tests and update documentation for behavior changes.
