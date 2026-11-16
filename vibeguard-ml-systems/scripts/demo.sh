#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"
vibeguard generate-data --output data/motor_vibration.csv --samples-per-class 1000 --seed 42
vibeguard train --data data/motor_vibration.csv --config configs/default.json --output-dir artifacts
vibeguard validate --data data/motor_vibration.csv --checkpoint artifacts/model.pt
vibeguard evaluate --data data/motor_vibration.csv --checkpoint artifacts/model.pt
vibeguard predict-window --input examples/sample_vibration_window.json --checkpoint artifacts/model.pt
vibeguard optimize --checkpoint artifacts/model.pt --data data/motor_vibration.csv --output-dir artifacts/optimization --runs 300 --warmup 40 --threads 1
pytest -q
