PYTHON ?= python
DATA ?= data/motor_vibration.csv
ARTIFACTS ?= artifacts
CONFIG ?= configs/default.json

.PHONY: setup data train validate evaluate infer benchmark optimize test lint format-check demo api docker cwru-setup cwru-data cwru-train cwru-profile

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	vibeguard generate-data --output $(DATA) --samples-per-class 1000 --seed 42

train: data
	vibeguard train --data $(DATA) --config $(CONFIG) --output-dir $(ARTIFACTS)

validate:
	vibeguard validate --data $(DATA) --checkpoint $(ARTIFACTS)/model.pt

evaluate:
	vibeguard evaluate --data $(DATA) --checkpoint $(ARTIFACTS)/model.pt

infer:
	vibeguard predict-window --input examples/sample_vibration_window.json --checkpoint $(ARTIFACTS)/model.pt

benchmark:
	vibeguard benchmark --checkpoint $(ARTIFACTS)/model.pt

optimize:
	vibeguard optimize --checkpoint $(ARTIFACTS)/model.pt --data $(DATA) --output-dir $(ARTIFACTS)/optimization

test:
	pytest -q

lint:
	ruff check .

format-check:
	ruff format --check .

demo: train validate evaluate infer optimize

api:
	uvicorn vibeguard.service:app --host 127.0.0.1 --port 8000 --reload

docker:
	docker compose up --build

# Real measured-signal benchmark: official CWRU bearing data (downloaded separately).
cwru-setup:
	$(PYTHON) -m pip install -e ".[dev,realdata]"

cwru-data:
	$(PYTHON) -m experiments.cwru_benchmark.prepare --raw-dir data/cwru/raw --output-dir data/cwru

cwru-train:
	$(PYTHON) -m experiments.cwru_benchmark.train --dataset data/cwru/cwru_features.npz --output-dir artifacts/cwru --seed 42

cwru-profile:
	$(PYTHON) -m experiments.cwru_benchmark.profile --checkpoint artifacts/cwru/cwru_mlp.pt --dataset data/cwru/cwru_features.npz --output-dir artifacts/cwru_profile
