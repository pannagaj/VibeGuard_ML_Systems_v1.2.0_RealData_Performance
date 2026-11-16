"""Safe-ish, versioned model-artifact persistence with schema validation."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vibeguard import __version__
from vibeguard.constants import CLASS_NAMES, FEATURE_NAMES, SCHEMA_VERSION
from vibeguard.model import VibrationMLP


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return output


def save_model_artifact(
    model: VibrationMLP,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    output_path: str | Path,
    seed: int,
    best_epoch: int,
    best_validation_loss: float,
) -> Path:
    metadata = {
        "project": "VibeGuard ML Systems",
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_names": list(CLASS_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "num_classes": len(CLASS_NAMES),
        "hidden_dims": list(model.hidden_dims),
        "dropout": model.dropout_rate,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
    }
    payload = {
        "metadata": metadata,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scaler_mean": np.asarray(scaler_mean, dtype=np.float64).tolist(),
        "scaler_scale": np.asarray(scaler_scale, dtype=np.float64).tolist(),
    }
    return atomic_torch_save(payload, output_path)


def load_model_artifact(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[VibrationMLP, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=True)
    required_payload_keys = {"metadata", "model_state_dict", "scaler_mean", "scaler_scale"}
    if not isinstance(payload, dict) or not required_payload_keys.issubset(payload):
        raise ValueError("Checkpoint does not match the VibeGuard artifact structure.")
    metadata = payload["metadata"]
    required_metadata_keys = {
        "schema_version", "feature_names", "class_names", "input_dim", "num_classes",
        "hidden_dims", "dropout", "seed", "project",
    }
    if not isinstance(metadata, dict) or not required_metadata_keys.issubset(metadata):
        raise ValueError("Checkpoint metadata is missing required fields.")
    if int(metadata["input_dim"]) != len(FEATURE_NAMES):
        raise ValueError("Checkpoint input dimension differs from the feature contract.")
    if int(metadata["num_classes"]) != len(CLASS_NAMES):
        raise ValueError("Checkpoint output dimension differs from the class contract.")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Checkpoint feature schema differs from this source version.")
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Checkpoint class ordering differs from this source version.")

    mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
    if mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape:
        raise ValueError("Checkpoint scaler dimensions do not match the feature schema.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Checkpoint scaler contains invalid values.")

    model = VibrationMLP(
        input_dim=int(metadata["input_dim"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dims=tuple(int(value) for value in metadata["hidden_dims"]),
        dropout=float(metadata["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    metadata = {**metadata, "scaler_mean": mean, "scaler_scale": scale}
    return model, metadata
