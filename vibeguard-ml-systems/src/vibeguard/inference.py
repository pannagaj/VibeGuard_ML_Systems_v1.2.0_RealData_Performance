"""Validated batch and single-record prediction interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from vibeguard.artifacts import load_model_artifact
from vibeguard.constants import CLASS_NAMES, FEATURE_NAMES


class Predictor:
    """Loaded model plus its training-only standardization parameters."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.metadata = load_model_artifact(checkpoint, device=self.device)
        self.mean = self.metadata["scaler_mean"]
        self.scale = self.metadata["scaler_scale"]

    def predict_proba(self, features: np.ndarray | list[list[float]]) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Input must have shape (n_samples, {len(FEATURE_NAMES)}).")
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("Input must contain at least one row of finite feature values.")
        standardized = (values - self.mean) / self.scale
        if not np.isfinite(standardized).all():
            raise FloatingPointError("Standardization produced non-finite values.")
        tensor = torch.from_numpy(np.ascontiguousarray(standardized)).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        result = probabilities.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError("Model inference produced non-finite probabilities.")
        return result

    def predict(self, features: np.ndarray | list[list[float]]) -> list[dict[str, Any]]:
        probabilities = self.predict_proba(features)
        results: list[dict[str, Any]] = []
        for row in probabilities:
            predicted_index = int(np.argmax(row))
            results.append(
                {
                    "predicted_label": CLASS_NAMES[predicted_index],
                    "confidence": float(row[predicted_index]),
                    "probabilities": {
                        class_name: float(row[index])
                        for index, class_name in enumerate(CLASS_NAMES)
                    },
                }
            )
        return results
