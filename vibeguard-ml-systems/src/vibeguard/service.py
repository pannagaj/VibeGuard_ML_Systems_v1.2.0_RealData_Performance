"""FastAPI inference service; set VIBEGUARD_CHECKPOINT to choose the model artifact."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from vibeguard.constants import FEATURE_NAMES
from vibeguard.features import extract_features
from vibeguard.inference import Predictor


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[dict[str, float]] = Field(min_length=1, max_length=256)


class WindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal: list[float] = Field(min_length=32, max_length=100_000)
    fundamental_hz: float = Field(gt=0)
    sample_rate_hz: int = Field(default=1_024, gt=0)


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]


@lru_cache(maxsize=4)
def _get_predictor(checkpoint: str) -> Predictor:
    return Predictor(checkpoint, device="cpu")


def create_app(checkpoint_path: str | Path | None = None) -> FastAPI:
    """Create an app with an explicit checkpoint (useful for tests and deployments)."""
    selected_path = str(checkpoint_path or os.getenv("VIBEGUARD_CHECKPOINT", "artifacts/model.pt"))
    app = FastAPI(
        title="VibeGuard Inference API",
        version="1.1.0",
        description=(
            "Predict motor operating condition from a versioned vibration feature vector "
            "or a raw time-domain sensor window."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        checkpoint = Path(selected_path)
        if not checkpoint.is_file():
            return {"status": "model_missing", "model_path": selected_path}
        try:
            _get_predictor(selected_path)
        except (FileNotFoundError, ValueError, RuntimeError):
            return {"status": "model_invalid", "model_path": selected_path}
        return {"status": "ready", "model_path": selected_path}

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        try:
            predictor = _get_predictor(selected_path)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "project": predictor.metadata["project"],
            "schema_version": predictor.metadata["schema_version"],
            "feature_names": FEATURE_NAMES,
            "class_names": predictor.metadata["class_names"],
            "device": "cpu",
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        rows: list[list[float]] = []
        for index, sample in enumerate(request.samples):
            missing = set(FEATURE_NAMES) - set(sample)
            extra = set(sample) - set(FEATURE_NAMES)
            if missing or extra:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "sample_index": index,
                        "missing_features": sorted(missing),
                        "unknown_features": sorted(extra),
                    },
                )
            rows.append([sample[name] for name in FEATURE_NAMES])
        try:
            matrix = np.asarray(rows, dtype=np.float32)
            if not np.isfinite(matrix).all():
                raise HTTPException(
                    status_code=422, detail="Feature samples must contain only finite numbers."
                )
            predictions = _get_predictor(selected_path).predict(matrix)
        except (FileNotFoundError, ValueError, FloatingPointError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return PredictResponse(predictions=predictions)

    @app.post("/predict-window")
    def predict_window(request: WindowRequest) -> dict[str, Any]:
        try:
            signal = np.asarray(request.signal, dtype=np.float64)
            features = extract_features(
                signal,
                fundamental_hz=request.fundamental_hz,
                sample_rate_hz=request.sample_rate_hz,
            )
            prediction = _get_predictor(selected_path).predict(
                np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
            )[0]
        except (FileNotFoundError, ValueError, FloatingPointError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"features": features, "prediction": prediction}

    return app


app = create_app()
