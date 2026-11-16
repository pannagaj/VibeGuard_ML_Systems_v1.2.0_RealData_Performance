"""Reproducible model training, validation-driven checkpointing, and baseline."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vibeguard.artifacts import save_model_artifact, write_json
from vibeguard.constants import CLASS_NAMES, FEATURE_NAMES
from vibeguard.data import load_dataset, split_dataset
from vibeguard.inference import Predictor
from vibeguard.metrics import classification_metrics
from vibeguard.model import VibrationMLP


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch; request deterministic kernels where available."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _evaluate_loss(
    model: VibrationMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss detected.")
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    hidden_dims: tuple[int, int] = (96, 48),
    dropout: float = 0.12,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 2.0,
) -> tuple[VibrationMLP, list[dict[str, float]], int, float]:
    model = VibrationMLP(
        len(FEATURE_NAMES), len(CLASS_NAMES), hidden_dims=hidden_dims, dropout=dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=0,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    waiting = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))

        train_loss = float(np.mean(batch_losses))
        validation_loss = _evaluate_loss(model, val_loader, criterion, device)
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, history, best_epoch, best_loss


def train_project(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    with_baseline: bool = True,
    hidden_dims: tuple[int, int] = (96, 48),
    dropout: float = 0.12,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 2.0,
    baseline_estimators: int = 250,
) -> dict[str, Any]:
    """Train, select on validation loss, and report metrics on untouched test data."""
    if (
        epochs < 1
        or batch_size < 1
        or patience < 1
        or learning_rate <= 0
        or weight_decay < 0
        or gradient_clip_norm <= 0
        or baseline_estimators < 10
        or len(hidden_dims) != 2
        or any(width < 1 for width in hidden_dims)
    ):
        raise ValueError("Training hyperparameters are outside their valid ranges.")
    seed_everything(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(data_path)
    splits = split_dataset(dataset, seed=seed)

    # Fit preprocessing on training data only: validation/test leakage is avoided.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits.train.features).astype(np.float32)
    X_val = scaler.transform(splits.validation.features).astype(np.float32)
    model, history, best_epoch, best_loss = _train_model(
        X_train,
        splits.train.labels,
        X_val,
        splits.validation.labels,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
    )
    checkpoint_path = output / "model.pt"
    save_model_artifact(
        model,
        scaler.mean_,
        scaler.scale_,
        checkpoint_path,
        seed=seed,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
    )

    predictor = Predictor(checkpoint_path, device=str(device))
    torch_metrics = classification_metrics(
        splits.test.labels, predictor.predict_proba(splits.test.features)
    )
    torch_metrics.update(
        {
            "model": "VibrationMLP",
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "device": str(device),
            "test_samples": int(splits.test.labels.size),
        }
    )
    write_json(output / "test_metrics.json", torch_metrics)

    history_path = output / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)

    comparison: dict[str, Any] = {"pytorch_mlp": torch_metrics}
    if with_baseline:
        baseline = ExtraTreesClassifier(
            n_estimators=baseline_estimators,
            max_features=0.9,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        baseline.fit(splits.train.features, splits.train.labels)
        baseline_probabilities = baseline.predict_proba(splits.test.features)
        baseline_metrics = classification_metrics(splits.test.labels, baseline_probabilities)
        baseline_metrics.update(
            {"model": "ExtraTreesClassifier", "test_samples": int(splits.test.labels.size)}
        )
        joblib.dump(baseline, output / "baseline_extratrees.joblib", compress=3)
        comparison["extratrees_baseline"] = baseline_metrics
    write_json(output / "model_comparison.json", comparison)

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output),
        "train_samples": int(splits.train.labels.size),
        "validation_samples": int(splits.validation.labels.size),
        "test_samples": int(splits.test.labels.size),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "device": str(device),
        "test_accuracy": torch_metrics["accuracy"],
        "test_macro_f1": torch_metrics["macro_f1"],
        "baseline_included": with_baseline,
        "model_hidden_dims": list(hidden_dims),
        "model_dropout": float(dropout),
        "baseline_estimators": baseline_estimators if with_baseline else None,
    }
    write_json(output / "training_summary.json", summary)
    return {"summary": summary, "metrics": comparison}
