"""Train and evaluate MLP / ExtraTrees on CWRU with a held-out motor-load protocol."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cwru_benchmark.data import (
    CLASS_NAMES,
    load_prepared_dataset,
    split_for_load,
)
from vibeguard.constants import FEATURE_NAMES
from vibeguard.model import VibrationMLP


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, predicted, labels=np.arange(len(CLASS_NAMES)),
            target_names=list(CLASS_NAMES), output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
    }


def train_experiment(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 12,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters.")
    _seed(seed)
    dataset = load_prepared_dataset(dataset_path)
    split_masks = {name: dataset["splits"] == name for name in ("train", "validation", "test")}
    for split_name, mask in split_masks.items():
        if not np.any(mask):
            raise ValueError(f"The {split_name} split is empty.")
        missing_classes = set(range(len(CLASS_NAMES))) - set(np.unique(dataset["labels"][mask]))
        if missing_classes:
            raise ValueError(
                f"The {split_name} split is missing classes: "
                f"{[CLASS_NAMES[index] for index in sorted(missing_classes)]}"
            )

    # Guard against recording-level leakage, not merely row-level overlap.
    recording_splits: dict[str, set[str]] = {}
    for recording_id, split_name in zip(dataset["recording_ids"], dataset["splits"], strict=True):
        recording_splits.setdefault(str(recording_id), set()).add(str(split_name))
    leaked = {
        recording_id: names
        for recording_id, names in recording_splits.items()
        if len(names) > 1
    }
    if leaked:
        raise ValueError(f"Recording leakage across splits detected: {leaked}")

    recording_metadata: dict[str, tuple[int, int, str]] = {}
    for recording_id, label, load_hp, split_name in zip(
        dataset["recording_ids"],
        dataset["labels"],
        dataset["load_hp"],
        dataset["splits"],
        strict=True,
    ):
        identity = (int(label), int(load_hp), str(split_name))
        previous = recording_metadata.setdefault(str(recording_id), identity)
        if previous != identity:
            raise ValueError(
                f"Recording {recording_id} has inconsistent label/load/split metadata."
            )
        if split_for_load(int(load_hp)) != str(split_name):
            raise ValueError(
                f"Recording {recording_id} is assigned to {split_name} but load {load_hp} "
                f"belongs to {split_for_load(int(load_hp))}."
            )

    X = dataset["features"]
    y = dataset["labels"]
    X_train, y_train = X[split_masks["train"]], y[split_masks["train"]]
    X_val, y_val = X[split_masks["validation"]], y[split_masks["validation"]]
    X_test, y_test = X[split_masks["test"]], y[split_masks["test"]]

    # Preprocessing is fitted on training loads only.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name)
    model = VibrationMLP(
        len(FEATURE_NAMES),
        len(CLASS_NAMES),
        hidden_dims=(96, 48),
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_tensor = torch.from_numpy(X_val_s).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(val_tensor), y_val_tensor).cpu())
        train_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    training_seconds = time.perf_counter() - started

    with torch.inference_mode():
        torch_test_logits = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()
    torch_test_pred = np.argmax(torch_test_logits, axis=1)
    with torch.inference_mode():
        torch_val_pred = np.argmax(
            model(torch.from_numpy(X_val_s).to(device)).cpu().numpy(), axis=1
        )
    torch_val_metrics = _metrics(y_val, torch_val_pred)
    torch_test_metrics = _metrics(y_test, torch_test_pred)

    # Same split, same train-only preprocessing, independent classical baseline.
    baseline = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    baseline.fit(X_train_s, y_train)
    baseline_val = _metrics(y_val, baseline.predict(X_val_s))
    baseline_test = _metrics(y_test, baseline.predict(X_test_s))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "cwru_mlp.pt"
    metadata = {
        "project": "VibeGuard CWRU measured-signal experiment",
        "dataset": "CWRU 12 kHz drive-end, 0.007-inch seeded faults",
        "class_names": list(CLASS_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "hidden_dims": [96, 48],
        "dropout": 0.1,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "split_protocol": {
            "train_load_hp": [0, 1],
            "validation_load_hp": [2],
            "test_load_hp": [3],
        },
        "sample_rate_hz": 12_000,
        "window_size": 4_096,
    }
    torch.save(
        {
            "metadata": metadata,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
            "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        },
        checkpoint,
    )
    joblib.dump(baseline, output / "cwru_extratrees.joblib")
    with (output / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)

    # Report per-recording test results so a strong average cannot hide one bad file.
    per_recording: dict[str, Any] = {}
    test_recordings = dataset["recording_ids"][split_masks["test"]]
    for recording_id in sorted(set(test_recordings.tolist())):
        mask = test_recordings == recording_id
        per_recording[str(recording_id)] = _metrics(y_test[mask], torch_test_pred[mask])

    report = {
        "experiment": "CWRU cross-load bearing-fault classification",
        "seed": seed,
        "device": str(device),
        "rows": {name: int(mask.sum()) for name, mask in split_masks.items()},
        "recordings_per_split": {
            name: sorted(set(dataset["recording_ids"][mask].tolist()))
            for name, mask in split_masks.items()
        },
        "training": {
            "seconds": training_seconds,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "train_rows": int(y_train.size),
            "validation_rows": int(y_val.size),
            "test_rows": int(y_test.size),
        },
        "pytorch_mlp": {"validation": torch_val_metrics, "test": torch_test_metrics},
        "extra_trees": {"validation": baseline_val, "test": baseline_test},
        "pytorch_test_metrics_by_recording": per_recording,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "baseline": str(output / "cwru_extratrees.joblib"),
        },
        "limitations": [
            (
                "This is measured laboratory data from a single test-rig programme, "
                "not factory field data."
            ),
            (
                "The final test split holds out the 3 HP load but not an "
                "independently sourced machine."
            ),
            "Only 0.007-inch seeded faults and the documented 12 kHz drive-end subset are used.",
            "Do not generalize performance to natural faults or unseen sensor installations.",
        ],
    }
    report_path = output / "cwru_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/cwru/cwru_features.npz")
    parser.add_argument("--output-dir", default="artifacts/cwru")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = train_experiment(
        args.dataset,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
