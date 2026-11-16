"""Consistent classification metrics and JSON-safe reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from vibeguard.constants import CLASS_NAMES


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute test metrics from class probabilities, with explicit class ordering."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape (n_samples, number_of_classes).")
    if y_true.ndim != 1 or y_true.size != probabilities.shape[0] or y_true.size == 0:
        raise ValueError("y_true and probabilities must contain the same non-zero sample count.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("y_true must contain integer class indices.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must lie in the closed interval [0, 1].")
    if np.any((y_true < 0) | (y_true >= len(CLASS_NAMES))):
        raise ValueError("y_true contains a class index outside the declared class schema.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Each probability row must sum to one.")
    # Normalize tiny floating-point deviations to avoid sklearn warnings and keep
    # log-loss behavior stable across CPU/GPU probability kernels.
    probabilities = probabilities / row_sums

    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(CLASS_NAMES)))),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
        "per_class": {
            name: {
                key: float(value)
                for key, value in report[name].items()
                if key in {"precision", "recall", "f1-score", "support"}
            }
            for name in CLASS_NAMES
        },
    }
