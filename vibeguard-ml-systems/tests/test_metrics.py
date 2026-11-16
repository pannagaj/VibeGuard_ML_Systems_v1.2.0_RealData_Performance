import numpy as np
import pytest

from vibeguard.constants import CLASS_NAMES
from vibeguard.metrics import classification_metrics


def test_metrics_report_macro_scores_and_confusion_matrix() -> None:
    y_true = np.arange(len(CLASS_NAMES)).repeat(2)
    probabilities = np.full((y_true.size, len(CLASS_NAMES)), 0.01)
    probabilities[np.arange(y_true.size), y_true] = 0.97
    result = classification_metrics(y_true, probabilities)

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert len(result["confusion_matrix"]) == len(CLASS_NAMES)


def test_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(np.array([0, 1]), np.array([[0.2, 0.2, 0.2, 0.2], [1, 0, 0, 0]]))


def test_metrics_normalize_small_probability_rounding_error_without_warning() -> None:
    y_true = np.arange(len(CLASS_NAMES))
    probabilities = np.full((len(CLASS_NAMES), len(CLASS_NAMES)), 0.001)
    probabilities[np.arange(len(CLASS_NAMES)), y_true] = 0.9979999
    result = classification_metrics(y_true, probabilities)
    assert result["accuracy"] == 1.0
