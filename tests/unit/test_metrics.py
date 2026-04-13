"""Tests for evaluation metrics."""

import numpy as np
import pytest
from rice_ml.processing.metrics import (
    accuracy_score, mean_squared_error, root_mean_squared_error,
    mean_absolute_error, r2_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report,
)


def test_accuracy_perfect():
    assert accuracy_score([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_accuracy_zero():
    assert accuracy_score([0, 0], [1, 1]) == 0.0


def test_mse_perfect():
    assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_mse_value():
    y_t, y_p = np.array([0.0, 0, 0]), np.array([1.0, 1, 1])
    assert np.isclose(mean_squared_error(y_t, y_p), 1.0)


def test_rmse_nonnegative():
    y_t, y_p = np.random.randn(20), np.random.randn(20)
    assert root_mean_squared_error(y_t, y_p) >= 0


def test_mae_value():
    assert np.isclose(mean_absolute_error([0, 0, 0], [1, 2, 3]), 2.0)


def test_r2_perfect():
    y = np.array([1.0, 2, 3, 4])
    assert np.isclose(r2_score(y, y), 1.0)


def test_r2_zero():
    y_t = np.array([1.0, 2, 3])
    y_p = np.full(3, np.mean(y_t))
    assert np.isclose(r2_score(y_t, y_p), 0.0, atol=1e-10)


def test_confusion_matrix_shape():
    y_t = [0, 1, 2, 0, 1, 2]
    y_p = [0, 1, 2, 0, 2, 1]
    cm = confusion_matrix(y_t, y_p)
    assert cm.shape == (3, 3)


def test_confusion_matrix_diagonal():
    y = [0, 1, 0, 1]
    cm = confusion_matrix(y, y)
    assert cm[0, 0] == 2 and cm[1, 1] == 2


def test_precision_perfect():
    assert np.isclose(precision_score([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)


def test_recall_perfect():
    assert np.isclose(recall_score([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)


def test_f1_perfect():
    assert np.isclose(f1_score([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)


def test_classification_report_runs():
    y_t = [0, 1, 0, 1, 0]
    y_p = [0, 1, 1, 1, 0]
    report = classification_report(y_t, y_p)
    assert "0" in report and "1" in report and "accuracy" in report
