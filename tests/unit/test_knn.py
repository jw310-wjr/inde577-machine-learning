"""Tests for KNNClassifier and KNNRegressor."""

import numpy as np
import pytest
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# ── Classifier ────────────────────────────────────────────────────────────────

def test_knn_clf_perfect_k1():
    """k=1 should correctly classify its own training points."""
    X = np.array([[0.0, 0], [1, 1], [5, 5], [6, 6]])
    y = np.array([0, 0, 1, 1])
    model = KNNClassifier(k=1).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


def test_knn_clf_separable():
    """Well-separated clusters should be classified perfectly."""
    np.random.seed(0)
    X = np.vstack([np.random.randn(50, 2) + 5, np.random.randn(50, 2) - 5])
    y = np.array([0] * 50 + [1] * 50)
    model = KNNClassifier(k=3).fit(X, y)
    assert model.score(X, y) > 0.95


def test_knn_clf_output_shape():
    X_tr, y_tr = np.random.randn(20, 3), np.random.randint(0, 2, 20)
    X_te = np.random.randn(7, 3)
    preds = KNNClassifier(k=3).fit(X_tr, y_tr).predict(X_te)
    assert preds.shape == (7,)


def test_knn_clf_manhattan():
    """Manhattan distance should still separate simple data."""
    X = np.array([[0.0, 0], [0.1, 0], [10, 10], [10.1, 10]])
    y = np.array([0, 0, 1, 1])
    model = KNNClassifier(k=1, distance_metric="manhattan").fit(X, y)
    assert model.score(X, y) == 1.0


# ── Regressor ─────────────────────────────────────────────────────────────────

def test_knn_reg_k1_exact():
    """k=1 regressor returns the exact training target for each training point."""
    X = np.array([[1.0], [2], [3], [4]])
    y = np.array([10.0, 20, 30, 40])
    preds = KNNRegressor(k=1).fit(X, y).predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-8)


def test_knn_reg_output_shape():
    X_tr, y_tr = np.random.randn(20, 2), np.random.randn(20)
    X_te = np.random.randn(5, 2)
    preds = KNNRegressor(k=3).fit(X_tr, y_tr).predict(X_te)
    assert preds.shape == (5,)


def test_knn_reg_r2_nonnegative():
    """R² on training data with k=1 should be perfect."""
    np.random.seed(1)
    X = np.random.randn(30, 2)
    y = np.random.randn(30)
    model = KNNRegressor(k=1).fit(X, y)
    assert model.score(X, y) > 0.99
