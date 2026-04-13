"""Tests for LogisticRegression."""

import numpy as np
import pytest
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


def test_separable_high_accuracy():
    """Linearly separable data should reach >90% accuracy."""
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) + 3, np.random.randn(50, 2) - 3])
    y = np.array([1] * 50 + [0] * 50, dtype=float)
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    assert model.score(X, y) > 0.9


def test_predict_proba_in_range():
    """Probabilities must lie in [0, 1]."""
    np.random.seed(0)
    X, y = np.random.randn(30, 3), np.random.randint(0, 2, 30).astype(float)
    model = LogisticRegression(n_iterations=100).fit(X, y)
    p = model.predict_proba(X)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_predict_binary_labels():
    """Predictions should only contain 0 or 1."""
    X, y = np.random.randn(20, 2), np.random.randint(0, 2, 20).astype(float)
    preds = LogisticRegression(n_iterations=50).fit(X, y).predict(X)
    assert set(preds).issubset({0, 1})


def test_weights_shape():
    X, y = np.random.randn(20, 5), np.random.randint(0, 2, 20).astype(float)
    model = LogisticRegression(n_iterations=50).fit(X, y)
    assert model.weights.shape == (5,)


def test_loss_history_length():
    X, y = np.random.randn(30, 2), np.random.randint(0, 2, 30).astype(float)
    model = LogisticRegression(n_iterations=80).fit(X, y)
    assert len(model.loss_history) == 80


def test_loss_positive():
    """Binary cross-entropy should always be positive."""
    X, y = np.random.randn(30, 2), np.random.randint(0, 2, 30).astype(float)
    model = LogisticRegression(n_iterations=50).fit(X, y)
    assert all(l > 0 for l in model.loss_history)
