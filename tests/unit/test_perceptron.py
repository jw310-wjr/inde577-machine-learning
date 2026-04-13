"""Tests for Perceptron."""

import numpy as np
import pytest
from rice_ml.supervised_learning.perceptron import Perceptron


def test_linearly_separable():
    """Perceptron should converge on linearly separable data."""
    np.random.seed(0)
    X = np.vstack([np.random.randn(30, 2) + 4, np.random.randn(30, 2) - 4])
    y = np.array([1] * 30 + [0] * 30)
    model = Perceptron(learning_rate=0.1, n_iterations=200)
    model.fit(X, y)
    assert model.score(X, y) > 0.9


def test_predict_binary_labels():
    """Predictions should only be 0 or 1."""
    X, y = np.random.randn(20, 3), np.random.randint(0, 2, 20)
    preds = Perceptron(n_iterations=50).fit(X, y).predict(X)
    assert set(preds).issubset({0, 1})


def test_errors_history_length():
    """errors_ list should have one entry per iteration."""
    X, y = np.random.randn(20, 2), np.random.randint(0, 2, 20)
    model = Perceptron(n_iterations=30).fit(X, y)
    assert len(model.errors_) == 30


def test_weights_shape():
    X, y = np.random.randn(10, 4), np.random.randint(0, 2, 10)
    model = Perceptron(n_iterations=20).fit(X, y)
    assert model.weights.shape == (4,)


def test_predict_shape():
    X_tr, y_tr = np.random.randn(20, 3), np.random.randint(0, 2, 20)
    X_te = np.random.randn(7, 3)
    preds = Perceptron(n_iterations=20).fit(X_tr, y_tr).predict(X_te)
    assert preds.shape == (7,)
