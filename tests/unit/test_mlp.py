"""Tests for MLP (Multi-Layer Perceptron) classifier."""

import numpy as np
import pytest
from rice_ml.supervised_learning.mlp import MLP


# ── Shape / basic sanity ──────────────────────────────────────────────────────

def test_predict_shape():
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    mlp = MLP(hidden_layers=(8,), n_iterations=10, random_state=0)
    mlp.fit(X, y)
    assert mlp.predict(X).shape == (50,)


def test_predict_binary_labels():
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)
    mlp = MLP(hidden_layers=(4,), n_iterations=10, random_state=1)
    mlp.fit(X, y)
    assert set(mlp.predict(X)).issubset({0, 1})


# ── XOR — nonlinear problem ───────────────────────────────────────────────────

def test_xor_learnable():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    # Repeat to give more signal
    X = np.tile(X, (20, 1))
    y = np.tile(y, 20)
    mlp = MLP(hidden_layers=(8,), n_iterations=500, learning_rate=0.05, random_state=0)
    mlp.fit(X, y)
    assert mlp.score(X, y) > 0.70


# ── Multiclass ────────────────────────────────────────────────────────────────

def test_multiclass_predict_valid():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    mlp = MLP(hidden_layers=(16,), n_iterations=200, learning_rate=0.01, random_state=0)
    mlp.fit(X, y)
    preds = mlp.predict(X)
    assert set(preds).issubset({0, 1, 2})
    assert preds.shape == (150,)


# ── Score improves with more epochs ──────────────────────────────────────────

def test_score_positive():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mlp = MLP(hidden_layers=(8,), n_iterations=300, learning_rate=0.05, random_state=0)
    mlp.fit(X, y)
    assert mlp.score(X, y) > 0.70


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_reproducible():
    X = np.random.randn(40, 3)
    y = np.random.randint(0, 2, 40)
    m1 = MLP(hidden_layers=(4,), n_iterations=20, random_state=3).fit(X, y)
    m2 = MLP(hidden_layers=(4,), n_iterations=20, random_state=3).fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
