"""Tests for LinearRegression."""

import numpy as np
import pytest
from rice_ml.supervised_learning.linear_regression import LinearRegression


def test_ols_perfect_fit():
    """OLS should recover exact coefficients for noiseless data."""
    X = np.array([[1.0, 2], [3, 4], [5, 6], [7, 8]])
    y = 2 * X[:, 0] + 3 * X[:, 1]
    model = LinearRegression(method="ols")
    model.fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-6)


def test_gd_predict_shape():
    """Gradient descent predictions should have the right shape."""
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    model = LinearRegression(learning_rate=0.01, n_iterations=200, method="gradient_descent")
    model.fit(X, y)
    assert model.predict(X).shape == (20,)


def test_r2_near_one_ols():
    """R² for noiseless linear data should be ~1.0."""
    X = np.random.randn(100, 4)
    y = X @ np.array([1.0, -2.0, 0.5, 3.0])
    model = LinearRegression(method="ols")
    model.fit(X, y)
    assert model.r2_score(X, y) > 0.999


def test_loss_history_length():
    """loss_history should have one entry per iteration."""
    X, y = np.random.randn(30, 2), np.random.randn(30)
    model = LinearRegression(n_iterations=150, method="gradient_descent")
    model.fit(X, y)
    assert len(model.loss_history) == 150


def test_loss_decreasing():
    """Loss should generally decrease over gradient descent iterations."""
    np.random.seed(0)
    X, y = np.random.randn(50, 2), np.random.randn(50)
    model = LinearRegression(learning_rate=0.01, n_iterations=500, method="gradient_descent")
    model.fit(X, y)
    assert model.loss_history[0] >= model.loss_history[-1]


def test_single_feature_ols():
    """Simple y=2x relationship recovered by OLS."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 2.0 * X.ravel()
    model = LinearRegression(method="ols")
    model.fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)


def test_rmse_nonnegative():
    X, y = np.random.randn(30, 2), np.random.randn(30)
    model = LinearRegression(method="ols").fit(X, y)
    assert model.rmse(X, y) >= 0
