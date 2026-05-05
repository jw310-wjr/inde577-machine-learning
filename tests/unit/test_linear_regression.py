"""Tests for LinearRegression, RidgeRegression, and LassoRegression."""

import numpy as np
import pytest
from rice_ml.supervised_learning.linear_regression import (
    LinearRegression, RidgeRegression, LassoRegression
)


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


# ── Ridge Regression ─────────────────────────────────────────────────────────

def test_ridge_r2_noiseless():
    """Ridge should achieve near-perfect R² on noiseless linear data."""
    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X @ np.array([1.0, -2.0, 0.5, 3.0])
    model = RidgeRegression(alpha=0.001).fit(X, y)
    assert model.r2_score(X, y) > 0.999


def test_ridge_predict_shape():
    """Ridge predictions should match the number of test samples."""
    X_tr, y_tr = np.random.randn(50, 3), np.random.randn(50)
    X_te = np.random.randn(20, 3)
    model = RidgeRegression(alpha=1.0).fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (20,)


def test_ridge_shrinks_weights():
    """Higher alpha should produce smaller coefficient norms."""
    np.random.seed(1)
    X = np.random.randn(80, 5)
    y = X @ np.ones(5) + np.random.randn(80) * 0.1
    low  = RidgeRegression(alpha=0.001).fit(X, y)
    high = RidgeRegression(alpha=1000.0).fit(X, y)
    assert np.linalg.norm(high.weights) < np.linalg.norm(low.weights)


def test_ridge_alpha_zero_matches_ols():
    """Ridge with alpha≈0 should closely match OLS."""
    np.random.seed(2)
    X = np.random.randn(60, 3)
    y = X @ [1.0, -1.0, 2.0] + np.random.randn(60) * 0.05
    ols   = LinearRegression(method="ols").fit(X, y)
    ridge = RidgeRegression(alpha=1e-9).fit(X, y)
    np.testing.assert_allclose(ridge.weights, ols.weights, rtol=1e-4)


def test_ridge_rmse_nonnegative():
    X, y = np.random.randn(40, 2), np.random.randn(40)
    model = RidgeRegression(alpha=1.0).fit(X, y)
    assert model.rmse(X, y) >= 0


# ── Lasso Regression ─────────────────────────────────────────────────────────

def test_lasso_r2_noisy():
    """Lasso should achieve reasonable R² on noisy linear data."""
    np.random.seed(3)
    X = np.random.randn(120, 5)
    y = X @ [2.0, -1.0, 0.0, 1.5, 0.0] + np.random.randn(120) * 0.3
    model = LassoRegression(alpha=0.05, n_iterations=2000).fit(X, y)
    assert model.r2_score(X, y) > 0.80


def test_lasso_sparsity():
    """High alpha should drive most coefficients to exactly zero."""
    np.random.seed(4)
    X = np.random.randn(80, 6)
    y = X[:, 0] + np.random.randn(80) * 0.1   # only first feature matters
    model = LassoRegression(alpha=2.0, n_iterations=2000).fit(X, y)
    n_zero = np.sum(np.abs(model.weights) < 1e-6)
    assert n_zero >= 3, f"Expected sparsity, got {n_zero} zero coefs"


def test_lasso_predict_shape():
    """Lasso predictions should match the number of test samples."""
    X_tr, y_tr = np.random.randn(60, 4), np.random.randn(60)
    X_te = np.random.randn(15, 4)
    model = LassoRegression(alpha=0.1).fit(X_tr, y_tr)
    assert model.predict(X_te).shape == (15,)


def test_lasso_rmse_nonnegative():
    X, y = np.random.randn(40, 2), np.random.randn(40)
    model = LassoRegression(alpha=0.5).fit(X, y)
    assert model.rmse(X, y) >= 0


def test_lasso_convergence_tolerance():
    """Lasso with tight tolerance should converge without error."""
    np.random.seed(5)
    X = np.random.randn(50, 3)
    y = X @ [1.0, 0.0, -1.0] + np.random.randn(50) * 0.2
    model = LassoRegression(alpha=0.01, n_iterations=5000, tol=1e-6).fit(X, y)
    assert model.r2_score(X, y) > 0.70
