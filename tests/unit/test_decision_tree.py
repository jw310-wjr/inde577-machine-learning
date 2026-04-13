"""Tests for DecisionTreeClassifier and DecisionTreeRegressor."""

import numpy as np
import pytest
from rice_ml.supervised_learning.decision_tree import (
    DecisionTreeClassifier, DecisionTreeRegressor
)


# ── Classifier ────────────────────────────────────────────────────────────────

def test_clf_perfect_split():
    """Depth-1 tree should perfectly split cleanly separable data."""
    X = np.array([[0.0, 1], [0, 2], [10, 1], [10, 2]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert model.score(X, y) == 1.0


def test_clf_gini_accuracy():
    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) + 3, np.random.randn(40, 2) - 3])
    y = np.array([0] * 40 + [1] * 40)
    model = DecisionTreeClassifier(max_depth=3, criterion="gini").fit(X, y)
    assert model.score(X, y) > 0.9


def test_clf_entropy_accuracy():
    np.random.seed(42)
    X = np.vstack([np.random.randn(40, 2) + 3, np.random.randn(40, 2) - 3])
    y = np.array([0] * 40 + [1] * 40)
    model = DecisionTreeClassifier(max_depth=3, criterion="entropy").fit(X, y)
    assert model.score(X, y) > 0.9


def test_clf_predict_shape():
    X, y = np.random.randn(30, 3), np.random.randint(0, 3, 30)
    preds = DecisionTreeClassifier(max_depth=3).fit(X, y).predict(X)
    assert preds.shape == (30,)


def test_clf_pure_leaf():
    """All same class → single leaf node, perfect accuracy."""
    X = np.random.randn(20, 2)
    y = np.zeros(20, dtype=int)
    model = DecisionTreeClassifier().fit(X, y)
    assert model.score(X, y) == 1.0


# ── Regressor ─────────────────────────────────────────────────────────────────

def test_reg_overfit():
    """Unlimited depth tree should memorise the training set."""
    np.random.seed(1)
    X, y = np.random.randn(25, 2), np.random.randn(25)
    model = DecisionTreeRegressor(max_depth=None).fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-6)


def test_reg_predict_shape():
    X, y = np.random.randn(40, 3), np.random.randn(40)
    preds = DecisionTreeRegressor(max_depth=4).fit(X, y).predict(X)
    assert preds.shape == (40,)


def test_reg_r2_depth_tradeoff():
    """Deeper tree should achieve at least as good R² as shallower tree."""
    np.random.seed(7)
    X = np.random.randn(60, 3)
    y = X @ np.array([1.0, -1.0, 2.0])
    r2_deep = DecisionTreeRegressor(max_depth=6).fit(X, y).score(X, y)
    r2_shallow = DecisionTreeRegressor(max_depth=1).fit(X, y).score(X, y)
    assert r2_deep >= r2_shallow
