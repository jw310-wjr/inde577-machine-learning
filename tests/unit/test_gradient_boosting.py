"""Tests for GradientBoostingClassifier and GradientBoostingRegressor."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from rice_ml.supervised_learning.gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


# ── Shape / basic sanity ──────────────────────────────────────────────────────

def test_classifier_predict_shape():
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 2, 60)
    clf = GradientBoostingClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    assert clf.predict(X).shape == (60,)


def test_classifier_predict_binary():
    X = np.random.randn(40, 3)
    y = np.random.randint(0, 2, 40)
    clf = GradientBoostingClassifier(n_estimators=5, random_state=1)
    clf.fit(X, y)
    assert set(clf.predict(X)).issubset({0, 1})


def test_regressor_predict_shape():
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    reg = GradientBoostingRegressor(n_estimators=10, random_state=0)
    reg.fit(X, y)
    assert reg.predict(X).shape == (50,)


# ── Performance ───────────────────────────────────────────────────────────────

def test_classifier_breast_cancer_accuracy():
    data = load_breast_cancer()
    X, y = data.data, data.target
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.88


def test_regressor_low_train_error():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 2)
    y = X[:, 0] * 3 - X[:, 1] * 2
    reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=0)
    reg.fit(X, y)
    mse = np.mean((reg.predict(X) - y) ** 2)
    assert mse < 1.5


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_regressor_reproducible():
    X = np.random.randn(40, 3)
    y = np.random.randn(40)
    r1 = GradientBoostingRegressor(n_estimators=10, random_state=9).fit(X, y)
    r2 = GradientBoostingRegressor(n_estimators=10, random_state=9).fit(X, y)
    np.testing.assert_allclose(r1.predict(X), r2.predict(X))
