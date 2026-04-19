"""Tests for RandomForestClassifier and RandomForestRegressor."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier as SkRFC
from sklearn.ensemble import RandomForestRegressor as SkRFR
from rice_ml.supervised_learning.random_forest import RandomForestClassifier, RandomForestRegressor


# ── Shape / basic sanity ──────────────────────────────────────────────────────

def test_classifier_predict_shape():
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 3, 60)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (60,)


def test_classifier_predict_valid_classes():
    X = np.random.randn(50, 3)
    y = np.array([0, 1, 2] * 16 + [0, 1])
    clf = RandomForestClassifier(n_estimators=5, random_state=1)
    clf.fit(X, y)
    assert set(clf.predict(X)).issubset({0, 1, 2})


def test_regressor_predict_shape():
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    reg = RandomForestRegressor(n_estimators=10, random_state=0)
    reg.fit(X, y)
    assert reg.predict(X).shape == (50,)


# ── Accuracy / performance ────────────────────────────────────────────────────

def test_classifier_breast_cancer_accuracy():
    data = load_breast_cancer()
    X, y = data.data, data.target
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.90


def test_regressor_low_train_error():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2]
    reg = RandomForestRegressor(n_estimators=30, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    mse = np.mean((preds - y) ** 2)
    assert mse < 1.0


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_classifier_reproducible():
    X = np.random.randn(40, 4)
    y = np.random.randint(0, 2, 40)
    clf1 = RandomForestClassifier(n_estimators=10, random_state=7)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=7)
    np.testing.assert_array_equal(clf1.fit(X, y).predict(X),
                                   clf2.fit(X, y).predict(X))


# ── Compare direction with sklearn ───────────────────────────────────────────

def test_classifier_comparable_to_sklearn():
    data = load_iris()
    X, y = data.data, data.target
    ours = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    sk = SkRFC(n_estimators=20, random_state=0).fit(X, y)
    # Both should beat 80% on train
    assert ours.score(X, y) > 0.80
    assert sk.score(X, y) > 0.80
