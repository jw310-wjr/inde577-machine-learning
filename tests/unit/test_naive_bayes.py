"""Tests for GaussianNaiveBayes."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB as SkGNB
from rice_ml.supervised_learning.naive_bayes import GaussianNaiveBayes


# ── Shape / basic sanity ──────────────────────────────────────────────────────

def test_predict_shape():
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 3, 60)
    clf = GaussianNaiveBayes()
    clf.fit(X, y)
    assert clf.predict(X).shape == (60,)


def test_predict_valid_classes():
    X = np.random.randn(30, 2)
    y = np.array([0, 1, 2] * 10)
    clf = GaussianNaiveBayes()
    clf.fit(X, y)
    assert set(clf.predict(X)).issubset({0, 1, 2})


def test_predict_proba_shape():
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    clf = GaussianNaiveBayes().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (20, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ── Performance ───────────────────────────────────────────────────────────────

def test_iris_accuracy():
    data = load_iris()
    X, y = data.data, data.target
    clf = GaussianNaiveBayes().fit(X, y)
    assert clf.score(X, y) > 0.85


# ── Compare with sklearn ──────────────────────────────────────────────────────

def test_comparable_to_sklearn():
    data = load_iris()
    X, y = data.data, data.target
    ours = GaussianNaiveBayes().fit(X, y)
    sk = SkGNB().fit(X, y)
    # Predictions should largely agree
    agreement = np.mean(ours.predict(X) == sk.predict(X))
    assert agreement > 0.90


# ── Binary case ───────────────────────────────────────────────────────────────

def test_binary_classification():
    rng = np.random.RandomState(42)
    X0 = rng.randn(50, 2)
    X1 = rng.randn(50, 2) + 3
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    clf = GaussianNaiveBayes().fit(X, y)
    assert clf.score(X, y) > 0.90
