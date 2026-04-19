"""Tests for SVM (SupportVectorClassifier)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from rice_ml.supervised_learning.svm import SVM


# ── Shape / basic sanity ──────────────────────────────────────────────────────

def test_predict_shape():
    X = np.random.randn(50, 4)
    y = np.random.choice([-1, 1], 50)
    svm = SVM(n_iterations=50)
    svm.fit(X, y)
    assert svm.predict(X).shape == (50,)


def test_predict_binary_labels():
    X = np.random.randn(40, 3)
    y = np.random.choice([-1, 1], 40)
    svm = SVM(n_iterations=50)
    svm.fit(X, y)
    assert set(svm.predict(X)).issubset({-1, 1})


# ── Linearly separable data ───────────────────────────────────────────────────

def test_linearly_separable():
    rng = np.random.RandomState(0)
    X_pos = rng.randn(30, 2) + np.array([3, 3])
    X_neg = rng.randn(30, 2) + np.array([-3, -3])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 30 + [-1] * 30)
    svm = SVM(C=1.0, n_iterations=500, learning_rate=0.01)
    svm.fit(X, y)
    assert svm.score(X, y) > 0.90


# ── Breast cancer with {0,1} labels ──────────────────────────────────────────

def test_breast_cancer_accuracy():
    data = load_breast_cancer()
    X, y = data.data, data.target
    y_svm = np.where(y == 0, -1, 1)
    from rice_ml.processing.preprocessing import StandardScaler
    X_s = StandardScaler().fit_transform(X)
    svm = SVM(C=0.1, n_iterations=200, learning_rate=0.001)
    svm.fit(X_s, y_svm)
    assert svm.score(X_s, y_svm) > 0.80


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_reproducible():
    X = np.random.randn(40, 3)
    y = np.random.choice([-1, 1], 40)
    s1 = SVM(n_iterations=20).fit(X, y)
    s2 = SVM(n_iterations=20).fit(X, y)
    np.testing.assert_array_equal(s1.predict(X), s2.predict(X))
