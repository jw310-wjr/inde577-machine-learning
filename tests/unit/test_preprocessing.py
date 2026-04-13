"""Tests for StandardScaler, MinMaxScaler, and train_test_split."""

import numpy as np
import pytest
from rice_ml.processing.preprocessing import StandardScaler, MinMaxScaler, train_test_split


# ── StandardScaler ────────────────────────────────────────────────────────────

def test_standard_mean_zero():
    X = np.array([[1.0, 2], [3, 4], [5, 6]])
    X_s = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(X_s.mean(axis=0), 0, atol=1e-10)


def test_standard_unit_variance():
    np.random.seed(0)
    X = np.random.randn(100, 3) * 5 + 10
    X_s = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(X_s.std(axis=0), 1.0, atol=1e-6)


def test_standard_inverse():
    X = np.random.randn(30, 4) * 3 + 5
    scaler = StandardScaler()
    X_back = scaler.inverse_transform(scaler.fit_transform(X))
    np.testing.assert_allclose(X_back, X, atol=1e-10)


def test_standard_transform_new_data():
    """Transform on new data should use training statistics."""
    X_tr = np.array([[0.0, 0], [2, 2], [4, 4]])
    X_te = np.array([[1.0, 1]])
    scaler = StandardScaler().fit(X_tr)
    result = scaler.transform(X_te)
    assert result.shape == (1, 2)


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

def test_minmax_range_01():
    X = np.array([[1.0, 10], [2, 20], [3, 30]])
    X_s = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    np.testing.assert_allclose(X_s.min(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X_s.max(axis=0), 1, atol=1e-10)


def test_minmax_custom_range():
    X = np.array([[1.0], [2], [3], [4], [5]])
    X_s = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    assert X_s.min() >= -1 - 1e-9
    assert X_s.max() <= 1 + 1e-9


def test_minmax_inverse():
    X = np.random.randn(20, 3)
    scaler = MinMaxScaler()
    X_back = scaler.inverse_transform(scaler.fit_transform(X))
    np.testing.assert_allclose(X_back, X, atol=1e-10)


# ── train_test_split ──────────────────────────────────────────────────────────

def test_split_sizes():
    X, y = np.random.randn(100, 3), np.random.randn(100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_tr) == 80 and len(X_te) == 20
    assert len(y_tr) == 80 and len(y_te) == 20


def test_split_no_overlap():
    """Train and test sets should not share any rows."""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    train_set = set(map(tuple, X_tr.tolist()))
    test_set = set(map(tuple, X_te.tolist()))
    assert len(train_set & test_set) == 0


def test_split_reproducible():
    X, y = np.random.randn(50, 2), np.arange(50)
    split1 = train_test_split(X, y, random_state=99)
    split2 = train_test_split(X, y, random_state=99)
    np.testing.assert_array_equal(split1[0], split2[0])


def test_split_total_size():
    X, y = np.random.randn(80, 2), np.random.randn(80)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25)
    assert len(X_tr) + len(X_te) == 80
