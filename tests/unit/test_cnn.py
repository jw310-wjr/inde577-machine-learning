"""
Tests for SimpleCNN
"""
import numpy as np
import pytest
from rice_ml.supervised_learning.cnn import SimpleCNN


def make_binary_images(n=80, H=8, W=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, H, W)
    y = rng.randint(0, 2, n)
    return X, y


def make_multiclass_images(n=100, n_classes=3, H=8, W=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, H, W)
    y = rng.randint(0, n_classes, n)
    return X, y


def test_predict_shape():
    X, y = make_binary_images()
    cnn = SimpleCNN(n_filters=4, kernel_size=3, n_iterations=3, random_state=0)
    cnn.fit(X, y)
    preds = cnn.predict(X)
    assert preds.shape == (len(X),)


def test_predict_proba_shape():
    X, y = make_binary_images()
    cnn = SimpleCNN(n_filters=4, kernel_size=3, n_iterations=3, random_state=0)
    cnn.fit(X, y)
    proba = cnn.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_sums_to_one():
    X, y = make_binary_images()
    cnn = SimpleCNN(n_filters=4, kernel_size=3, n_iterations=3, random_state=0)
    cnn.fit(X, y)
    proba = cnn.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_loss_history_length():
    X, y = make_binary_images()
    n_iter = 7
    cnn = SimpleCNN(n_iterations=n_iter, random_state=0)
    cnn.fit(X, y)
    assert len(cnn.loss_history_) == n_iter


def test_loss_decreases():
    from sklearn.datasets import load_digits
    digits = load_digits()
    mask = digits.target < 2
    X = digits.data[mask].reshape(-1, 8, 8) / 16.0
    y = digits.target[mask]
    cnn = SimpleCNN(n_filters=8, kernel_size=3, dense_units=32,
                    learning_rate=1e-2, n_iterations=30, random_state=0)
    cnn.fit(X, y)
    assert cnn.loss_history_[0] > cnn.loss_history_[-1]


def test_binary_accuracy():
    from sklearn.datasets import load_digits
    digits = load_digits()
    mask = digits.target < 2
    X = digits.data[mask].reshape(-1, 8, 8) / 16.0
    y = digits.target[mask]
    cnn = SimpleCNN(n_filters=8, kernel_size=3, dense_units=32,
                    learning_rate=1e-2, n_iterations=40, random_state=0)
    cnn.fit(X, y)
    assert cnn.score(X, y) > 0.85


def test_multiclass_output():
    X, y = make_multiclass_images(n_classes=4)
    cnn = SimpleCNN(n_filters=4, kernel_size=3, n_iterations=5, random_state=0)
    cnn.fit(X, y)
    preds = cnn.predict(X)
    assert set(preds).issubset({0, 1, 2, 3})


def test_reproducible():
    X, y = make_binary_images()
    c1 = SimpleCNN(n_iterations=5, random_state=7)
    c2 = SimpleCNN(n_iterations=5, random_state=7)
    c1.fit(X, y)
    c2.fit(X, y)
    np.testing.assert_array_equal(c1.predict(X), c2.predict(X))
