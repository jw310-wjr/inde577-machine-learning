"""
Tests for Autoencoder
"""
import numpy as np
import pytest
from rice_ml.unsupervised_learning.autoencoder import Autoencoder


def make_data(n=100, d=8, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, d).astype(float)


def test_fit_transform_shape():
    X = make_data()
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=10, random_state=0)
    Z = ae.fit_transform(X)
    assert Z.shape == (100, 2)


def test_reconstruct_shape():
    X = make_data()
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=10, random_state=0)
    ae.fit(X)
    R = ae.reconstruct(X)
    assert R.shape == X.shape


def test_reconstruction_loss_decreases():
    X = make_data(n=80, d=6)
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=200,
                     learning_rate=1e-3, random_state=0)
    ae.fit(X)
    assert ae.loss_history_[0] > ae.loss_history_[-1]


def test_reconstruction_loss_value():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.datasets import load_iris
    X = MinMaxScaler().fit_transform(load_iris().data)
    ae = Autoencoder(encoding_dims=(8, 3), n_iterations=300,
                     learning_rate=1e-3, random_state=0)
    ae.fit(X)
    assert ae.reconstruction_loss(X) < 0.05


def test_encode_decode_roundtrip():
    X = make_data(n=50, d=6)
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=50, random_state=1)
    ae.fit(X)
    Z = ae.encode(X)
    R = ae.decode(Z)
    assert R.shape == X.shape


def test_transform_equals_encode():
    X = make_data()
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=10, random_state=2)
    ae.fit(X)
    np.testing.assert_array_equal(ae.transform(X), ae.encode(X))


def test_loss_history_length():
    X = make_data()
    n_iter = 15
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=n_iter, random_state=0)
    ae.fit(X)
    assert len(ae.loss_history_) == n_iter


def test_bottleneck_dimension():
    X = make_data(n=60, d=10)
    ae = Autoencoder(encoding_dims=(6, 3, 2), n_iterations=10, random_state=0)
    Z = ae.fit_transform(X)
    assert Z.shape[1] == 2


def test_reconstruction_in_range():
    X = make_data()
    ae = Autoencoder(encoding_dims=(4, 2), n_iterations=50, random_state=0)
    ae.fit(X)
    R = ae.reconstruct(X)
    assert R.min() >= -0.01 and R.max() <= 1.01
