"""Tests for PCA."""

import numpy as np
import pytest
from rice_ml.unsupervised_learning.pca import PCA


def test_output_shape():
    X = np.random.randn(50, 5)
    X_t = PCA(n_components=2).fit_transform(X)
    assert X_t.shape == (50, 2)


def test_explained_variance_ratio_sums_to_one():
    X = np.random.randn(100, 4)
    model = PCA(n_components=4).fit(X)
    assert np.isclose(model.explained_variance_ratio_.sum(), 1.0, atol=1e-6)


def test_components_orthonormal():
    """Principal components should be orthonormal: V Vᵀ = I."""
    X = np.random.randn(100, 4)
    model = PCA(n_components=4).fit(X)
    gram = model.components_ @ model.components_.T
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-6)


def test_perfect_reconstruction_full_components():
    """Transforming with all components and inverting should recover X."""
    X = np.random.randn(30, 4)
    model = PCA(n_components=4)
    X_t = model.fit_transform(X)
    X_rec = model.inverse_transform(X_t)
    np.testing.assert_allclose(X_rec, X, atol=1e-6)


def test_variance_sorted_descending():
    X = np.random.randn(100, 4)
    model = PCA(n_components=4).fit(X)
    ev = model.explained_variance_
    assert all(ev[i] >= ev[i + 1] for i in range(len(ev) - 1))


def test_mean_subtracted():
    """After fitting, PCA should store the training mean."""
    X = np.random.randn(50, 3) + 10
    model = PCA(n_components=2).fit(X)
    np.testing.assert_allclose(model.mean_, X.mean(axis=0), atol=1e-10)
