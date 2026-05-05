"""Tests for TSNE."""
import numpy as np
import pytest
from rice_ml.unsupervised_learning.tsne import TSNE


def make_clusters(n_per_class=25, n_features=8, random_state=0):
    rng = np.random.RandomState(random_state)
    X = np.vstack([
        rng.randn(n_per_class, n_features) + 6,
        rng.randn(n_per_class, n_features) - 6,
        rng.randn(n_per_class, n_features) + [0] * n_features,
    ])
    labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    return X, labels


def test_output_shape():
    """Embedding should have shape (n_samples, n_components)."""
    X, _ = make_clusters()
    Y = TSNE(n_components=2, n_iter=100, random_state=0).fit_transform(X)
    assert Y.shape == (75, 2)


def test_custom_n_components():
    X, _ = make_clusters()
    Y = TSNE(n_components=3, n_iter=100, random_state=0).fit_transform(X)
    assert Y.shape == (75, 3)


def test_clusters_separated():
    """Well-separated input clusters should be separable in 2D embedding."""
    X, labels = make_clusters()
    Y = TSNE(n_components=2, perplexity=10, n_iter=400, random_state=42).fit_transform(X)
    c0 = Y[labels == 0].mean(0)
    c1 = Y[labels == 1].mean(0)
    assert np.linalg.norm(c0 - c1) > 1.0, "Clusters 0 and 1 not separated in embedding"


def test_reproducibility():
    """Same random_state must produce identical embeddings."""
    X, _ = make_clusters()
    Y1 = TSNE(n_iter=100, random_state=7).fit_transform(X)
    Y2 = TSNE(n_iter=100, random_state=7).fit_transform(X)
    np.testing.assert_array_equal(Y1, Y2)


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different embeddings."""
    X, _ = make_clusters()
    Y1 = TSNE(n_iter=100, random_state=0).fit_transform(X)
    Y2 = TSNE(n_iter=100, random_state=99).fit_transform(X)
    assert not np.allclose(Y1, Y2)


def test_kl_divergence_stored():
    """kl_divergence_ should be a positive float after fitting."""
    X, _ = make_clusters()
    model = TSNE(n_iter=100, random_state=0)
    model.fit_transform(X)
    assert model.kl_divergence_ > 0


def test_loss_history_length():
    """loss_history_ should have exactly n_iter entries."""
    X, _ = make_clusters()
    model = TSNE(n_iter=150, random_state=0)
    model.fit_transform(X)
    assert len(model.loss_history_) == 150


def test_embedding_attribute():
    """embedding_ should be set after fit()."""
    X, _ = make_clusters()
    model = TSNE(n_iter=80, random_state=0)
    model.fit(X)
    assert model.embedding_ is not None
    assert model.embedding_.shape == (75, 2)


def test_centred_output():
    """Embedding should be re-centred to near zero mean."""
    X, _ = make_clusters()
    Y = TSNE(n_iter=200, random_state=0).fit_transform(X)
    np.testing.assert_allclose(Y.mean(axis=0), 0.0, atol=1e-6)
