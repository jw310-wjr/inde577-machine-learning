"""Tests for KMeans."""

import numpy as np
import pytest
from rice_ml.unsupervised_learning.kmeans import KMeans


def test_correct_n_clusters():
    """Should produce exactly n_clusters unique labels."""
    np.random.seed(0)
    X = np.vstack([
        np.random.randn(30, 2) + [6, 6],
        np.random.randn(30, 2) + [-6, -6],
        np.random.randn(30, 2) + [6, -6],
    ])
    model = KMeans(n_clusters=3, random_state=42).fit(X)
    assert len(np.unique(model.labels_)) == 3


def test_labels_shape():
    X = np.random.randn(50, 3)
    model = KMeans(n_clusters=4, random_state=0).fit(X)
    assert model.labels_.shape == (50,)


def test_inertia_nonnegative():
    X = np.random.randn(40, 2)
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert model.inertia_ >= 0


def test_more_clusters_less_inertia():
    """Inertia decreases as n_clusters increases."""
    np.random.seed(1)
    X = np.random.randn(80, 2)
    inertia2 = KMeans(n_clusters=2, random_state=0).fit(X).inertia_
    inertia5 = KMeans(n_clusters=5, random_state=0).fit(X).inertia_
    assert inertia5 <= inertia2


def test_predict_shape():
    np.random.seed(3)
    X = np.vstack([np.random.randn(20, 2) + 4, np.random.randn(20, 2) - 4])
    model = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = model.predict(X)
    assert labels.shape == (40,)


def test_fit_predict_consistent():
    X = np.random.randn(30, 2)
    model = KMeans(n_clusters=3, random_state=7)
    labels_fit = model.fit(X).labels_
    labels_pred = model.predict(X)
    np.testing.assert_array_equal(labels_fit, labels_pred)


def test_centroids_shape():
    X = np.random.randn(40, 5)
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert model.centroids.shape == (3, 5)
