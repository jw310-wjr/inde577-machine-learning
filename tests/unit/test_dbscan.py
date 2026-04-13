"""Tests for DBSCAN."""

import numpy as np
import pytest
from rice_ml.unsupervised_learning.dbscan import DBSCAN


def test_two_dense_clusters():
    """Two well-separated dense blobs → two clusters, no noise."""
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) * 0.2 + [0, 0],
        np.random.randn(20, 2) * 0.2 + [10, 10],
    ])
    model = DBSCAN(eps=0.5, min_samples=3).fit(X)
    valid_labels = model.labels_[model.labels_ != -1]
    assert len(np.unique(valid_labels)) == 2


def test_noise_point_labeled_minus_one():
    """Isolated point far from any cluster should be labelled -1 (noise)."""
    X = np.array([[0.0, 0], [0.1, 0.1], [0.2, 0], [100, 100]])
    model = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert model.labels_[3] == -1


def test_labels_shape():
    X = np.random.randn(30, 2)
    model = DBSCAN(eps=0.5, min_samples=3).fit(X)
    assert model.labels_.shape == (30,)


def test_fit_predict_returns_labels():
    X = np.random.randn(25, 2)
    labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(X)
    assert labels.shape == (25,)


def test_core_sample_indices_exist():
    np.random.seed(5)
    X = np.vstack([np.random.randn(15, 2) * 0.3, np.random.randn(15, 2) * 0.3 + 5])
    model = DBSCAN(eps=0.8, min_samples=3).fit(X)
    assert model.core_sample_indices_ is not None
    assert len(model.core_sample_indices_) > 0
