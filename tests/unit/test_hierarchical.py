"""Tests for HierarchicalClustering."""

import numpy as np
import pytest
from rice_ml.unsupervised_learning.hierarchical import HierarchicalClustering


# ── Shape / basic sanity ──────────────────────────────────────────────────────

def test_predict_shape():
    X = np.random.randn(50, 3)
    model = HierarchicalClustering(n_clusters=3)
    labels = model.fit_predict(X)
    assert labels.shape == (50,)


def test_predict_label_count():
    X = np.random.randn(40, 2)
    for k in [2, 3, 4]:
        labels = HierarchicalClustering(n_clusters=k).fit_predict(X)
        assert len(set(labels)) == k


# ── Well-separated clusters ───────────────────────────────────────────────────

def test_separated_clusters():
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.randn(30, 2) + [0, 0],
        rng.randn(30, 2) + [10, 0],
        rng.randn(30, 2) + [5, 8],
    ])
    labels = HierarchicalClustering(n_clusters=3).fit_predict(X)
    # Each true cluster should map to exactly one predicted label
    from collections import Counter
    for start in [0, 30, 60]:
        c = Counter(labels[start:start + 30])
        assert c.most_common(1)[0][1] >= 25  # ≥25/30 in same cluster


# ── Linkage options ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_linkage_options(linkage):
    X = np.random.randn(30, 2)
    labels = HierarchicalClustering(
        n_clusters=3, linkage=linkage
    ).fit_predict(X)
    assert labels.shape == (30,)
    assert len(set(labels)) == 3


# ── fit and fit_predict consistency ──────────────────────────────────────────

def test_fit_then_labels():
    X = np.random.randn(20, 2)
    model = HierarchicalClustering(n_clusters=2)
    model.fit(X)
    assert hasattr(model, "labels_")
    assert model.labels_.shape == (20,)
