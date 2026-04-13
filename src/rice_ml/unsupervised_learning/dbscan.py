"""
dbscan.py
DBSCAN — Density-Based Spatial Clustering of Applications with Noise.
"""

import numpy as np


class DBSCAN:
    """
    DBSCAN Clustering.

    Finds clusters of arbitrary shape by grouping together points that are
    closely packed. Points in low-density regions are labelled as noise (-1).

    Parameters
    ----------
    eps         : float, maximum distance to be considered a neighbor (default 0.5)
    min_samples : int, minimum points in a neighborhood to form a core point (default 5)

    Attributes
    ----------
    labels_              : ndarray of shape (n_samples,)
                           Cluster labels; -1 indicates noise.
    core_sample_indices_ : ndarray of indices of core samples.
    """

    _UNVISITED = -2
    _NOISE = -1

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _euclidean(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _region_query(self, X, idx):
        """Return indices of all points within eps of X[idx]."""
        return [
            i for i, pt in enumerate(X)
            if self._euclidean(X[idx], pt) <= self.eps
        ]

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id):
        """Grow cluster cluster_id from seed point idx."""
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            pt = neighbors[i]
            if labels[pt] == self._UNVISITED:
                labels[pt] = cluster_id
                new_nbrs = self._region_query(X, pt)
                if len(new_nbrs) >= self.min_samples:
                    neighbors.extend(new_nbrs)
            elif labels[pt] == self._NOISE:
                labels[pt] = cluster_id
            i += 1

    # ── public API ────────────────────────────────────────────────────

    def fit(self, X):
        X = np.array(X, dtype=float)
        n = len(X)
        labels = np.full(n, self._UNVISITED, dtype=int)
        cluster_id = 0

        for idx in range(n):
            if labels[idx] != self._UNVISITED:
                continue
            neighbors = self._region_query(X, idx)
            if len(neighbors) < self.min_samples:
                labels[idx] = self._NOISE
            else:
                self._expand_cluster(X, labels, idx, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array([
            i for i in range(n)
            if len(self._region_query(X, i)) >= self.min_samples
        ])
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        return self.fit(X).labels_
