"""
hierarchical.py
Agglomerative Hierarchical Clustering with multiple linkage criteria.
"""

import numpy as np


class HierarchicalClustering:
    """
    Agglomerative (bottom-up) Hierarchical Clustering.

    Starts with each sample as its own cluster and iteratively merges the
    two closest clusters until the desired number of clusters is reached.

    Parameters
    ----------
    n_clusters : int  (default 2)
    linkage    : str, distance between clusters —
                 'single'   : minimum pairwise distance
                 'complete' : maximum pairwise distance
                 'average'  : mean pairwise distance
                 'ward'     : minimises within-cluster variance (default)

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,) — cluster index per sample
    """

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    @staticmethod
    def _euclidean(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _cluster_distance(self, c1_idx, c2_idx, X):
        """Distance between two clusters given their point index sets."""
        pts1 = X[list(c1_idx)]
        pts2 = X[list(c2_idx)]
        pairwise = np.array([
            [self._euclidean(p1, p2) for p2 in pts2]
            for p1 in pts1
        ])
        if self.linkage == "single":
            return np.min(pairwise)
        if self.linkage == "complete":
            return np.max(pairwise)
        if self.linkage == "average":
            return np.mean(pairwise)
        if self.linkage == "ward":
            n1, n2 = len(pts1), len(pts2)
            c1, c2 = pts1.mean(axis=0), pts2.mean(axis=0)
            return np.sqrt(n1 * n2 / (n1 + n2)) * self._euclidean(c1, c2)
        raise ValueError(f"Unknown linkage: {self.linkage}")

    def fit(self, X):
        X = np.array(X, dtype=float)
        n_samples = len(X)
        clusters = [{i} for i in range(n_samples)]

        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            merge_i, merge_j = 0, 1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._cluster_distance(clusters[i], clusters[j], X)
                    if d < min_dist:
                        min_dist, merge_i, merge_j = d, i, j

            merged = clusters[merge_i] | clusters[merge_j]
            clusters = [c for k, c in enumerate(clusters) if k not in (merge_i, merge_j)]
            clusters.append(merged)

        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label
        return self

    def fit_predict(self, X):
        """Fit and return cluster labels."""
        return self.fit(X).labels_
