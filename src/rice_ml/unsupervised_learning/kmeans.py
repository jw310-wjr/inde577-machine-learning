"""
kmeans.py
K-Means Clustering with k-means++ initialisation.
"""

import numpy as np


class KMeans:
    """
    K-Means Clustering algorithm.

    Uses k-means++ initialisation by default, which chooses initial centroids
    that are spread out, leading to faster convergence and better results.

    Parameters
    ----------
    n_clusters   : int   (default 3)
    max_iter     : int, maximum number of iterations (default 300)
    tol          : float, convergence tolerance on centroid shift (default 1e-4)
    init         : str, 'k-means++' | 'random' (default 'k-means++')
    random_state : int or None

    Attributes
    ----------
    centroids  : ndarray of shape (n_clusters, n_features)
    labels_    : ndarray of shape (n_samples,) — cluster index per sample
    inertia_   : float — sum of squared distances to nearest centroid
    n_iter_    : int   — number of iterations run
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 init="k-means++", random_state=None, k=None):
        self.n_clusters = k if k is not None else n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    # ── initialisation ────────────────────────────────────────────────

    def _init_centroids(self, X, rng):
        n_samples = len(X)
        if self.init == "random":
            return X[rng.choice(n_samples, self.n_clusters, replace=False)].copy()

        # k-means++ : spread-out probabilistic seeding
        centroids = [X[rng.randint(0, n_samples)]]
        for _ in range(1, self.n_clusters):
            # Squared distance to the nearest chosen centroid
            dists = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dists / dists.sum()
            centroids.append(X[rng.choice(n_samples, p=probs)])
        return np.array(centroids)

    # ── fitting ───────────────────────────────────────────────────────

    def fit(self, X):
        X = np.array(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        self.centroids = self._init_centroids(X, rng)

        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()

            # Assignment step
            dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in self.centroids])
            self.labels_ = np.argmin(dists, axis=0)

            # Update step
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    self.centroids[k] = X[mask].mean(axis=0)

            self.n_iter_ = i + 1

            # Convergence check
            shift = np.max(np.linalg.norm(self.centroids - old_centroids, axis=1))
            if shift < self.tol:
                break

        self.inertia_ = float(sum(
            np.sum((X[self.labels_ == k] - self.centroids[k]) ** 2)
            for k in range(self.n_clusters)
        ))
        return self

    def predict(self, X):
        """Assign new samples to the nearest centroid."""
        X = np.array(X, dtype=float)
        dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in self.centroids])
        return np.argmin(dists, axis=0)

    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        return self.fit(X).labels_
