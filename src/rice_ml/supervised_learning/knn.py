"""
knn.py
K-Nearest Neighbors — Classifier and Regressor.
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    Predicts the class of a query point by majority vote among its k
    nearest neighbors in the training set.

    Parameters
    ----------
    k               : int, number of neighbors (default 5)
    distance_metric : str, 'euclidean' | 'manhattan' | 'minkowski' (default 'euclidean')
    """

    def __init__(self, k=5, distance_metric="euclidean", distance=None):
        self.k = k
        self.distance_metric = distance if distance is not None else distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data (lazy learner)."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self

    def _distance(self, x1, x2):
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == "minkowski":
            p = 3
            return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _predict_single(self, x):
        distances = np.array([self._distance(x, x_tr) for x_tr in self.X_train])
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        """Return predicted class labels for samples in X."""
        return np.array([self._predict_single(x) for x in np.array(X, dtype=float)])

    def score(self, X, y):
        """Accuracy on dataset (X, y)."""
        return np.mean(self.predict(X) == np.array(y))


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor.

    Predicts the target of a query point as the mean of its k nearest
    neighbors' targets.

    Parameters
    ----------
    k               : int  (default 5)
    distance_metric : str, 'euclidean' | 'manhattan' (default 'euclidean')
    """

    def __init__(self, k=5, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        return self

    def _distance(self, x1, x2):
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _predict_single(self, x):
        distances = np.array([self._distance(x, x_tr) for x_tr in self.X_train])
        k_indices = np.argsort(distances)[: self.k]
        return np.mean(self.y_train[k_indices])

    def predict(self, X):
        """Return predicted target values for samples in X."""
        return np.array([self._predict_single(x) for x in np.array(X, dtype=float)])

    def score(self, X, y):
        """R² score on dataset (X, y)."""
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
