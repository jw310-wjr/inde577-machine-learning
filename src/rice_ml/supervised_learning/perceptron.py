"""
perceptron.py
Single-layer Perceptron for binary classification.
"""

import numpy as np


class Perceptron:
    """
    Single-layer Perceptron — the simplest neural network unit.

    Update rule (online learning):
        w ← w + η (y - ŷ) x
        b ← b + η (y - ŷ)

    Converges only for linearly separable data (Perceptron Convergence Theorem).

    Parameters
    ----------
    learning_rate : float (default 0.01)
    n_iterations  : int   (default 1000)

    Attributes
    ----------
    weights  : ndarray of shape (n_features,)
    bias     : float
    errors_  : list of int — number of misclassifications per epoch
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_ = []

    @staticmethod
    def _step(z):
        """Heaviside step function: 1 if z >= 0 else 0."""
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """Train the perceptron on binary-labeled data (y ∈ {0, 1})."""
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_ = []

        for _ in range(self.n_iterations):
            epoch_errors = 0
            for xi, yi in zip(X, y):
                y_hat = self._step(np.dot(xi, self.weights) + self.bias)
                update = self.learning_rate * (yi - y_hat)
                self.weights += update * xi
                self.bias += update
                epoch_errors += int(update != 0)
            self.errors_.append(epoch_errors)

        return self

    def predict(self, X):
        """Return binary predictions for samples in X."""
        return self._step(np.array(X, dtype=float) @ self.weights + self.bias)

    def score(self, X, y):
        """Accuracy on dataset (X, y)."""
        return np.mean(self.predict(X) == np.array(y))

    # ── sklearn-style aliases ─────────────────────────────────────────
    @property
    def weights_(self):
        return self.weights

    @property
    def bias_(self):
        return self.bias
