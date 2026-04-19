"""
svm.py
Support Vector Machine — soft-margin SVM via stochastic gradient descent (hinge loss + L2).
"""

import numpy as np


class SVM:
    """
    Support Vector Machine Classifier (binary).

    Uses a soft-margin formulation with hinge loss and L2 regularization,
    optimised via stochastic gradient descent.

    Decision boundary: w·x + b = 0
    Labels must be {-1, 1}. If {0, 1} labels are passed, they are auto-converted.

    Parameters
    ----------
    C             : float, regularization strength — higher C = less regularization (default 1.0)
    learning_rate : float (default 0.001)
    n_iterations  : int   (default 1000)
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    # ── sklearn-style aliases ─────────────────────────────────────────
    @property
    def weights_(self):
        return self.weights

    @property
    def bias_(self):
        return self.bias

    @property
    def loss_history_(self):
        return self.loss_history

    @staticmethod
    def _to_pm1(y):
        """Convert {0, 1} labels to {-1, +1}."""
        y = np.array(y, dtype=float)
        if set(np.unique(y)).issubset({0.0, 1.0}):
            return 2 * y - 1
        return y

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = self._to_pm1(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            for i, xi in enumerate(X):
                margin = y[i] * (xi @ self.weights + self.bias)
                if margin >= 1:
                    # Correct, only regularization term
                    self.weights -= self.learning_rate * (2 / n_samples) * self.weights
                else:
                    # Hinge loss gradient
                    self.weights -= self.learning_rate * (
                        2 / n_samples * self.weights - self.C * y[i] * xi
                    )
                    self.bias -= self.learning_rate * (-self.C * y[i])

            # Epoch hinge loss
            margins = y * (X @ self.weights + self.bias)
            hinge = np.maximum(0.0, 1.0 - margins)
            loss = np.dot(self.weights, self.weights) / n_samples + self.C * np.mean(hinge)
            self.loss_history.append(loss)
        return self

    def predict(self, X):
        """Return predictions in {-1, +1}."""
        raw = np.array(X, dtype=float) @ self.weights + self.bias
        return np.where(raw >= 0, 1, -1).astype(int)

    def predict_binary(self, X):
        """Return predictions in {0, 1} for downstream compatibility."""
        return ((self.predict(X) + 1) // 2).astype(int)

    def score(self, X, y):
        """Accuracy, handling both {-1,1} and {0,1} label conventions."""
        y_internal = self._to_pm1(y)
        return np.mean(self.predict(X) == y_internal)
