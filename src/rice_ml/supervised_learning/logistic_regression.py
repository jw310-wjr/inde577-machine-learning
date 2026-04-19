"""
logistic_regression.py
Binary Logistic Regression using gradient descent.
"""

import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression with optional L2 regularization.

    Uses the sigmoid activation and binary cross-entropy loss, optimized
    via full-batch gradient descent.

    Parameters
    ----------
    learning_rate : float  (default 0.01)
    n_iterations  : int    (default 1000)
    lambda_reg    : float, L2 regularization strength (default 0.0)

    Attributes
    ----------
    weights      : ndarray of shape (n_features,)
    bias         : float
    loss_history : list of binary cross-entropy at each iteration
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # ── public API ────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit via gradient descent."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._sigmoid(X @ self.weights + self.bias)

            dw = (1 / n_samples) * X.T @ (y_pred - y) + (self.lambda_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Binary cross-entropy loss
            y_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
            loss = -np.mean(y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """Return probability estimates P(y=1|X)."""
        return self._sigmoid(np.array(X, dtype=float) @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        """Return binary class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """Accuracy on dataset (X, y)."""
        return np.mean(self.predict(X) == np.array(y))

    # ── sklearn-style trailing-underscore aliases ─────────────────────
    @property
    def weights_(self):
        return self.weights

    @property
    def bias_(self):
        return self.bias

    @property
    def loss_history_(self):
        return self.loss_history
