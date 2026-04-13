"""
linear_regression.py
Ordinary Least Squares (OLS) and Gradient Descent Linear Regression.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression implemented from scratch.

    Supports two fitting methods:
      - 'ols'              : Closed-form solution  w = (XᵀX)⁻¹ Xᵀy
      - 'gradient_descent' : Iterative gradient descent

    Parameters
    ----------
    learning_rate : float, step size for gradient descent (default 0.01)
    n_iterations  : int, number of gradient descent steps (default 1000)
    method        : str, 'ols' or 'gradient_descent' (default 'gradient_descent')

    Attributes
    ----------
    weights      : ndarray of shape (n_features,)
    bias         : float
    loss_history : list of MSE at each iteration (gradient_descent only)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, method="gradient_descent"):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """Fit the model to training data."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.loss_history = []

        if self.method == "ols":
            # Augment X with bias column
            X_b = np.c_[np.ones(n_samples), X]
            params = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = params[0]
            self.weights = params[1:]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            for _ in range(self.n_iterations):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                dw = (2 / n_samples) * X.T @ error
                db = (2 / n_samples) * np.sum(error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                self.loss_history.append(np.mean(error ** 2))
        return self

    def predict(self, X):
        """Predict continuous target values."""
        return np.array(X, dtype=float) @ self.weights + self.bias

    def mse(self, X, y):
        """Mean Squared Error on a dataset."""
        return np.mean((np.array(y) - self.predict(X)) ** 2)

    def rmse(self, X, y):
        """Root Mean Squared Error on a dataset."""
        return np.sqrt(self.mse(X, y))

    def r2_score(self, X, y):
        """
        Coefficient of determination R².
        R² = 1 - SS_res / SS_tot
        """
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
