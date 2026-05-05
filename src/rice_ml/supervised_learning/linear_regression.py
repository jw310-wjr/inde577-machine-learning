"""
Linear Regression -- OLS, Gradient Descent, Ridge, and Lasso.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression supporting OLS and Gradient Descent.

    Parameters
    ----------
    method       : 'ols' | 'gradient_descent'
    learning_rate: float (gradient_descent only)
    n_iterations : int   (gradient_descent only)
    """

    def __init__(self, method="ols", learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        if self.method == "ols":
            X_b = np.c_[np.ones(n_samples), X]
            params = np.linalg.pinv(X_b) @ y
            self.bias = params[0]
            self.weights = params[1:]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            self.loss_history = []
            for _ in range(self.n_iterations):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                self.weights -= self.learning_rate * (X.T @ error) / n_samples
                self.bias -= self.learning_rate * np.mean(error)
                self.loss_history.append(np.mean(error ** 2))
        return self

    def predict(self, X):
        return np.array(X, dtype=float) @ self.weights + self.bias

    def mse(self, X, y):
        return np.mean((np.array(y) - self.predict(X)) ** 2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    def r2_score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


class RidgeRegression:
    """
    Ridge Regression (L2 regularisation).

    Minimises: ||y - Xw||^2 + alpha * ||w||^2

    Closed-form solution: w = (X^T X + alpha*I)^{-1} X^T y
    The bias term is NOT regularised (standard convention).

    Parameters
    ----------
    alpha : float
        Regularisation strength (default 1.0).  alpha=0 recovers OLS.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones(n_samples), X]
        reg = self.alpha * np.eye(n_features + 1)
        reg[0, 0] = 0.0          # do not penalise bias
        params = np.linalg.pinv(X_b.T @ X_b + reg) @ X_b.T @ y
        self.bias = params[0]
        self.weights = params[1:]
        return self

    def predict(self, X):
        return np.array(X, dtype=float) @ self.weights + self.bias

    def mse(self, X, y):
        return np.mean((np.array(y) - self.predict(X)) ** 2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    def r2_score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


class LassoRegression:
    """
    Lasso Regression (L1 regularisation) via coordinate descent.

    Minimises: (1/2n) ||y - Xw||^2 + alpha * ||w||_1

    The soft-thresholding operator drives small coefficients to exactly zero,
    performing implicit feature selection.

    Parameters
    ----------
    alpha        : float  Regularisation strength (default 1.0).
    n_iterations : int    Max coordinate-descent iterations (default 1000).
    tol          : float  Convergence tolerance on max coef change (default 1e-4).
    """

    def __init__(self, alpha=1.0, n_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None

    @staticmethod
    def _soft_threshold(rho, alpha):
        """Soft-thresholding operator S(rho, alpha)."""
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        return 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            w_old = self.weights.copy()
            # Update bias (no regularisation on intercept)
            r = y - X @ self.weights
            self.bias = np.mean(r)
            # Coordinate descent over each feature
            r = y - self.bias - X @ self.weights
            for j in range(n_features):
                r += X[:, j] * self.weights[j]           # restore j-th contribution
                rho = (X[:, j] @ r) / n_samples
                self.weights[j] = self._soft_threshold(rho, self.alpha)
                r -= X[:, j] * self.weights[j]           # subtract updated contribution
            if np.max(np.abs(self.weights - w_old)) < self.tol:
                break
        return self

    def predict(self, X):
        return np.array(X, dtype=float) @ self.weights + self.bias

    def mse(self, X, y):
        return np.mean((np.array(y) - self.predict(X)) ** 2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    def r2_score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
