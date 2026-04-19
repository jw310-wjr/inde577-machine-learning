"""
gradient_boosting.py
Gradient Boosting — sequential ensemble of Decision Trees.
"""

import numpy as np
from .decision_tree import DecisionTreeRegressor


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier (binary) using log-loss.

    Each iteration fits a regression tree to the negative gradient
    (residuals) of the log-loss with respect to the current prediction.

    Parameters
    ----------
    n_estimators      : int   (default 100)
    learning_rate     : float (default 0.1)
    max_depth         : int   (default 3)
    min_samples_split : int   (default 2)
    random_state      : int or None (ignored, for API compatibility)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.initial_pred = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        # Initialise with log-odds of the mean
        p0 = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
        self.initial_pred = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.initial_pred)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - self._sigmoid(F)           # negative gradient of log-loss
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        F = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return self._sigmoid(F)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor using MSE (L2) loss.

    Parameters
    ----------
    n_estimators      : int   (default 100)
    learning_rate     : float (default 0.1)
    max_depth         : int   (default 3)
    min_samples_split : int   (default 2)
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        self.initial_pred = np.mean(y)
        F = np.full(len(y), self.initial_pred)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        F = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
