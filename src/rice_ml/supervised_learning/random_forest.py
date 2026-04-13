"""
random_forest.py
Random Forest — Bagging ensemble of Decision Trees.
"""

import numpy as np
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Trains `n_estimators` decision trees on bootstrap samples, using a random
    subset of features at each split. Final prediction = majority vote.

    Parameters
    ----------
    n_estimators      : int   (default 100)
    max_depth         : int or None (default None)
    min_samples_split : int   (default 2)
    max_features      : 'sqrt' | 'log2' | int (default 'sqrt')
    random_state      : int or None
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices_ = []

    def _n_features(self, n_total):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_total)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_total)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_total)
        return n_total

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y)
        n_samples, n_total = X.shape
        rng = np.random.RandomState(self.random_state)
        self.trees, self.feature_indices_ = [], []

        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            feat_idx = rng.choice(n_total, self._n_features(n_total), replace=False)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X[boot_idx][:, feat_idx], y[boot_idx])
            self.trees.append(tree)
            self.feature_indices_.append(feat_idx)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        # Shape: (n_estimators, n_samples)
        preds = np.array([
            t.predict(X[:, fi]) for t, fi in zip(self.trees, self.feature_indices_)
        ])
        # Majority vote per sample
        return np.array([
            np.bincount(preds[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


class RandomForestRegressor:
    """
    Random Forest Regressor.

    Final prediction = mean of all tree predictions.

    Parameters
    ----------
    n_estimators      : int   (default 100)
    max_depth         : int or None (default None)
    min_samples_split : int   (default 2)
    max_features      : 'sqrt' | 'log2' | int (default 'sqrt')
    random_state      : int or None
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices_ = []

    def _n_features(self, n_total):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_total)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_total)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_total)
        return n_total

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples, n_total = X.shape
        rng = np.random.RandomState(self.random_state)
        self.trees, self.feature_indices_ = [], []

        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            feat_idx = rng.choice(n_total, self._n_features(n_total), replace=False)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X[boot_idx][:, feat_idx], y[boot_idx])
            self.trees.append(tree)
            self.feature_indices_.append(feat_idx)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.array([
            t.predict(X[:, fi]) for t, fi in zip(self.trees, self.feature_indices_)
        ])
        return np.mean(preds, axis=0)

    def score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
