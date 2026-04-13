"""
preprocessing.py
Data preprocessing utilities: scalers and train/test split.
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Formula: z = (x - mean) / std

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
    std_  : ndarray of shape (n_features,)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute mean and std from training data."""
        X = np.array(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X):
        """Apply standardization."""
        return (np.array(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the standardization."""
        return np.array(X_scaled, dtype=float) * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1]).

    Formula: x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min

    Parameters
    ----------
    feature_range : tuple (min, max), default (0, 1)
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """Compute per-feature min/max from training data."""
        X = np.array(X, dtype=float)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        """Apply min-max scaling."""
        X = np.array(X, dtype=float)
        scale = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        X_std = (X - self.min_) / scale
        range_min, range_max = self.feature_range
        return X_std * (range_max - range_min) + range_min

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling."""
        range_min, range_max = self.feature_range
        X_std = (np.array(X_scaled, dtype=float) - range_min) / (range_max - range_min)
        return X_std * (self.max_ - self.min_) + self.min_


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X           : array-like of shape (n_samples, n_features)
    y           : array-like of shape (n_samples,)
    test_size   : float, fraction of data for test set (default 0.2)
    random_state: int or None, seed for reproducibility
    shuffle     : bool, whether to shuffle before splitting

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = np.array(X), np.array(y)
    n_samples = len(X)
    n_test = max(1, int(n_samples * test_size))

    if shuffle:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
