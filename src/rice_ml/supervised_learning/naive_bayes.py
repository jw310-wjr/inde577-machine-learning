"""
naive_bayes.py
Gaussian Naïve Bayes classifier — assumes Gaussian feature distributions per class.
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naïve Bayes Classifier.

    Assumes that each feature follows a Gaussian distribution within each
    class and that features are conditionally independent given the class.

    log P(y=k | x) ∝ log P(y=k) + Σ log P(xⱼ | y=k)

    Parameters
    ----------
    var_smoothing : float, variance smoothing to avoid zero variance (default 1e-9)

    Attributes
    ----------
    classes_       : ndarray of unique class labels
    class_prior_   : ndarray of shape (n_classes,) — P(y=k)
    theta_         : ndarray of shape (n_classes, n_features) — per-class means
    sigma_         : ndarray of shape (n_classes, n_features) — per-class variances
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.sigma_ = None

    def fit(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_prior_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[i] = len(X_c) / len(X)
            self.theta_[i] = np.mean(X_c, axis=0)
            self.sigma_[i] = np.var(X_c, axis=0) + self.var_smoothing
        return self

    def _log_likelihood(self, X, class_idx):
        """Log-likelihood under the Gaussian assumption for class class_idx."""
        mean = self.theta_[class_idx]
        var = self.sigma_[class_idx]
        return np.sum(
            -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2 / var),
            axis=1,
        )

    def predict_log_proba(self, X):
        """Return log posterior for each class."""
        X = np.array(X, dtype=float)
        log_proba = np.zeros((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            log_proba[:, i] = np.log(self.class_prior_[i]) + self._log_likelihood(X, i)
        return log_proba

    def predict_proba(self, X):
        """Return normalized posterior probabilities."""
        log_proba = self.predict_log_proba(X)
        log_proba -= np.max(log_proba, axis=1, keepdims=True)  # numerical stability
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Return predicted class labels."""
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y):
        """Accuracy on dataset (X, y)."""
        return np.mean(self.predict(X) == np.array(y))
