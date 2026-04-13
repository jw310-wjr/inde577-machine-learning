"""
pca.py
Principal Component Analysis via eigenvalue decomposition of the covariance matrix.
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA).

    Finds the directions (principal components) that capture the most variance
    in the data, then projects data onto a lower-dimensional subspace.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If None, keep all. (default None)

    Attributes
    ----------
    components_              : ndarray of shape (n_components, n_features)
                               Principal axes (rows = eigenvectors, sorted by variance).
    explained_variance_      : ndarray of shape (n_components,)
                               Eigenvalues (variance along each component).
    explained_variance_ratio_: ndarray of shape (n_components,)
                               Fraction of total variance explained.
    mean_                    : ndarray of shape (n_features,)
                               Per-feature mean of the training data.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """Compute principal components from training data X."""
        X = np.array(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        X_c = X - self.mean_

        # Covariance matrix and eigendecomposition
        cov = np.cov(X_c, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)   # guaranteed real, symmetric

        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        k = self.n_components if self.n_components else X.shape[1]
        self.components_ = eigenvectors[:, :k].T              # shape (k, n_features)
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = eigenvalues[:k] / np.sum(eigenvalues)
        return self

    def transform(self, X):
        """Project X onto the principal components."""
        X_c = np.array(X, dtype=float) - self.mean_
        return X_c @ self.components_.T                      # shape (n_samples, k)

    def fit_transform(self, X):
        """Fit and project in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced):
        """Reconstruct data from the reduced representation."""
        return np.array(X_reduced, dtype=float) @ self.components_ + self.mean_
