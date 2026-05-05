"""
cross_validation.py
K-Fold cross-validation utilities implemented from scratch.
"""

import numpy as np


class KFold:
    """
    K-Fold Cross-Validator.

    Splits data into `n_splits` consecutive folds (with optional shuffling).
    Each fold is used once as a validation set while the remaining k-1 folds
    form the training set.

    Parameters
    ----------
    n_splits     : int   Number of folds (default 5). Must be >= 2.
    shuffle      : bool  Whether to shuffle before splitting (default False).
    random_state : int or None  Seed for the RNG when shuffle=True.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        Generate (train_indices, val_indices) for each fold.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
        y : ignored (kept for API compatibility)

        Yields
        ------
        train_idx : ndarray  Indices for the training set.
        val_idx   : ndarray  Indices for the validation set.
        """
        X = np.asarray(X)
        n = len(X)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1  # distribute remainder

        current = 0
        for size in fold_sizes:
            val_idx = indices[current : current + size]
            train_idx = np.concatenate([indices[:current], indices[current + size:]])
            yield train_idx, val_idx
            current += size

    def get_n_splits(self):
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold Cross-Validator.

    Like KFold but preserves the percentage of samples for each class
    in each fold — important for imbalanced datasets.

    Parameters
    ----------
    n_splits     : int
    shuffle      : bool
    random_state : int or None
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """
        Generate stratified (train_idx, val_idx) pairs.

        Parameters
        ----------
        X : array-like
        y : array-like of class labels

        Yields
        ------
        train_idx, val_idx : ndarrays
        """
        y = np.asarray(y)
        classes, y_idx = np.unique(y, return_inverse=True)
        rng = np.random.RandomState(self.random_state)

        # For each class, get sorted indices (optionally shuffled within class)
        class_indices = []
        for c in range(len(classes)):
            c_idx = np.where(y_idx == c)[0]
            if self.shuffle:
                c_idx = rng.permutation(c_idx)
            class_indices.append(c_idx)

        # Assign each class's indices to folds in round-robin order
        fold_indices = [[] for _ in range(self.n_splits)]
        for c_idx in class_indices:
            for fold, i in enumerate(range(0, len(c_idx), 1)):
                fold_indices[fold % self.n_splits].append(c_idx[i])

        all_indices = np.arange(len(y))
        for k in range(self.n_splits):
            val_idx = np.array(fold_indices[k])
            train_idx = np.setdiff1d(all_indices, val_idx)
            yield train_idx, val_idx

    def get_n_splits(self):
        return self.n_splits


def cross_val_score(estimator, X, y, cv=5, scoring="accuracy"):
    """
    Evaluate an estimator's performance via K-Fold cross-validation.

    The estimator must implement `fit(X, y)` and `score(X, y)` (or
    `predict(X, y)` for custom scoring — see `scoring` parameter).

    Parameters
    ----------
    estimator : object
        A machine-learning object with `fit` and `score` methods.
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    cv : int or KFold instance
        Number of folds (int) or a pre-built cross-validator.  Default 5.
    scoring : str
        Metric name.  Supported:
        - ``'accuracy'``  — fraction of correct predictions (classification).
        - ``'r2'``        — coefficient of determination R² (regression).
        - ``'neg_mse'``   — negative mean squared error (regression).
        - ``'neg_rmse'``  — negative root mean squared error (regression).

    Returns
    -------
    scores : ndarray, shape (n_splits,)
        Array of scores for each fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if isinstance(cv, int):
        cv = KFold(n_splits=cv)

    SCORING = {
        "accuracy":  lambda est, Xt, yt: np.mean(est.predict(Xt) == yt),
        "r2":        lambda est, Xt, yt: _r2(est.predict(Xt), yt),
        "neg_mse":   lambda est, Xt, yt: -np.mean((est.predict(Xt) - yt) ** 2),
        "neg_rmse":  lambda est, Xt, yt: -np.sqrt(np.mean((est.predict(Xt) - yt) ** 2)),
    }
    if scoring not in SCORING:
        raise ValueError(
            f"scoring='{scoring}' not recognised. "
            f"Choose from {list(SCORING.keys())}."
        )
    score_fn = SCORING[scoring]

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        import copy
        est = copy.deepcopy(estimator)
        est.fit(X_tr, y_tr)
        scores.append(score_fn(est, X_val, y_val))

    return np.array(scores)


def _r2(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
