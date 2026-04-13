"""
decision_tree.py
CART Decision Tree — Classifier (Gini / Entropy) and Regressor (Variance reduction).
"""

import numpy as np


# ─────────────────────────── Tree Node ──────────────────────────────

class _Node:
    """Internal node or leaf of a decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # feature index for split
        self.threshold = threshold  # split threshold
        self.left = left            # left subtree  (x <= threshold)
        self.right = right          # right subtree (x >  threshold)
        self.value = value          # leaf value (not None ⟹ leaf)

    def is_leaf(self):
        return self.value is not None


# ─────────────────────────── Classifier ─────────────────────────────

class DecisionTreeClassifier:
    """
    Decision Tree Classifier using CART (Classification And Regression Trees).

    Splitting criterion: Gini impurity or Information Gain (entropy).

    Parameters
    ----------
    max_depth         : int or None (default None — grow until pure)
    min_samples_split : int, minimum samples required to split (default 2)
    criterion         : str, 'gini' | 'entropy' (default 'gini')
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    # ── impurity helpers ─────────────────────────────────────────────

    @staticmethod
    def _gini(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    @staticmethod
    def _entropy(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-10))

    def _impurity(self, y):
        return self._gini(y) if self.criterion == "gini" else self._entropy(y)

    def _information_gain(self, y, y_left, y_right):
        n = len(y)
        weighted = (len(y_left) / n) * self._impurity(y_left) + \
                   (len(y_right) / n) * self._impurity(y_right)
        return self._impurity(y) - weighted

    # ── tree construction ────────────────────────────────────────────

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -np.inf, None, None
        for feat in range(X.shape[1]):
            for thresh in np.unique(X[:, feat]):
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = self._information_gain(y, left, right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
                len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(value=int(np.bincount(y.astype(int)).argmax()))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return _Node(value=int(np.bincount(y.astype(int)).argmax()))

        mask = X[:, feat] <= thresh
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    # ── public API ───────────────────────────────────────────────────

    def fit(self, X, y):
        self.root = self._build(np.array(X, dtype=float), np.array(y), 0)
        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in np.array(X, dtype=float)])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# ─────────────────────────── Regressor ──────────────────────────────

class DecisionTreeRegressor:
    """
    Decision Tree Regressor using CART with variance (MSE) reduction.

    Parameters
    ----------
    max_depth         : int or None (default None)
    min_samples_split : int (default 2)
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    @staticmethod
    def _variance_reduction(y, y_left, y_right):
        n = len(y)
        return np.var(y) - (len(y_left) / n) * np.var(y_left) - \
               (len(y_right) / n) * np.var(y_right)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -np.inf, None, None
        for feat in range(X.shape[1]):
            for thresh in np.unique(X[:, feat]):
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = self._variance_reduction(y, left, right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if (self.max_depth is not None and depth >= self.max_depth) or \
                len(y) < self.min_samples_split:
            return _Node(value=float(np.mean(y)))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return _Node(value=float(np.mean(y)))

        mask = X[:, feat] <= thresh
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(np.array(X, dtype=float), np.array(y, dtype=float), 0)
        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in np.array(X, dtype=float)])

    def score(self, X, y):
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
