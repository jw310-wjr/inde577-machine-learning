"""Tests for KFold, StratifiedKFold, and cross_val_score."""
import numpy as np
import pytest
from rice_ml.processing.cross_validation import KFold, StratifiedKFold, cross_val_score
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


# ── KFold ─────────────────────────────────────────────────────────────────────

def test_kfold_n_splits():
    kf = KFold(n_splits=5)
    X = np.arange(100).reshape(50, 2)
    splits = list(kf.split(X))
    assert len(splits) == 5


def test_kfold_no_overlap():
    """Val indices across folds must be disjoint and cover all samples."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    X = np.arange(100).reshape(100, 1)
    all_val = np.concatenate([v for _, v in kf.split(X)])
    assert len(np.unique(all_val)) == 100


def test_kfold_train_val_partition():
    """train + val indices must equal the full index set in each fold."""
    kf = KFold(n_splits=4)
    X = np.arange(60).reshape(60, 1)
    for tr, val in kf.split(X):
        assert len(np.union1d(tr, val)) == 60
        assert len(np.intersect1d(tr, val)) == 0


def test_kfold_val_sizes_balanced():
    """Fold sizes should differ by at most 1 when n not divisible by k."""
    kf = KFold(n_splits=4)
    X = np.arange(101).reshape(101, 1)
    sizes = [len(v) for _, v in kf.split(X)]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_shuffle_reproducible():
    kf1 = KFold(n_splits=3, shuffle=True, random_state=42)
    kf2 = KFold(n_splits=3, shuffle=True, random_state=42)
    X = np.arange(30).reshape(30, 1)
    for (tr1, v1), (tr2, v2) in zip(kf1.split(X), kf2.split(X)):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(v1, v2)


def test_kfold_invalid_n_splits():
    with pytest.raises(ValueError):
        KFold(n_splits=1)


# ── StratifiedKFold ───────────────────────────────────────────────────────────

def test_stratified_kfold_n_splits():
    skf = StratifiedKFold(n_splits=5)
    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 50 + [1] * 50)
    splits = list(skf.split(X, y))
    assert len(splits) == 5


def test_stratified_kfold_covers_all():
    skf = StratifiedKFold(n_splits=3)
    X = np.ones((90, 2))
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    all_val = np.concatenate([v for _, v in skf.split(X, y)])
    assert len(np.unique(all_val)) == 90


# ── cross_val_score ───────────────────────────────────────────────────────────

def test_cross_val_score_r2():
    np.random.seed(0)
    X = np.random.randn(100, 3)
    y = X @ [1.0, -2.0, 0.5] + np.random.randn(100) * 0.2
    scores = cross_val_score(LinearRegression(method='ols'), X, y, cv=4, scoring='r2')
    assert len(scores) == 4
    assert all(s > 0.8 for s in scores)


def test_cross_val_score_accuracy():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    scores = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='accuracy')
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_cross_val_score_neg_mse():
    np.random.seed(2)
    X = np.random.randn(80, 2)
    y = X @ [2.0, -1.0] + np.random.randn(80) * 0.1
    scores = cross_val_score(LinearRegression(method='ols'), X, y, cv=4, scoring='neg_mse')
    assert all(s <= 0 for s in scores)


def test_cross_val_score_neg_rmse():
    np.random.seed(3)
    X = np.random.randn(80, 2)
    y = X @ [1.0, 1.0]
    scores = cross_val_score(LinearRegression(method='ols'), X, y, cv=3, scoring='neg_rmse')
    assert all(s <= 0 for s in scores)
    # neg_rmse should be between neg_mse and 0 for unit-scale data
    mse_scores = cross_val_score(LinearRegression(method='ols'), X, y, cv=3, scoring='neg_mse')
    assert all(r >= m for r, m in zip(scores, mse_scores))


def test_cross_val_score_custom_cv():
    np.random.seed(4)
    X = np.random.randn(60, 2)
    y = X[:, 0]
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_val_score(LinearRegression(method='ols'), X, y, cv=kf, scoring='r2')
    assert len(scores) == 3


def test_cross_val_score_invalid_scoring():
    X = np.random.randn(20, 2)
    y = np.random.randn(20)
    with pytest.raises(ValueError):
        cross_val_score(LinearRegression(), X, y, scoring='bad_metric')


def test_cross_val_score_does_not_mutate_estimator():
    """cross_val_score should not modify the original estimator."""
    np.random.seed(5)
    X, y = np.random.randn(60, 3), np.random.randn(60)
    model = LinearRegression(method='ols')
    assert model.weights is None
    cross_val_score(model, X, y, cv=3, scoring='r2')
    assert model.weights is None  # original untouched
