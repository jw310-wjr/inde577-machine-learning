"""
metrics.py
Evaluation metrics for classification and regression tasks.
"""

import numpy as np


# ─────────────────────────── Regression ────────────────────────────

def mean_squared_error(y_true, y_pred):
    """Mean Squared Error (MSE)."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error (MAE)."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def r2_score(y_true, y_pred):
    """
    Coefficient of determination R².
    R² = 1 - SS_res / SS_tot
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ─────────────────────────── Classification ─────────────────────────

def accuracy_score(y_true, y_pred):
    """Fraction of correctly classified samples."""
    return np.mean(np.array(y_true) == np.array(y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        cm[i, j] = number of samples with true class i predicted as class j.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    idx_map = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx_map[t], idx_map[p]] += 1
    return cm


def precision_score(y_true, y_pred, pos_label=1):
    """Precision = TP / (TP + FP)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, pos_label=1):
    """Recall = TP / (TP + FN)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """F1 = 2 * precision * recall / (precision + recall)."""
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred):
    """
    Build a text report showing per-class precision, recall, F1, and support.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(y_true)
    header = f"{'Class':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    lines = [header, "-" * 55]
    for c in classes:
        p = precision_score(y_true, y_pred, c)
        r = recall_score(y_true, y_pred, c)
        f1 = f1_score(y_true, y_pred, c)
        support = int(np.sum(y_true == c))
        lines.append(f"{str(c):>10} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}")
    lines.append("-" * 55)
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {acc:>10.4f} {len(y_true):>10}")
    return "\n".join(lines)
