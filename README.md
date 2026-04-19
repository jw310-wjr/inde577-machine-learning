# INDE 577 / CMOR 438 — Data Science & Machine Learning

**Author:** Jingru Wu  
**Institution:** Rice University  
**Course:** INDE 577 / CMOR 438 — Data Science & Machine Learning (Spring 2026)  
**Instructor:** Randy R. Davila, PhD  

---

## Overview

This repository is the final project for **INDE 577 / CMOR 438**. It contains:

1. **`rice_ml`** — A custom machine learning library built entirely from scratch using NumPy
2. **Jupyter Notebooks** — Educational, step-by-step demonstrations of every algorithm
3. **Unit Tests** — Comprehensive `pytest` test suite with full coverage
4. **GitHub Actions CI** — Automated test runs on every push

The core philosophy: implement every algorithm **from first principles** to build genuine
understanding of the mathematics, not just API familiarity. Every class follows a
scikit-learn-style API (`fit` / `predict` / `score`) for consistency.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions — auto pytest on push
│
├── src/
│   └── rice_ml/
│       ├── __init__.py
│       ├── supervised_learning/
│       │   ├── linear_regression.py    # OLS & Gradient Descent
│       │   ├── logistic_regression.py  # Sigmoid + BCE loss
│       │   ├── knn.py                  # Classifier & Regressor
│       │   ├── perceptron.py           # Single-layer, step activation
│       │   ├── mlp.py                  # Backprop + Adam/SGD
│       │   ├── decision_tree.py        # CART (Gini / Entropy / MSE)
│       │   ├── random_forest.py        # Bagging ensemble
│       │   ├── gradient_boosting.py    # Boosting ensemble
│       │   ├── svm.py                  # Soft-margin SVM (hinge loss + SGD)
│       │   └── naive_bayes.py          # Gaussian Naïve Bayes
│       │
│       ├── unsupervised_learning/
│       │   ├── kmeans.py               # K-Means++ clustering
│       │   ├── dbscan.py               # Density-based clustering
│       │   ├── pca.py                  # Eigendecomposition PCA
│       │   └── hierarchical.py         # Agglomerative clustering
│       │
│       └── processing/
│           ├── preprocessing.py        # StandardScaler, MinMaxScaler, train_test_split
│           └── metrics.py              # accuracy, MSE, RMSE, R², F1, confusion matrix
│
├── notebooks/
│   ├── supervised_learning/
│   │   ├── 01_linear_regression.ipynb
│   │   ├── 02_logistic_regression.ipynb
│   │   ├── 03_knn.ipynb
│   │   ├── 04_perceptron.ipynb
│   │   ├── 05_mlp.ipynb
│   │   ├── 06_decision_tree.ipynb
│   │   ├── 07_random_forest.ipynb
│   │   ├── 08_gradient_boosting.ipynb
│   │   ├── 09_svm.ipynb
│   │   └── 10_naive_bayes.ipynb
│   └── unsupervised_learning/
│       ├── 11_kmeans.ipynb
│       ├── 12_dbscan.ipynb
│       ├── 13_pca.ipynb
│       └── 14_hierarchical.ipynb
│
├── tests/
│   └── unit/
│       ├── test_linear_regression.py
│       ├── test_logistic_regression.py
│       ├── test_knn.py
│       ├── test_perceptron.py
│       ├── test_decision_tree.py
│       ├── test_random_forest.py
│       ├── test_gradient_boosting.py
│       ├── test_svm.py
│       ├── test_naive_bayes.py
│       ├── test_mlp.py
│       ├── test_kmeans.py
│       ├── test_dbscan.py
│       ├── test_pca.py
│       ├── test_hierarchical.py
│       ├── test_preprocessing.py
│       └── test_metrics.py
│
├── README.md
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## `rice_ml` Package

### Supervised Learning

| Algorithm | Module | Key Features |
|-----------|--------|-------------|
| **Linear Regression** | `linear_regression.py` | OLS closed-form + gradient descent; RMSE, R² |
| **Logistic Regression** | `logistic_regression.py` | Sigmoid + binary cross-entropy; L2 regularization |
| **K-Nearest Neighbors** | `knn.py` | Euclidean / Manhattan / Minkowski; classifier & regressor |
| **Perceptron** | `perceptron.py` | Online learning; step activation; convergence theorem |
| **MLP** | `mlp.py` | Backpropagation; ReLU/Sigmoid/Tanh; Adam & SGD; L2 reg |
| **Decision Tree** | `decision_tree.py` | CART; Gini impurity & entropy; variance reduction |
| **Random Forest** | `random_forest.py` | Bagging; random feature subsets; classifier & regressor |
| **Gradient Boosting** | `gradient_boosting.py` | Sequential residual fitting; log-loss & MSE |
| **SVM** | `svm.py` | Soft-margin; hinge loss + L2; SGD |
| **Naïve Bayes** | `naive_bayes.py` | Gaussian likelihood; multi-class; log-posterior |

### Unsupervised Learning

| Algorithm | Module | Key Features |
|-----------|--------|-------------|
| **K-Means** | `kmeans.py` | k-means++ init; inertia; convergence detection |
| **DBSCAN** | `dbscan.py` | Arbitrary shapes; noise detection; core points |
| **PCA** | `pca.py` | Eigendecomposition; explained variance ratio; inverse transform |
| **Hierarchical** | `hierarchical.py` | Agglomerative; single / complete / average / Ward linkage |

### Processing & Utilities

| Module | Contents |
|--------|---------|
| `preprocessing.py` | `StandardScaler`, `MinMaxScaler`, `train_test_split` |
| `metrics.py` | `accuracy_score`, `mse`, `rmse`, `mae`, `r2_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `classification_report` |

---

## Installation

```bash
git clone <repo-url>
cd <repo-name>
pip install -e ".[dev]"
```

## Quick Start

```python
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.processing.preprocessing import StandardScaler, train_test_split
from rice_ml.processing.metrics import r2_score, root_mean_squared_error

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train
model = LinearRegression(method="ols")
model.fit(X_train_s, y_train)

# Evaluate
print("R²  :", model.r2_score(X_test_s, y_test))
print("RMSE:", model.rmse(X_test_s, y_test))
```

---

## Testing

```bash
pytest
```

**113 tests** across 16 test files — one file per algorithm/module:

| File | Algorithm covered |
|------|------------------|
| `test_linear_regression.py` | Linear Regression |
| `test_logistic_regression.py` | Logistic Regression |
| `test_knn.py` | K-Nearest Neighbors |
| `test_perceptron.py` | Perceptron |
| `test_mlp.py` | Multi-Layer Perceptron |
| `test_decision_tree.py` | Decision Tree |
| `test_random_forest.py` | Random Forest |
| `test_gradient_boosting.py` | Gradient Boosting |
| `test_svm.py` | Support Vector Machine |
| `test_naive_bayes.py` | Gaussian Naïve Bayes |
| `test_kmeans.py` | K-Means |
| `test_dbscan.py` | DBSCAN |
| `test_pca.py` | PCA |
| `test_hierarchical.py` | Hierarchical Clustering |
| `test_preprocessing.py` | StandardScaler, MinMaxScaler, train_test_split |
| `test_metrics.py` | Evaluation metrics |

Tests cover numerical correctness, input/output shapes, edge cases, reproducibility, and comparisons against scikit-learn baselines.

---

## Notebooks

Each algorithm has a dedicated Jupyter notebook that covers:

1. **Mathematical intuition** — key equations and derivations
2. **Data exploration** — visualizations, distributions, correlations
3. **Preprocessing** — scaling, encoding, train/test split
4. **Training** — hyperparameter tuning, learning curves
5. **Evaluation** — metrics, confusion matrices, residual plots
6. **Comparison** — side-by-side with scikit-learn baseline

---

## License

MIT License — see [LICENSE](LICENSE)
