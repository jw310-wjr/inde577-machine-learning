# Supervised Learning Notebooks

This directory contains step-by-step Jupyter Notebook demonstrations for all **10 supervised learning algorithms** implemented in the `rice_ml` package.

---

## Notebooks

| # | Notebook | Algorithm | Dataset(s) |
|---|----------|-----------|-----------|
| 01 | [01_linear_regression.ipynb](01_linear_regression.ipynb) | Linear Regression (OLS & Gradient Descent) | California Housing, synthetic |
| 02 | [02_logistic_regression.ipynb](02_logistic_regression.ipynb) | Logistic Regression (Sigmoid + BCE) | Breast Cancer, synthetic |
| 03 | [03_knn.ipynb](03_knn.ipynb) | K-Nearest Neighbors | Iris, synthetic |
| 04 | [04_perceptron.ipynb](04_perceptron.ipynb) | Perceptron (single-layer) | Breast Cancer, synthetic |
| 05 | [05_mlp.ipynb](05_mlp.ipynb) | Multi-Layer Perceptron (Backprop + Adam) | Breast Cancer, synthetic XOR |
| 06 | [06_decision_tree.ipynb](06_decision_tree.ipynb) | Decision Tree (CART — Gini / Entropy / MSE) | Iris, Breast Cancer, synthetic |
| 07 | [07_random_forest.ipynb](07_random_forest.ipynb) | Random Forest (Bagging Ensemble) | Breast Cancer, synthetic |
| 08 | [08_gradient_boosting.ipynb](08_gradient_boosting.ipynb) | Gradient Boosting (Boosting Ensemble) | Breast Cancer, synthetic regression |
| 09 | [09_svm.ipynb](09_svm.ipynb) | Support Vector Machine (Soft-Margin, Hinge Loss + SGD) | Breast Cancer, synthetic |
| 10 | [10_naive_bayes.ipynb](10_naive_bayes.ipynb) | Gaussian Naïve Bayes | Iris, synthetic |

---

## Structure

```
supervised_learning/
├── 01_linear_regression.ipynb
├── 02_logistic_regression.ipynb
├── 03_knn.ipynb
├── 04_perceptron.ipynb
├── 05_mlp.ipynb
├── 06_decision_tree.ipynb
├── 07_random_forest.ipynb
├── 08_gradient_boosting.ipynb
├── 09_svm.ipynb
├── 10_naive_bayes.ipynb
├── figures/                  ← saved plots from notebook runs
└── README.md
```

---

## Key Concepts Covered

Each notebook includes:
- **Mathematical background** — derivation of the core update rules and objective functions
- **From-scratch implementation demo** using the custom `rice_ml` package
- **Comparison with scikit-learn** where applicable
- **Visualizations** — decision boundaries, loss curves, feature importance, etc.
- **Hyperparameter analysis** — effect of key parameters on model performance
