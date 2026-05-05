# Unsupervised Learning Notebooks

This directory contains step-by-step Jupyter Notebook demonstrations for all **5 unsupervised learning algorithms** implemented in the `rice_ml` package.

---

## Notebooks

| # | Notebook | Algorithm | Dataset(s) |
|---|----------|-----------|-----------|
| 11 | [11_kmeans.ipynb](11_kmeans.ipynb) | K-Means Clustering (K-Means++ initialization) | Iris, synthetic blobs |
| 12 | [12_dbscan.ipynb](12_dbscan.ipynb) | DBSCAN (Density-Based Spatial Clustering) | Synthetic moons/rings, noisy data |
| 13 | [13_pca.ipynb](13_pca.ipynb) | Principal Component Analysis (Eigendecomposition) | Iris, Digits |
| 14 | [14_hierarchical.ipynb](14_hierarchical.ipynb) | Hierarchical Clustering (Agglomerative — single/complete/average linkage) | Iris, synthetic |
| 15 | [15_tsne.ipynb](15_tsne.ipynb) | t-SNE (van der Maaten & Hinton 2008) | Iris, Digits, Breast Cancer |
| 17 | [17_autoencoder.ipynb](17_autoencoder.ipynb) | Autoencoder (encoder–decoder, MSE loss) | Iris, Digits — compression & anomaly detection |

---

## Structure

```
unsupervised_learning/
├── 11_kmeans.ipynb
├── 12_dbscan.ipynb
├── 13_pca.ipynb
├── 14_hierarchical.ipynb
├── 15_tsne.ipynb
├── figures/                  ← saved plots from notebook runs
└── README.md
```

---

## Key Concepts Covered

Each notebook includes:
- **Mathematical background** — distance metrics, objective functions, decompositions
- **From-scratch implementation demo** using the custom `rice_ml` package
- **Comparison with scikit-learn** where applicable
- **Visualizations** — cluster assignments, dendrograms, PCA biplots, ε-neighborhood plots
- **Hyperparameter analysis** — effect of k, ε, min_samples, linkage criterion, etc.
