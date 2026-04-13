"""
rice_ml: A custom machine learning library for INDE 577 / CMOR 438.

Built from scratch using NumPy to emphasize algorithmic transparency.
"""

from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor
from rice_ml.supervised_learning.perceptron import Perceptron
from rice_ml.supervised_learning.mlp import MLP
from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from rice_ml.supervised_learning.random_forest import RandomForestClassifier, RandomForestRegressor
from rice_ml.supervised_learning.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from rice_ml.supervised_learning.svm import SVM
from rice_ml.supervised_learning.naive_bayes import GaussianNaiveBayes
from rice_ml.unsupervised_learning.kmeans import KMeans
from rice_ml.unsupervised_learning.dbscan import DBSCAN
from rice_ml.unsupervised_learning.pca import PCA
from rice_ml.unsupervised_learning.hierarchical import HierarchicalClustering
from rice_ml.processing.preprocessing import StandardScaler, MinMaxScaler, train_test_split
from rice_ml.processing.metrics import (
    accuracy_score, mean_squared_error, root_mean_squared_error,
    mean_absolute_error, r2_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report,
)

__version__ = "0.1.0"
__author__ = "Jingru Wu"
