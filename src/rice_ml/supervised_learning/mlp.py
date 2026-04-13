"""
mlp.py
Multi-Layer Perceptron (Feedforward Neural Network) with backpropagation.
Supports SGD and Adam optimizers, multiple activations, and L2 regularization.
"""

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron for classification or regression.

    Architecture: Input → [Hidden layers] → Output (1 neuron)

    Parameters
    ----------
    hidden_layers      : tuple of ints, neurons per hidden layer (default (64, 32))
    activation         : str, hidden activation — 'relu'|'sigmoid'|'tanh' (default 'relu')
    learning_rate      : float (default 0.001)
    n_iterations       : int, number of full passes over data (default 1000)
    optimizer          : str, 'adam'|'sgd' (default 'adam')
    batch_size         : int, mini-batch size (default 32)
    lambda_reg         : float, L2 regularization strength (default 0.0)
    task               : str, 'classification'|'regression' (default 'classification')
    random_state       : int or None
    """

    def __init__(self, hidden_layers=(64, 32), activation="relu",
                 learning_rate=0.001, n_iterations=1000, optimizer="adam",
                 batch_size=32, lambda_reg=0.0, task="classification",
                 random_state=None):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.task = task
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.loss_history = []

    # ── activations ───────────────────────────────────────────────────

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_deriv(a):
        return (a > 0).astype(float)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_deriv(a):
        return a * (1.0 - a)

    @staticmethod
    def _tanh_deriv(a):
        return 1.0 - a ** 2

    def _activate(self, z):
        if self.activation == "relu":
            return self._relu(z)
        if self.activation == "sigmoid":
            return self._sigmoid(z)
        if self.activation == "tanh":
            return np.tanh(z)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _activate_deriv(self, a):
        if self.activation == "relu":
            return self._relu_deriv(a)
        if self.activation == "sigmoid":
            return self._sigmoid_deriv(a)
        if self.activation == "tanh":
            return self._tanh_deriv(a)
        raise ValueError(f"Unknown activation: {self.activation}")

    # ── weight initialisation ─────────────────────────────────────────

    def _init_weights(self, layer_sizes):
        rng = np.random.RandomState(self.random_state)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i]) if self.activation == "relu" \
                else np.sqrt(1.0 / layer_sizes[i])
            W = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    # ── forward pass ──────────────────────────────────────────────────

    def _forward(self, X):
        """Return list of activations for every layer (including input)."""
        acts = [X]
        cur = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = cur @ W + b
            if i < len(self.weights) - 1:
                cur = self._activate(z)
            else:
                # Output layer: sigmoid for classification, linear for regression
                cur = self._sigmoid(z) if self.task == "classification" else z
            acts.append(cur)
        return acts

    # ── loss ──────────────────────────────────────────────────────────

    def _loss(self, y_pred, y_true):
        n = len(y_true)
        if self.task == "classification":
            y_clip = np.clip(y_pred, 1e-10, 1 - 1e-10)
            data_loss = -np.mean(y_true * np.log(y_clip) + (1 - y_true) * np.log(1 - y_clip))
        else:
            data_loss = np.mean((y_pred.ravel() - y_true) ** 2)
        reg = self.lambda_reg * sum(np.sum(W ** 2) for W in self.weights) / (2 * n)
        return data_loss + reg

    # ── backpropagation ───────────────────────────────────────────────

    def _backprop(self, acts, y):
        n = len(y)
        y_pred = acts[-1]

        # Output delta
        if self.task == "classification":
            delta = y_pred - y.reshape(-1, 1)        # sigmoid + BCE → clean gradient
        else:
            delta = (y_pred - y.reshape(-1, 1))       # linear output + MSE

        dWs, dbs = [], []
        for i in reversed(range(len(self.weights))):
            dW = acts[i].T @ delta / n + self.lambda_reg * self.weights[i] / n
            db = np.sum(delta, axis=0, keepdims=True) / n
            dWs.insert(0, dW)
            dbs.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T * self._activate_deriv(acts[i])
        return dWs, dbs

    # ── public API ────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self._init_weights(layer_sizes)
        self.loss_history = []

        rng = np.random.RandomState(self.random_state)

        # Adam state
        if self.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            mW = [np.zeros_like(W) for W in self.weights]
            vW = [np.zeros_like(W) for W in self.weights]
            mb = [np.zeros_like(b) for b in self.biases]
            vb = [np.zeros_like(b) for b in self.biases]
            t = 0

        for epoch in range(self.n_iterations):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = idx[start: start + self.batch_size]
                Xb, yb = X[batch], y[batch]
                acts = self._forward(Xb)
                dWs, dbs = self._backprop(acts, yb)

                if self.optimizer == "adam":
                    t += 1
                    for i in range(len(self.weights)):
                        mW[i] = beta1 * mW[i] + (1 - beta1) * dWs[i]
                        vW[i] = beta2 * vW[i] + (1 - beta2) * dWs[i] ** 2
                        mb[i] = beta1 * mb[i] + (1 - beta1) * dbs[i]
                        vb[i] = beta2 * vb[i] + (1 - beta2) * dbs[i] ** 2
                        mWh = mW[i] / (1 - beta1 ** t)
                        vWh = vW[i] / (1 - beta2 ** t)
                        mbh = mb[i] / (1 - beta1 ** t)
                        vbh = vb[i] / (1 - beta2 ** t)
                        self.weights[i] -= self.learning_rate * mWh / (np.sqrt(vWh) + eps)
                        self.biases[i] -= self.learning_rate * mbh / (np.sqrt(vbh) + eps)
                else:
                    for i in range(len(self.weights)):
                        self.weights[i] -= self.learning_rate * dWs[i]
                        self.biases[i] -= self.learning_rate * dbs[i]

            # Record epoch loss
            acts_full = self._forward(X)
            self.loss_history.append(self._loss(acts_full[-1], y))

        return self

    def predict_proba(self, X):
        """Return raw output neuron value (probability for classification)."""
        return self._forward(np.array(X, dtype=float))[-1]

    def predict(self, X):
        proba = self.predict_proba(X).ravel()
        if self.task == "classification":
            return (proba >= 0.5).astype(int)
        return proba

    def score(self, X, y):
        y = np.array(y, dtype=float)
        if self.task == "classification":
            return np.mean(self.predict(X) == y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
