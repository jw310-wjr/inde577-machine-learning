"""
autoencoder.py
Feed-forward Autoencoder (encoder–decoder) from scratch using NumPy.

Architecture: Input → Encoder (dense layers) → Bottleneck → Decoder → Reconstruction
Loss: Mean Squared Error (reconstruction loss)
Optimizer: Adam
"""

import numpy as np


class Autoencoder:
    """
    Symmetric feed-forward Autoencoder for unsupervised dimensionality
    reduction and feature learning.

    Parameters
    ----------
    encoding_dims : tuple of ints
        Neuron counts for each encoder hidden layer (decoder mirrors these).
        The last value is the bottleneck (latent) dimension.
        E.g. (128, 64, 32) → encoder 128→64→32, decoder 32→64→128.
    activation    : str, 'relu' | 'sigmoid' | 'tanh' (default 'relu')
    learning_rate : float (default 1e-3)
    n_iterations  : int, training epochs (default 500)
    batch_size    : int (default 32)
    lambda_reg    : float, L2 weight-decay (default 0.0)
    random_state  : int or None
    verbose       : bool, print loss every 100 epochs (default False)
    """

    def __init__(self, encoding_dims=(64, 32), activation="relu",
                 learning_rate=1e-3, n_iterations=500, batch_size=32,
                 lambda_reg=0.0, random_state=None, verbose=False):
        self.encoding_dims = encoding_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.verbose = verbose
        self.weights_ = []
        self.biases_ = []
        self.loss_history_ = []

    # ── activations ───────────────────────────────────────────────────────────

    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0.0, z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        if self.activation == "tanh":
            return np.tanh(z)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _act_d(self, a):
        if self.activation == "relu":
            return (a > 0).astype(float)
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        if self.activation == "tanh":
            return 1.0 - a ** 2
        raise ValueError(f"Unknown activation: {self.activation}")

    # ── weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self, layer_sizes):
        rng = np.random.RandomState(self.random_state)
        self.weights_, self.biases_ = [], []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i]) if self.activation == "relu" \
                else np.sqrt(1.0 / layer_sizes[i])
            W = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights_.append(W)
            self.biases_.append(b)
        # Adam moment estimates
        self._m = [np.zeros_like(W) for W in self.weights_]
        self._v = [np.zeros_like(W) for W in self.weights_]
        self._mb = [np.zeros_like(b) for b in self.biases_]
        self._vb = [np.zeros_like(b) for b in self.biases_]
        self._t = 0

    # ── forward pass ─────────────────────────────────────────────────────────

    def _forward(self, X):
        acts = [X]
        cur = X
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = cur @ W + b
            # output layer uses linear (sigmoid clamped) activation
            if i == len(self.weights_) - 1:
                a = self._sigmoid_out(z)
            else:
                a = self._act(z)
            acts.append(a)
            cur = a
        return acts

    @staticmethod
    def _sigmoid_out(z):
        """Sigmoid for output layer — keeps reconstruction in (0,1) range."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # ── backprop ──────────────────────────────────────────────────────────────

    def _backprop(self, acts, X_batch):
        n = X_batch.shape[0]
        # MSE reconstruction loss gradient at output
        out = acts[-1]
        dA = (2.0 / n) * (out - X_batch) * out * (1.0 - out)  # sigmoid deriv at output

        grads_W = [None] * len(self.weights_)
        grads_b = [None] * len(self.biases_)

        for i in reversed(range(len(self.weights_))):
            grads_W[i] = acts[i].T @ dA + self.lambda_reg * self.weights_[i]
            grads_b[i] = dA.sum(axis=0, keepdims=True)
            if i > 0:
                dA = (dA @ self.weights_[i].T) * self._act_d(acts[i])
        return grads_W, grads_b

    def _adam_update(self, grads_W, grads_b):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        lr_t = self.learning_rate * np.sqrt(1 - beta2 ** self._t) / (1 - beta1 ** self._t)
        for i in range(len(self.weights_)):
            self._m[i]  = beta1 * self._m[i]  + (1 - beta1) * grads_W[i]
            self._v[i]  = beta2 * self._v[i]  + (1 - beta2) * grads_W[i] ** 2
            self._mb[i] = beta1 * self._mb[i] + (1 - beta1) * grads_b[i]
            self._vb[i] = beta2 * self._vb[i] + (1 - beta2) * grads_b[i] ** 2
            self.weights_[i] -= lr_t * self._m[i]  / (np.sqrt(self._v[i])  + eps)
            self.biases_[i]  -= lr_t * self._mb[i] / (np.sqrt(self._vb[i]) + eps)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Train the autoencoder on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), values in [0, 1]
        """
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        # Build symmetric layer sizes: input → enc1 → enc2 … → dec2 → dec1 → input
        enc = list(self.encoding_dims)
        dec = list(reversed(enc[:-1]))
        layer_sizes = [n_features] + enc + dec + [n_features]
        self._bottleneck_idx = len(enc)  # index of bottleneck activations
        self._layer_sizes = layer_sizes
        self._init_weights(layer_sizes)

        rng = np.random.RandomState(self.random_state)
        self.loss_history_ = []
        n = X.shape[0]

        for epoch in range(1, self.n_iterations + 1):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = X[idx[start:start + self.batch_size]]
                acts = self._forward(batch)
                loss = np.mean((acts[-1] - batch) ** 2)
                epoch_loss += loss * len(batch)
                grads_W, grads_b = self._backprop(acts, batch)
                self._adam_update(grads_W, grads_b)
            self.loss_history_.append(epoch_loss / n)
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d}  MSE={self.loss_history_[-1]:.6f}")
        return self

    def encode(self, X):
        """Map X to the bottleneck (latent) representation."""
        X = np.asarray(X, dtype=float)
        acts = self._forward(X)
        return acts[self._bottleneck_idx]

    def decode(self, Z):
        """Reconstruct from latent codes Z."""
        Z = np.asarray(Z, dtype=float)
        cur = Z
        for i in range(self._bottleneck_idx, len(self.weights_)):
            z = cur @ self.weights_[i] + self.biases_[i]
            if i == len(self.weights_) - 1:
                cur = self._sigmoid_out(z)
            else:
                cur = self._act(z)
        return cur

    def transform(self, X):
        """Alias for encode()."""
        return self.encode(X)

    def fit_transform(self, X):
        """Fit and return latent codes."""
        return self.fit(X).encode(X)

    def reconstruct(self, X):
        """Return full reconstruction."""
        X = np.asarray(X, dtype=float)
        return self._forward(X)[-1]

    def reconstruction_loss(self, X):
        """MSE between X and its reconstruction."""
        X = np.asarray(X, dtype=float)
        return float(np.mean((self.reconstruct(X) - X) ** 2))
