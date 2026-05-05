"""
cnn.py
Introductory Convolutional Neural Network (CNN) from scratch using NumPy.

Architecture: Conv2D (+ ReLU) → MaxPool2D → Flatten → Dense (+ ReLU) → Output (softmax)

Supports:
- Single conv layer with multiple filters (kernels)
- ReLU activation after convolution
- Max-pooling
- Fully-connected output layer
- SGD or Adam optimisation
- Multi-class classification (softmax + cross-entropy)
"""

import numpy as np


class Conv2D:
    """Single convolutional layer (no padding, stride=1)."""

    def __init__(self, n_filters, kernel_size, rng):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        k = kernel_size
        scale = np.sqrt(2.0 / (k * k))
        # shape: (n_filters, k, k)
        self.W = rng.randn(n_filters, k, k) * scale
        self.b = np.zeros(n_filters)

    def forward(self, X):
        """
        X : (n, H, W)
        Returns : (n, n_filters, H-k+1, W-k+1)
        """
        n, H, W = X.shape
        k = self.kernel_size
        out_h, out_w = H - k + 1, W - k + 1
        self._X = X
        out = np.zeros((n, self.n_filters, out_h, out_w))
        for f in range(self.n_filters):
            for i in range(out_h):
                for j in range(out_w):
                    out[:, f, i, j] = np.sum(
                        X[:, i:i+k, j:j+k] * self.W[f], axis=(1, 2)
                    ) + self.b[f]
        return out

    def backward(self, dout, lr):
        """dout : (n, n_filters, out_h, out_w)"""
        n, H, W = self._X.shape
        k = self.kernel_size
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self._X)

        for f in range(self.n_filters):
            for i in range(dout.shape[2]):
                for j in range(dout.shape[3]):
                    dW[f] += np.sum(
                        self._X[:, i:i+k, j:j+k] * dout[:, f, i:i+1, j:j+1],
                        axis=0
                    )
                    db[f] += dout[:, f, i, j].sum()
                    dX[:, i:i+k, j:j+k] += dout[:, f, i, j][:, None, None] * self.W[f]

        self.W -= lr * dW / n
        self.b -= lr * db / n
        return dX


class MaxPool2D:
    """2×2 max-pooling with stride 2."""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        """X : (n, C, H, W)"""
        n, C, H, W = X.shape
        p = self.pool_size
        out_h, out_w = H // p, W // p
        self._X = X
        out = np.zeros((n, C, out_h, out_w))
        self._mask = np.zeros_like(X)
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                out[:, :, i, j] = patch.max(axis=(2, 3))
                # store max-position mask for backprop
                max_val = out[:, :, i, j][:, :, None, None]
                self._mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (patch == max_val)
        return out

    def backward(self, dout):
        """dout : (n, C, out_h, out_w)"""
        p = self.pool_size
        dX = np.zeros_like(self._X)
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                dX[:, :, i*p:(i+1)*p, j*p:(j+1)*p] += (
                    self._mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                    * dout[:, :, i, j][:, :, None, None]
                )
        return dX


class SimpleCNN:
    """
    A minimal CNN for image classification (NumPy from scratch).

    Architecture:
        Input (n, H, W)
        → Conv2D(n_filters, kernel_size) + ReLU
        → MaxPool2D(pool_size)
        → Flatten
        → Dense(dense_units) + ReLU
        → Output (n_classes, softmax)

    Parameters
    ----------
    n_filters    : int, number of convolutional filters (default 8)
    kernel_size  : int, square kernel size (default 3)
    pool_size    : int, max-pool window (default 2)
    dense_units  : int, units in the dense hidden layer (default 64)
    learning_rate: float (default 1e-3)
    n_iterations : int, training epochs (default 20)
    batch_size   : int (default 32)
    random_state : int or None
    verbose      : bool (default False)
    """

    def __init__(self, n_filters=8, kernel_size=3, pool_size=2,
                 dense_units=64, learning_rate=1e-3, n_iterations=20,
                 batch_size=32, random_state=None, verbose=False):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history_ = []

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_d(a):
        return (a > 0).astype(float)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ── init weights ──────────────────────────────────────────────────────────

    def _init(self, input_shape, n_classes):
        rng = np.random.RandomState(self.random_state)
        H, W = input_shape
        self._conv = Conv2D(self.n_filters, self.kernel_size, rng)
        self._pool = MaxPool2D(self.pool_size)

        conv_out_h = (H - self.kernel_size + 1) // self.pool_size
        conv_out_w = (W - self.kernel_size + 1) // self.pool_size
        flat_size = self.n_filters * conv_out_h * conv_out_w

        scale1 = np.sqrt(2.0 / flat_size)
        self._W1 = rng.randn(flat_size, self.dense_units) * scale1
        self._b1 = np.zeros(self.dense_units)

        scale2 = np.sqrt(2.0 / self.dense_units)
        self._W2 = rng.randn(self.dense_units, n_classes) * scale2
        self._b2 = np.zeros(n_classes)

    # ── forward ───────────────────────────────────────────────────────────────

    def _forward(self, X):
        """X : (n, H, W), pixel values in [0, 1]"""
        c_out = self._conv.forward(X)            # (n, F, H', W')
        c_act = self._relu(c_out)
        p_out = self._pool.forward(c_act)        # (n, F, H'', W'')
        flat = p_out.reshape(X.shape[0], -1)     # (n, flat)
        d1 = self._relu(flat @ self._W1 + self._b1)
        logits = d1 @ self._W2 + self._b2
        probs = self._softmax(logits)
        return c_out, c_act, p_out, flat, d1, probs

    # ── backward ──────────────────────────────────────────────────────────────

    def _backward(self, cache, y_oh):
        c_out, c_act, p_out, flat, d1, probs = cache
        n = y_oh.shape[0]
        lr = self.learning_rate

        # output layer
        dlogits = (probs - y_oh) / n
        dW2 = d1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dd1 = dlogits @ self._W2.T

        # dense hidden
        dd1 *= self._relu_d(d1)
        dW1 = flat.T @ dd1
        db1 = dd1.sum(axis=0)
        dflat = dd1 @ self._W1.T

        # update dense
        self._W2 -= lr * dW2
        self._b2 -= lr * db2
        self._W1 -= lr * dW1
        self._b1 -= lr * db1

        # unflatten → pool backprop
        dp_out = dflat.reshape(p_out.shape)
        dc_act = self._pool.backward(dp_out)

        # relu after conv
        dc_out = dc_act * self._relu_d(c_act)

        # conv backprop
        self._conv.backward(dc_out, lr)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the CNN.

        Parameters
        ----------
        X : array-like of shape (n_samples, H, W), pixel values in [0, 1]
        y : array-like of shape (n_samples,), integer class labels
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_classes = int(y.max()) + 1
        self.n_classes_ = n_classes
        self._init(X.shape[1:], n_classes)

        rng = np.random.RandomState(self.random_state)
        self.loss_history_ = []

        for epoch in range(1, self.n_iterations + 1):
            idx = rng.permutation(len(X))
            epoch_loss = 0.0
            for start in range(0, len(X), self.batch_size):
                bi = idx[start:start + self.batch_size]
                Xb, yb = X[bi], y[bi]
                y_oh = np.eye(n_classes)[yb]
                cache = self._forward(Xb)
                probs = cache[-1]
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-12))
                epoch_loss += loss * len(yb)
                self._backward(cache, y_oh)
            self.loss_history_.append(epoch_loss / len(X))
            if self.verbose and epoch % 5 == 0:
                acc = self.score(X, y)
                print(f"Epoch {epoch:3d}  loss={self.loss_history_[-1]:.4f}  acc={acc:.4f}")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._forward(X)[-1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
