"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) — from scratch.

Algorithm outline
-----------------
1. For each pair (i, j), compute high-dimensional conditional probabilities
   p_{j|i} using a Gaussian kernel whose bandwidth sigma_i is chosen so that
   the perplexity of the distribution P_i equals the requested `perplexity`.
   Symmetrize: p_{ij} = (p_{j|i} + p_{i|j}) / (2n).

2. Initialise the low-dimensional embedding Y randomly.

3. At each gradient-descent step:
   - Compute low-dimensional affinities q_{ij} via a Student t-distribution
     with 1 degree of freedom (heavy tail prevents crowding).
   - Compute the gradient of KL(P || Q) w.r.t. Y.
   - Update Y with momentum and learning rate.

References
----------
van der Maaten & Hinton (2008). Visualizing Data using t-SNE. JMLR 9, 2579-2605.
"""

import numpy as np


class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Reduces high-dimensional data to `n_components` dimensions (typically 2)
    for visualization.

    Parameters
    ----------
    n_components  : int    Target dimensionality (default 2).
    perplexity    : float  Effective number of neighbours (default 30).
                           Typical range: 5 – 50.
    learning_rate : float  Gradient-descent step size (default 200.0).
    n_iter        : int    Number of gradient-descent iterations (default 1000).
    momentum      : float  Momentum coefficient for gradient update (default 0.8).
    early_exaggeration : float
                           Multiply P by this factor for the first
                           `n_iter_early_exag` iterations to help clusters
                           separate early (default 12.0).
    n_iter_early_exag  : int
                           Iterations to apply early exaggeration (default 250).
    tol           : float  Binary-search tolerance for sigma (default 1e-5).
    random_state  : int or None   RNG seed (default None).
    verbose       : bool   Print loss every 100 iterations (default False).
    """

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=1000,
        momentum=0.8,
        early_exaggeration=12.0,
        n_iter_early_exag=250,
        tol=1e-5,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.momentum = momentum
        self.early_exaggeration = early_exaggeration
        self.n_iter_early_exag = n_iter_early_exag
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.embedding_ = None
        self.kl_divergence_ = None
        self.loss_history_ = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_sq_distances(X):
        """Compute ||x_i - x_j||^2 for all pairs. Returns (n, n) matrix."""
        sum_sq = np.sum(X ** 2, axis=1, keepdims=True)
        return sum_sq + sum_sq.T - 2.0 * (X @ X.T)

    def _compute_p_matrix(self, D_sq):
        """
        Compute symmetrised joint probabilities P using binary search for sigma.

        D_sq : (n, n) squared-distance matrix (diagonal zeroed).
        Returns P : (n, n) symmetric probability matrix.
        """
        n = D_sq.shape[0]
        log_perp = np.log(self.perplexity)
        P = np.zeros((n, n))

        for i in range(n):
            # distances from point i (exclude self)
            d_i = D_sq[i].copy()
            d_i[i] = np.inf

            # Binary search for sigma_i such that H(P_i) == log(perplexity)
            beta_lo, beta_hi = -np.inf, np.inf
            beta = 1.0  # beta = 1 / (2 * sigma_i^2)

            for _ in range(50):  # max 50 binary-search steps
                exp_d = np.exp(-d_i * beta)
                exp_d[i] = 0.0
                sum_exp = exp_d.sum() + 1e-10
                # Entropy H = log(sum) + beta * weighted_mean_dist
                H = np.log(sum_exp) + beta * np.dot(d_i, exp_d) / sum_exp
                diff = H - log_perp

                if abs(diff) < self.tol:
                    break
                if diff > 0:
                    beta_lo = beta
                    beta = (beta + beta_hi) / 2.0 if beta_hi != np.inf else beta * 2.0
                else:
                    beta_hi = beta
                    beta = (beta + beta_lo) / 2.0 if beta_lo != -np.inf else beta / 2.0

            P[i] = exp_d / sum_exp

        # Symmetrize and normalise
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)
        return P

    @staticmethod
    def _compute_q_matrix(Y):
        """
        Compute low-dim affinities Q using Student t-distribution (df=1).
        Returns Q (n, n) and the (n, n) unnormalised numerator matrix.
        """
        sum_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        D_sq = sum_sq + sum_sq.T - 2.0 * (Y @ Y.T)
        num = 1.0 / (1.0 + D_sq)
        np.fill_diagonal(num, 0.0)
        Q = num / (num.sum() + 1e-10)
        Q = np.maximum(Q, 1e-12)
        return Q, num

    def _gradient(self, P, Q, num, Y):
        """
        Gradient of KL(P || Q) w.r.t. Y.
        dC/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i-y_j||^2)^{-1}
        """
        PQ = (P - Q) * num               # (n, n)
        # grad[i] = 4 * sum_j PQ[i,j] * (Y[i] - Y[j])
        grad = 4.0 * (np.diag(PQ.sum(axis=1)) - PQ) @ Y
        return grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, X):
        """
        Fit t-SNE to X and return the low-dimensional embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Y : ndarray, shape (n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Step 1: high-dim probabilities
        D_sq = self._pairwise_sq_distances(X)
        np.fill_diagonal(D_sq, 0.0)
        P = self._compute_p_matrix(D_sq)

        # Step 2: initialise Y
        Y = rng.randn(n, self.n_components) * 1e-4
        velocity = np.zeros_like(Y)
        self.loss_history_ = []

        # Step 3: gradient descent
        for t in range(1, self.n_iter + 1):
            exag = self.early_exaggeration if t <= self.n_iter_early_exag else 1.0
            Q, num = self._compute_q_matrix(Y)
            kl = np.sum(P * np.log(P / Q))
            self.loss_history_.append(kl)

            grad = self._gradient(exag * P, Q, num, Y)
            velocity = self.momentum * velocity - self.learning_rate * grad
            Y = Y + velocity

            # Re-centre to prevent drift
            Y -= Y.mean(axis=0)

            if self.verbose and t % 100 == 0:
                print(f"  t-SNE iter {t:4d}  KL = {kl:.4f}")

        self.embedding_ = Y
        self.kl_divergence_ = self.loss_history_[-1]
        return Y

    def fit(self, X):
        """Fit t-SNE (same as fit_transform, embedding stored in embedding_)."""
        self.fit_transform(X)
        return self
