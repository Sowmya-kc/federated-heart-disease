"""
model.py -- A small neural network (MLP) written in pure NumPy.
Architecture: input -> 16 (ReLU) -> 8 (ReLU) -> 1 (sigmoid).   Loss: binary cross-entropy + L2.
Written from scratch so every FedAvg step is visible and explainable in the viva.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class MLP:
    def __init__(self, n_in, hidden=(16, 8), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    # ---- parameters as a flat vector (this is what clients send to the server) ----
    def get_params(self):
        return np.concatenate([p.ravel() for p in (*self.W, *self.b)])

    def set_params(self, vec):
        i = 0
        for lst in (self.W, self.b):
            for p in lst:
                p[...] = vec[i:i + p.size].reshape(p.shape)
                i += p.size

    @property
    def n_params(self):
        return sum(p.size for p in (*self.W, *self.b))

    def forward(self, X):
        acts = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(sigmoid(z) if k == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def predict_proba(self, X):
        return self.forward(X)[-1].ravel()

    def loss(self, X, y, l2=0.0):
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return bce + 0.5 * l2 * sum((W ** 2).sum() for W in self.W)

    def grads(self, X, y, l2=0.0):
        acts = self.forward(X)
        n = len(X)
        delta = (acts[-1].ravel() - y)[:, None] / n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = acts[k].T @ delta + l2 * self.W[k]
            gb[k] = delta.sum(0)
            if k > 0:
                delta = (delta @ self.W[k].T) * (acts[k] > 0)
        return gW, gb

    def sgd_epochs(self, X, y, epochs, lr, batch, l2=0.0, rng=None, mu=0.0, w_global=None):
        """Mini-batch SGD with momentum.  mu>0 adds the FedProx proximal term
        mu/2 * ||w - w_global||^2 that keeps local models close to the global one."""
        rng = rng or np.random.default_rng(0)
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                j = idx[s:s + batch]
                gW, gb = self.grads(X[j], y[j], l2)
                if mu > 0 and w_global is not None:
                    flat = self.get_params()
                    prox = mu * (flat - w_global)
                    pW, pb = self._unflatten(prox)
                    gW = [g + p for g, p in zip(gW, pW)]
                    gb = [g + p for g, p in zip(gb, pb)]
                for k in range(len(self.W)):
                    vW[k] = 0.9 * vW[k] - lr * gW[k]
                    vb[k] = 0.9 * vb[k] - lr * gb[k]
                    self.W[k] += vW[k]
                    self.b[k] += vb[k]

    def _unflatten(self, vec):
        i, Ws, bs = 0, [], []
        for lst, out in ((self.W, Ws), (self.b, bs)):
            for p in lst:
                out.append(vec[i:i + p.size].reshape(p.shape)); i += p.size
        return Ws, bs
