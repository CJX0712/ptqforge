"""A small, dependency-free numpy MLP used as the quantization target.

Trained with Adam on cross-entropy. Weights are plain ``np.ndarray`` lists so
they can be quantized layer-by-layer. Deterministic given an ``rng``.

Author: 晨星
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.errors import ModelError


class MLP:
    """Minimal ReLU MLP (linear output, softmax + cross-entropy in fit)."""

    def __init__(
        self,
        layers: List[int],
        lr: float = 0.01,
        epochs: int = 60,
        batch: int = 128,
        weight_decay: float = 1e-4,
    ) -> None:
        if len(layers) < 2:
            raise ModelError("MLP needs at least 2 layers")
        self.layers = list(layers)
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.weight_decay = weight_decay
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

    @staticmethod
    def _relu(a: np.ndarray) -> np.ndarray:
        return np.maximum(a, 0.0)

    def _init_params(self, rng: np.random.RandomState) -> None:
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            nin, nout = self.layers[i], self.layers[i + 1]
            scale = np.sqrt(2.0 / nin)
            self.W.append(rng.standard_normal((nout, nin)).astype(np.float64) * scale)
            self.b.append(np.zeros(nout, dtype=np.float64))

    def forward(self, X, W: Optional[List[np.ndarray]] = None, b: Optional[List[np.ndarray]] = None) -> np.ndarray:
        W = self.W if W is None else W
        b = self.b if b is None else b
        a = np.asarray(X, dtype=np.float64)
        for i in range(len(W) - 1):
            a = self._relu(a @ W[i].T + b[i])
        return a @ W[-1].T + b[-1]

    def layer_inputs(self, X, W: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Activations feeding each layer (a_0 = X, a_i = output of layer i-1)."""
        W = self.W if W is None else W
        outs: List[np.ndarray] = [np.asarray(X, dtype=np.float64)]
        a = outs[0]
        for i in range(len(W) - 1):
            a = self._relu(a @ W[i].T + self.b[i])
            outs.append(a)
        return outs

    def predict(self, X, W: Optional[List[np.ndarray]] = None) -> np.ndarray:
        return np.argmax(self.forward(X, W=W), axis=1)

    def score(self, X, y, W: Optional[List[np.ndarray]] = None) -> float:
        return float(np.mean(self.predict(X, W=W) == np.asarray(y)))

    def score_with_weights(self, X, y, Ws: List[np.ndarray]) -> float:
        return self.score(X, y, W=Ws)

    def fit(self, X, y, rng: np.random.RandomState) -> "MLP":
        self._init_params(rng)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        n_out = self.layers[-1]
        Y = np.zeros((n, n_out), dtype=np.float64)
        Y[np.arange(n), y] = 1.0
        idx = np.arange(n)
        b1, b2, eps = 0.9, 0.999, 1e-8
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        t = 0
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for s in range(0, n, self.batch):
                xb = X[idx[s : s + self.batch]]
                yb = Y[idx[s : s + self.batch]]
                B = xb.shape[0]
                acts = [xb]
                a = xb
                zs = []
                for i in range(len(self.W) - 1):
                    z = a @ self.W[i].T + self.b[i]
                    zs.append(z)
                    a = self._relu(z)
                    acts.append(a)
                logits = a @ self.W[-1].T + self.b[-1]
                logits -= logits.max(axis=1, keepdims=True)
                e = np.exp(logits)
                p = e / e.sum(axis=1, keepdims=True)
                dlogits = (p - yb) / B
                gW = [None] * len(self.W)
                gb = [None] * len(self.b)
                gW[-1] = dlogits.T @ acts[-1]
                gb[-1] = dlogits.sum(0)
                delta = dlogits @ self.W[-1]
                for i in range(len(self.W) - 2, -1, -1):
                    delta = delta * (zs[i] > 0).astype(np.float64)
                    gW[i] = delta.T @ acts[i]
                    gb[i] = delta.sum(0)
                    if i > 0:
                        delta = delta @ self.W[i]
                t += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * (gW[i] ** 2)
                    mhat = mW[i] / (1 - b1**t)
                    vhat = vW[i] / (1 - b2**t)
                    self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
                    self.W[i] -= self.weight_decay * self.W[i]
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * (gb[i] ** 2)
                    mhatb = mb[i] / (1 - b1**t)
                    vhatb = vb[i] / (1 - b2**t)
                    self.b[i] -= self.lr * mhatb / (np.sqrt(vhatb) + eps)
        return self


def build_default_mlp(n_features: int, n_classes: int) -> MLP:
    # 3 hidden layers: deeper nets compound quantization error, so mixed-precision
    # PTQ shows a clearer, realistic accuracy gap (the real deployment scenario).
    return MLP(layers=[n_features, 64, 48, 32, n_classes], lr=0.02, epochs=200, batch=64)
