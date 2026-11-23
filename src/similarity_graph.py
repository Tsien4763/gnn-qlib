from __future__ import annotations

import numpy as np


def build_daily_similarity_graph(embeddings: np.ndarray, top_k: int = 20, eps: float = 1e-12) -> np.ndarray:
    """Construct symmetric normalized daily cosine-similarity adjacency matrix.

    Matches the daily sparse top-k cosine graph formulation in qlib_gcn_model.py:
    row-wise L2 normalization, nonnegative off-diagonal cosine similarities,
    top-k sparsification, max-symmetrization, and D^{-1/2} (A + I) D^{-1/2}.
    """
    h = np.asarray(embeddings, dtype=float)
    if h.ndim != 2 or h.shape[0] == 0:
        raise ValueError("embeddings must be a nonempty 2D array")
    n = h.shape[0]
    if n == 1:
        return np.ones((1, 1), dtype=float)

    norms = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), eps)
    h_norm = h / norms
    sim = np.maximum(0.0, h_norm @ h_norm.T)
    np.fill_diagonal(sim, 0.0)

    if top_k < n:
        sparse = np.zeros_like(sim)
        for i in range(n):
            idx = np.argpartition(sim[i], -top_k)[-top_k:]
            sparse[i, idx] = sim[i, idx]
    else:
        sparse = sim

    adj = np.maximum(sparse, sparse.T)
    adj_hat = adj + np.eye(n, dtype=float)
    deg = adj_hat.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (inv_sqrt[:, None] * adj_hat) * inv_sqrt[None, :]


def two_layer_gcn_forward(
    embeddings: np.ndarray,
    adj_norm: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    head_w: np.ndarray,
) -> np.ndarray:
    """Two-layer GCN propagation concatenated with original temporal embeddings."""
    z1 = np.maximum(0.0, adj_norm @ (embeddings @ w1))
    z2 = np.maximum(0.0, adj_norm @ (z1 @ w2))
    fused = np.concatenate([embeddings, z2], axis=1)
    return fused @ head_w
