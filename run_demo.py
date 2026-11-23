from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from similarity_graph import build_daily_similarity_graph, two_layer_gcn_forward


def main() -> None:
    rng = np.random.default_rng(42)
    n_nodes, d_in, d_hid = 30, 16, 8
    embeddings = rng.normal(size=(n_nodes, d_in))
    adj = build_daily_similarity_graph(embeddings, top_k=6)
    w1 = rng.normal(scale=0.2, size=(d_in, d_hid))
    w2 = rng.normal(scale=0.2, size=(d_hid, d_hid))
    head = rng.normal(scale=0.2, size=(d_in + d_hid,))
    scores = two_layer_gcn_forward(embeddings, adj, w1, w2, head)

    print("=== Sequence-Graph Equity Ranking MVP (synthetic demo) ===")
    print(f"Nodes: {n_nodes} | Top-k: 6 | Adjacency symmetric: {bool(np.allclose(adj, adj.T))}")
    print(f"Nonzero edges (incl. self-loops): {int(np.count_nonzero(adj))}")
    print(f"Score summary: mean={scores.mean():.4f}, std={scores.std():.4f}, top3={np.argsort(-scores)[:3].tolist()}")


if __name__ == "__main__":
    main()
