from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from similarity_graph import build_daily_similarity_graph, two_layer_gcn_forward


class TestSimilarityGraph(unittest.TestCase):
    def test_single_node_identity(self) -> None:
        adj = build_daily_similarity_graph(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(adj.shape, (1, 1))
        self.assertAlmostEqual(float(adj[0, 0]), 1.0)

    def test_symmetry_and_forward_shape(self) -> None:
        rng = np.random.default_rng(7)
        h = rng.normal(size=(12, 8))
        adj = build_daily_similarity_graph(h, top_k=4)
        self.assertTrue(np.allclose(adj, adj.T))
        self.assertTrue(np.all(adj >= 0.0))
        scores = two_layer_gcn_forward(
            h, adj, rng.normal(size=(8, 4)), rng.normal(size=(4, 4)), rng.normal(size=(12,))
        )
        self.assertEqual(scores.shape, (12,))
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
