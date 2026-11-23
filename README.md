# Sequence-Graph Equity Ranking

Daily cosine-similarity graph construction and two-layer GCN ranking head for cross-sectional time-series panels.

Instead of relying on static industry tables, the model builds a sparse daily similarity graph directly from temporal node embeddings and concatenates the original sequence representation with graph-aggregated features before scoring.

- `src/qlib_gcn_model.py`: PyTorch and Qlib implementation (`GraphConvolution`, `TemporalGCNNet`, `QlibGCN`).
- `src/similarity_graph.py`: NumPy reference for nonnegative top-k cosine graph construction, symmetric normalization, and two-layer graph propagation.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` constructs a 30-node synthetic daily batch, builds the top-6 normalized similarity graph, and prints score summary statistics.

## Scope

The demo runs on synthetic node embeddings with NumPy only; full Qlib training requires PyTorch, Qlib, and user-supplied market data.
