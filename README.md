# Sequence-Graph Equity Ranking

[中文说明](README_ZH.md)

Can a cross-sectional ranking model learn useful daily stock relations directly from price-volume embeddings without hand-curated sector or supply-chain graphs?

This repository implements a temporal-graph architecture compatible with Qlib. A shared two-layer GRU encodes Alpha360 price-volume histories, constructs a sparse symmetric top-k cosine similarity graph per trading day (`D^{-1/2}(A + I)D^{-1/2}`), runs two graph-convolution layers, and concatenates the graph representation with the original temporal embedding before scoring.

- `src/qlib_gcn_model.py`: full PyTorch and Qlib implementation (`GraphConvolution`, `TemporalGCNNet`, `QlibGCN`).
- `src/similarity_graph.py`: NumPy reference for nonnegative top-k cosine graph construction, symmetric normalization, and two-layer GCN propagation.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` runs the NumPy similarity-graph and two-layer GCN forward pass on a 30-node synthetic daily batch and prints adjacency symmetry, edge density, and top-ranked nodes.

## Scope

This is the public MVP with a zero-PyTorch synthetic demo alongside the full Qlib model module. Full Qlib training configs and market datasets are available upon request.
