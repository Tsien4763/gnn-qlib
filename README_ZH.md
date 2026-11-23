# Sequence-Graph Equity Ranking（时序动态图排序）

[English](README.md)

在没有人工维护的行业分类或供应链关系表时，截面股票排序模型能否直接从量价时序嵌入中学习每日动态股票关联图？

本仓库实现与 Qlib 兼容的时序动态图排序架构。共享双层 GRU 先编码 Alpha360 量价序列，按交易日构造对称归一化的稀疏 Top-k 余弦相似度图（`D^{-1/2}(A + I)D^{-1/2}`），经过两层图卷积聚合邻域信息后，与原始时序嵌入拼接输入预测头。

- `src/qlib_gcn_model.py`：完整 PyTorch 与 Qlib 模型实现（`GraphConvolution`、`TemporalGCNNet`、`QlibGCN`）。
- `src/similarity_graph.py`：纯 NumPy 实现的非负 Top-k 余弦构图、对称归一化与双层 GCN 前向参考。

## 快速开始

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` 在 30 节点合成日度批次上运行 NumPy 动态构图与双层图卷积前向传播，输出邻接对称性、边密度与排序得分。

## 范围说明

本仓库为包含完整 Qlib 模型源码与纯 NumPy 合成演示的公开 MVP，完整行情训练脚本与数据配置按需提供。
