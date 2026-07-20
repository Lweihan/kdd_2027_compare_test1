"""
wide_deep.py — Wide & Deep 精排模型
=======================================
Google 2016: Wide (线性记忆) + Deep (泛化) 组合模型
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .base_ranker import BaseRanker


class WideDeep(BaseRanker):
    """
    Wide & Deep 精排模型。
    
    架构:
    - Wide 部分: 稀疏特征的一阶项 (线性组合)
    - Deep 部分: Embedding → MLP
    - 输出: Wide + Deep → Sigmoid
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        feature_voc_sizes: Dict[str, int],
        embed_dim: int = 16,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        hidden_dims = hidden_dims or [256, 128, 64]
        super().__init__(
            sparse_feature_names=sparse_feature_names,
            feature_voc_sizes=feature_voc_sizes,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

        # Wide 部分: 稀疏特征的一阶线性项
        self.wide_bias = nn.Parameter(torch.zeros(1))
        self.wide_weights = nn.ModuleDict({
            feat_name: nn.Embedding(
                num_embeddings=feature_voc_sizes[feat_name],
                embedding_dim=1,
                padding_idx=0,
            )
            for feat_name in sparse_feature_names
        })

        # Deep 部分: MLP
        layers = []
        input_dim = self.concat_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*layers)
        self.deep_output = nn.Linear(input_dim, 1)

        # 用于提取 embedding
        self._embed_projection = nn.Linear(self.concat_embed_dim + len(sparse_feature_names), hidden_dims[-1])

    def wide_forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Wide 部分: 一阶特征交叉"""
        wide_sum = self.wide_bias.expand(sparse_values[self.sparse_feature_names[0]].size(0))
        for feat_name in self.sparse_feature_names:
            wide_sum = wide_sum + self.wide_weights[feat_name](sparse_values[feat_name]).squeeze(-1)
        return wide_sum  # (batch,)

    def deep_forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Deep 部分: Embedding → MLP"""
        concat_embeds = self.embed_input(sparse_values)
        deep_out = self.deep_mlp(concat_embeds)
        return self.deep_output(deep_out).squeeze(-1)  # (batch,)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        wide_out = self.wide_forward(sparse_values)
        deep_out = self.deep_forward(sparse_values)
        logit = wide_out + deep_out
        return logit

    def predict(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测接口，返回 sigmoid 后的概率"""
        return torch.sigmoid(self.forward(sparse_values))

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 Wide + Deep 联合 embedding"""
        concat_embeds = self.embed_input(sparse_values)
        # 拼接 Wide 的一阶特征
        wide_features = []
        for feat_name in self.sparse_feature_names:
            wide_features.append(self.wide_weights[feat_name](sparse_values[feat_name]).squeeze(-1))
        wide_tensor = torch.stack(wide_features, dim=1)  # (batch, num_sparse)
        combined = torch.cat([concat_embeds, wide_tensor], dim=1)
        return self._embed_projection(combined)
