"""
dnn.py — DNN (Deep Neural Network) 精排模型
===============================================
最基本的深度学习 CTR 模型: Embedding → MLP → Sigmoid
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .base_ranker import BaseRanker


class DNN(BaseRanker):
    """
    DNN 精排模型。
    
    架构: Sparse Features → Embedding → Concat → MLP → Sigmoid
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

        # MLP 层
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

        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # 用于提取 embedding 的中间层
        self.embedding_layer = nn.Sequential()  # placeholder
        self._embed_projection = nn.Linear(self.concat_embed_dim, hidden_dims[-1])

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat_embeds = self.embed_input(sparse_values)  # (batch, num_sparse * embed_dim)
        logit = self.mlp(concat_embeds)  # (batch, 1)
        return logit.squeeze(-1)

    def predict(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测接口，返回 sigmoid 后的概率"""
        return torch.sigmoid(self.forward(sparse_values))

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 embedding (拼接后的 embedding 投影到 hidden_dims[-1] 维)"""
        concat_embeds = self.embed_input(sparse_values)
        return self._embed_projection(concat_embeds)  # (batch, hidden_dims[-1])
