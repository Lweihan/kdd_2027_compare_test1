"""
deepfm.py — DeepFM 精排模型
===============================
华为 2017: FM (二阶特征交叉) + DNN (深度特征) 组合模型
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .base_ranker import BaseRanker


class FMComponent(nn.Module):
    """FM 二阶交叉组件: 计算特征之间的二阶交叉项"""

    def __init__(self):
        super().__init__()

    def forward(self, embed_list: List[torch.Tensor]) -> torch.Tensor:
        """
        计算 FM 二阶交叉项。
        
        公式: FM(x) = 0.5 * (sum(embed)^2 - sum(embed^2))
        
        Args:
            embed_list: 各特征的 embedding 列表，每个 (batch, embed_dim)
        
        Returns:
            fm_out: (batch,) FM 二阶输出
        """
        # Stack: (batch, num_features, embed_dim)
        stacked = torch.stack(embed_list, dim=1)

        # sum of square
        sum_of_square = torch.sum(stacked ** 2, dim=1)  # (batch, embed_dim)

        # square of sum
        square_of_sum = torch.sum(stacked, dim=1) ** 2  # (batch, embed_dim)

        # FM 二阶交叉
        fm_out = 0.5 * (square_of_sum - sum_of_square)  # (batch, embed_dim)
        fm_out = torch.sum(fm_out, dim=1)  # (batch,)

        return fm_out


class DeepFM(BaseRanker):
    """
    DeepFM 精排模型。
    
    架构:
    - FM 部分: 一阶项 + 二阶交叉项
    - Deep 部分: Embedding → MLP
    - 输出: FM + Deep → Sigmoid
    
    特点: FM 和 Deep 共享 Embedding 层
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

        # FM 一阶项
        self.fm_first_order = nn.ModuleDict({
            feat_name: nn.Embedding(
                num_embeddings=feature_voc_sizes[feat_name],
                embedding_dim=1,
                padding_idx=0,
            )
            for feat_name in sparse_feature_names
        })

        # FM 二阶交叉
        self.fm_component = FMComponent()

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
        self._embed_projection = nn.Linear(self.concat_embed_dim, hidden_dims[-1])

    def fm_first_order_forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FM 一阶项"""
        first_order = []
        for feat_name in self.sparse_feature_names:
            first_order.append(self.fm_first_order[feat_name](sparse_values[feat_name]).squeeze(-1))
        return torch.stack(first_order, dim=1).sum(dim=1)  # (batch,)

    def fm_second_order_forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """FM 二阶交叉项"""
        embed_list = []
        for feat_name in self.sparse_feature_names:
            embed_list.append(self.embeddings[feat_name](sparse_values[feat_name]))
        return self.fm_component(embed_list)  # (batch,)

    def deep_forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Deep 部分"""
        concat_embeds = self.embed_input(sparse_values)
        deep_out = self.deep_mlp(concat_embeds)
        return self.deep_output(deep_out).squeeze(-1)  # (batch,)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        fm_first = self.fm_first_order_forward(sparse_values)
        fm_second = self.fm_second_order_forward(sparse_values)
        deep_out = self.deep_forward(sparse_values)
        logit = fm_first + fm_second + deep_out
        return torch.sigmoid(logit)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 Deep 部分的 embedding"""
        concat_embeds = self.embed_input(sparse_values)
        return self._embed_projection(concat_embeds)
