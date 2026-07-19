"""
dcn.py — DCN (Deep & Cross Network) 精排模型
=================================================
Google 2017: 显式特征交叉网络 + 深度网络的组合
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .base_ranker import BaseRanker


class CrossNetwork(nn.Module):
    """
    Cross Network: 显式特征交叉。
    
    每一层: x_{l+1} = x_0 * (w_l^T * x_l + b_l) + x_l
    
    这种设计能高效地捕获有界阶的特征交叉。
    """

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim) * 0.01)
            for _ in range(num_layers)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) 输入特征
        
        Returns:
            x_out: (batch, input_dim) 交叉网络输出
        """
        x0 = x
        xl = x
        for l in range(self.num_layers):
            # w_l^T * x_l + b_l → (batch,)
            dot = torch.matmul(xl, self.cross_weights[l]) + self.cross_bias[l]  # (batch,)
            # x_0 * dot → (batch, input_dim)
            xl = x0 * dot.unsqueeze(-1) + xl  # (batch, input_dim)
        return xl


class DCN(BaseRanker):
    """
    DCN (Deep & Cross Network) 精排模型。
    
    架构:
    - Cross Network: 显式特征交叉
    - Deep Network: MLP
    - 输出: Cross + Deep → Sigmoid
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        feature_voc_sizes: Dict[str, int],
        embed_dim: int = 16,
        hidden_dims: List[int] = None,
        num_cross_layers: int = 3,
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
        self.num_cross_layers = num_cross_layers

        # Cross Network
        self.cross_network = CrossNetwork(self.concat_embed_dim, num_cross_layers)

        # Deep Network: MLP
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

        # 最终输出: Cross + Deep
        self.output_layer = nn.Linear(self.concat_embed_dim + 1, 1)

        # 用于提取 embedding
        self._embed_projection = nn.Linear(self.concat_embed_dim, hidden_dims[-1])

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat_embeds = self.embed_input(sparse_values)

        # Cross 部分
        cross_out = self.cross_network(concat_embeds)  # (batch, concat_embed_dim)

        # Deep 部分
        deep_out = self.deep_mlp(concat_embeds)
        deep_out = self.deep_output(deep_out)  # (batch, 1)

        # 合并 Cross + Deep
        combined = torch.cat([cross_out, deep_out], dim=1)  # (batch, concat_embed_dim + 1)
        logit = self.output_layer(combined).squeeze(-1)
        return torch.sigmoid(logit)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 Cross + Deep 联合 embedding"""
        concat_embeds = self.embed_input(sparse_values)
        return self._embed_projection(concat_embeds)
