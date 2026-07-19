"""
autoint.py — AutoInt 精排模型
=================================
AAAI 2019: 基于 Multi-Head Self-Attention 的自动特征交互模型
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .base_ranker import BaseRanker


class MultiHeadAttentionInteracting(nn.Module):
    """
    Multi-Head Self-Attention 特征交互层。
    
    将每个特征的 embedding 视为一个 token，
    通过 Multi-Head Attention 学习特征之间的交互。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        attention_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim

        # Q, K, V 投影
        self.W_q = nn.Linear(embed_dim, num_heads * attention_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, num_heads * attention_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, num_heads * attention_dim, bias=False)

        # 输出投影
        self.W_out = nn.Linear(num_heads * attention_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features, embed_dim) 特征 embedding 序列
        
        Returns:
            out: (batch, num_features, embed_dim) 交互后的特征表示
        """
        batch_size, num_features, _ = x.shape

        # Q, K, V: (batch, num_features, num_heads * attention_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape: (batch, num_features, num_heads, attention_dim) → (batch, num_heads, num_features, attention_dim)
        Q = Q.view(batch_size, num_features, self.num_heads, self.attention_dim).transpose(1, 2)
        K = K.view(batch_size, num_features, self.num_heads, self.attention_dim).transpose(1, 2)
        V = V.view(batch_size, num_features, self.num_heads, self.attention_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = self.attention_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, num_heads, num_features, num_features)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_out = torch.matmul(attn_weights, V)  # (batch, num_heads, num_features, attention_dim)

        # Concat heads: (batch, num_features, num_heads * attention_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_features, -1)

        # Output projection + residual + layer norm
        out = self.W_out(attn_out)
        out = self.dropout(out)
        out = self.layer_norm(out + x)

        return out


class AutoInt(BaseRanker):
    """
    AutoInt 精排模型。
    
    架构:
    - Embedding 层: 稀疏特征 → embedding
    - Attention 交互层: Multi-Head Self-Attention × N
    - 输出层: Attention 输出 → Flatten → Linear → Sigmoid
    """

    def __init__(
        self,
        sparse_feature_names: List[str],
        feature_voc_sizes: Dict[str, int],
        embed_dim: int = 16,
        hidden_dims: List[int] = None,
        num_attention_layers: int = 3,
        num_heads: int = 4,
        attention_dim: int = 64,
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
        self.num_attention_layers = num_attention_layers

        # Attention 交互层
        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionInteracting(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attention_dim=attention_dim,
                dropout=dropout,
            )
            for _ in range(num_attention_layers)
        ])

        # 输出层: Flatten attention 输出 → Linear
        # Attention 输出: (batch, num_features, embed_dim) → Flatten → (batch, num_features * embed_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(self.concat_embed_dim, 1),
        )

        # 用于提取 embedding
        self._embed_projection = nn.Linear(self.concat_embed_dim, hidden_dims[-1])

    def embed_input_stacked(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将稀疏特征 embedding 堆叠为序列形式 (用于 Attention)。
        
        Returns:
            embed_stack: (batch, num_features, embed_dim)
        """
        embed_list = []
        for feat_name in self.sparse_feature_names:
            emb = self.embeddings[feat_name](sparse_values[feat_name])  # (batch, embed_dim)
            embed_list.append(emb)
        return torch.stack(embed_list, dim=1)  # (batch, num_features, embed_dim)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Embedding → Stack
        x = self.embed_input_stacked(sparse_values)  # (batch, num_features, embed_dim)

        # Attention 交互层
        for attention_layer in self.attention_layers:
            x = attention_layer(x)  # (batch, num_features, embed_dim)

        # Flatten
        x_flat = x.view(x.size(0), -1)  # (batch, num_features * embed_dim)

        # 输出
        logit = self.output_layer(x_flat).squeeze(-1)  # (batch,)
        return torch.sigmoid(logit)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 Attention 交互后的 embedding"""
        x = self.embed_input_stacked(sparse_values)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        x_flat = x.view(x.size(0), -1)
        return self._embed_projection(x_flat)
