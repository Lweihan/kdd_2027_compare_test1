"""
base_ranker.py — 精排模型基类
=================================
所有精排模型继承此类，统一接口。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class BaseRanker(nn.Module):
    """
    精排模型基类。
    
    所有精排模型共享:
    - Embedding 层 (稀疏特征编码)
    - 统一的 forward / predict / extract_embedding 接口
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
        super().__init__()
        self.sparse_feature_names = sparse_feature_names
        self.feature_voc_sizes = feature_voc_sizes
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout_rate = dropout
        self.activation_name = activation

        # 创建 Embedding 层
        self.embeddings = nn.ModuleDict({
            feat_name: nn.Embedding(
                num_embeddings=feature_voc_sizes[feat_name],
                embedding_dim=embed_dim,
                padding_idx=0,
            )
            for feat_name in sparse_feature_names
        })

        # 计算拼接后的 embedding 维度
        self.concat_embed_dim = len(sparse_feature_names) * embed_dim

        # 激活函数
        self.activation = self._get_activation(activation)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}, choose from {list(activations.keys())}")
        return activations[name]

    def embed_input(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将稀疏特征值通过 Embedding 层映射为稠密向量并拼接。
        
        Args:
            sparse_values: {feat_name: (batch,)} 各稀疏特征的整数编码值
        
        Returns:
            concat_embeds: (batch, num_sparse * embed_dim) 拼接后的 embedding
        """
        embed_list = []
        for feat_name in self.sparse_feature_names:
            emb = self.embeddings[feat_name](sparse_values[feat_name])  # (batch, embed_dim)
            embed_list.append(emb)
        return torch.cat(embed_list, dim=1)  # (batch, num_sparse * embed_dim)

    def forward(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播 (子类必须实现)。
        
        Args:
            sparse_values: {feat_name: (batch,)} 各稀疏特征的整数编码值
        
        Returns:
            prediction: (batch, 1) 预测概率
        """
        raise NotImplementedError

    def predict(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测接口，返回 sigmoid 后的概率"""
        return self.forward(sparse_values)

    def extract_embedding(self, sparse_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        提取模型中间层 embedding (用于 FM 优化)。
        
        默认返回 embedding 拼接后的向量，子类可以覆盖以返回更有意义的中间表示。
        
        Args:
            sparse_values: {feat_name: (batch,)} 各稀疏特征的整数编码值
        
        Returns:
            embedding: (batch, embed_dim_output) 模型中间层 embedding
        """
        raise NotImplementedError("子类必须实现 extract_embedding 方法")
