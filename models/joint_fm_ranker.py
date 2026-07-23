"""
joint_fm_ranker.py — 联合训练模型
=====================================
将精排模型 (BaseRanker) + FM 优化器 (FMOptimizer) 封装为端到端联合训练模型。

训练模式:
- 两条预测路径同时训练:
  1. 直接预测: sparse_values → ranker → logit_direct
  2. FM 优化后: sparse_values → ranker.extract_embedding → FM → logit_fm
- 总损失 = α * L_direct + (1-α) * L_FM_total
  其中 L_FM_total = fm_weight * L_FM_velocity + (1-fm_weight) * L_BCE_fm

推理模式:
- 直接预测: model.predict_direct(sparse_values)
- FM 优化后: model.predict_fm(sparse_values)
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_ranker import BaseRanker
from models.fm_optimizer import FMOptimizer
from utils.config import FMOptimizerConfig

logger = logging.getLogger(__name__)


class JointFMRanker(nn.Module):
    """
    联合训练模型: 精排模型 + FM 优化器端到端联合训练。

    关键: ranker 的 embedding 层和 FM 的 velocity_net 同时接收梯度,
    使得 ranker 学会生成更适合 FM 优化的 embedding。
    """

    def __init__(
        self,
        ranker: BaseRanker,
        fm_config: FMOptimizerConfig,
        input_dim: int = None,
        fm_checkpoint: str = None,
        freeze_fm: bool = False,
        joint_alpha: float = 0.5,
    ):
        """
        Args:
            ranker: 精排模型实例
            fm_config: FM 优化器配置
            input_dim: 精排模型 embedding 输出维度 (自动添加投影层若不匹配)
            fm_checkpoint: 预训练 FM checkpoint 路径
            freeze_fm: 是否冻结 FM backbone (联合训练通常不冻结)
            joint_alpha: 直接预测损失权重 (0~1, 默认 0.5)
                L_total = α * L_direct + (1-α) * L_FM_total
        """
        super().__init__()
        self.ranker = ranker
        self.joint_alpha = joint_alpha

        # 获取 ranker embedding 维度
        if input_dim is not None:
            self.embed_output_dim = input_dim
        else:
            # 从 ranker 的 hidden_dims 推断
            self.embed_output_dim = ranker.hidden_dims[-1]

        # FM 优化器
        self.fm_optimizer = FMOptimizer(
            fm_config,
            input_dim=self.embed_output_dim,
            fm_checkpoint=fm_checkpoint if fm_checkpoint else None,
            freeze_fm=freeze_fm,
        )

    def forward(self, sparse_values, fast_train: bool = True):
        """
        联合前向传播, 返回两条路径的 logits。

        Args:
            sparse_values: {feat_name: (batch,)} 稀疏特征
            fast_train: FM 快速训练模式

        Returns:
            dict: {
                'direct_logit': 直接预测 logit,
                'fm_logit': FM 优化后预测 logit,
                'fm_loss_dict': FM 内部损失详情,
                'embeddings': ranker 提取的 embedding,
            }
        """
        # 1) 直接预测
        direct_logit = self.ranker(sparse_values)  # (batch,)

        # 2) 提取 embedding → FM 优化 → 预测
        embeddings = self.ranker.extract_embedding(sparse_values)  # (batch, embed_dim)

        # FM: 投影 + velocity loss
        projected = self.fm_optimizer.project_input(embeddings)
        fm_loss_dict = self.fm_optimizer.fm_model.compute_loss(
            x_clean=projected,
            condition=projected,
            time_weight_mode=self.fm_optimizer.config.time_weight_mode,
            time_weight_scale=self.fm_optimizer.config.time_weight_scale,
        )

        if fast_train:
            fm_logit = self.fm_optimizer.pred_head(projected).squeeze(-1)
        else:
            optimized_emb = self.fm_optimizer.fm_model.optimize_embedding(
                projected,
                delta_t=self.fm_optimizer.config.train_delta_t,
                num_steps=self.fm_optimizer.config.train_ode_steps,
            )
            fm_logit = self.fm_optimizer.pred_head(optimized_emb).squeeze(-1)

        return {
            'direct_logit': direct_logit,
            'fm_logit': fm_logit,
            'fm_loss_dict': fm_loss_dict,
            'embeddings': embeddings,
        }

    def compute_joint_loss(self, sparse_values, labels, fast_train: bool = True):
        """
        计算联合训练总损失。

        L_total = α * L_direct + (1-α) * [fm_weight * L_FM + (1-fm_weight) * L_BCE_fm]

        Args:
            sparse_values: {feat_name: (batch,)}
            labels: (batch,)
            fast_train: FM 快速训练模式

        Returns:
            dict with: loss, direct_loss, fm_total_loss, fm_velocity_loss, bce_fm_loss, bce_direct_loss
        """
        fwd = self.forward(sparse_values, fast_train=fast_train)
        direct_logit = fwd['direct_logit']
        fm_logit = fwd['fm_logit']
        fm_loss_dict = fwd['fm_loss_dict']

        # 直接预测 BCE
        bce_direct = F.binary_cross_entropy_with_logits(direct_logit, labels)

        # FM 预测 BCE
        bce_fm = F.binary_cross_entropy_with_logits(fm_logit, labels)

        # FM 速度损失
        fm_velocity_loss = fm_loss_dict['loss']

        # FM 总损失
        fm_weight = self.fm_optimizer.fm_weight
        fm_total = fm_weight * fm_velocity_loss + (1 - fm_weight) * bce_fm

        # 联合总损失
        total_loss = self.joint_alpha * bce_direct + (1 - self.joint_alpha) * fm_total

        return {
            'loss': total_loss,
            'direct_loss': bce_direct.item(),
            'fm_total_loss': fm_total.item(),
            'fm_velocity_loss': fm_velocity_loss.item(),
            'bce_fm': bce_fm.item(),
            'bce_direct': bce_direct.item(),
        }

    @torch.no_grad()
    def predict_direct(self, sparse_values) -> torch.Tensor:
        """直接预测: 返回 sigmoid 概率"""
        logit = self.ranker(sparse_values)
        return torch.sigmoid(logit)

    @torch.no_grad()
    def predict_fm(self, sparse_values, num_steps: int = 50, delta_t: float = 0.5) -> torch.Tensor:
        """FM 优化后预测: 返回 sigmoid 概率"""
        embeddings = self.ranker.extract_embedding(sparse_values)
        pred = self.fm_optimizer(embeddings, num_steps=num_steps, delta_t=delta_t)
        return pred

    @torch.no_grad()
    def evaluate_direct(self, sparse_values, labels) -> dict:
        """评估直接预测路径"""
        logit = self.ranker(sparse_values)
        loss = F.binary_cross_entropy_with_logits(logit, labels)
        pred = torch.sigmoid(logit)
        return {'loss': loss.item(), 'pred': pred}

    @torch.no_grad()
    def evaluate_fm(self, sparse_values, labels) -> dict:
        """评估 FM 优化后预测路径"""
        embeddings = self.ranker.extract_embedding(sparse_values)
        pred = self.fm_optimizer(embeddings)
        loss = F.binary_cross_entropy(pred, labels)
        return {'loss': loss.item(), 'pred': pred}
