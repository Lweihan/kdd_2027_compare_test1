"""
fm_optimizer.py — FM (Flow Matching) Embedding 优化器
=======================================================
改编自 falcon 的 Flow Matching Teacher 模型，用于优化精排模型的 embedding。

核心思想:
1. 精排模型提取初始 embedding → FM 学习 embedding 空间的流形结构
2. FM 通过 ODE 求解生成 "优化" 后的 embedding
3. 优化的 embedding 更适合下游 CTR/CVR 预测任务

与 falcon 的对应关系:
- falcon: condition(当前 embedding) → FM → 预测未来 embedding
- 本模块: condition(精排模型 embedding) → FM → 优化的 embedding → CTR 预测

FM 权重 (fm_weight) 控制流匹配损失与预测损失的平衡:
- fm_weight ↑: 更关注 embedding 空间的流形结构
- fm_weight ↓: 更关注下游预测任务
"""

import math
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from utils.config import FMOptimizerConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# Flow Matching 核心数学 (来自 falcon/fm/flow_core.py)
# ──────────────────────────────────────────────────

def flow_match_forward(
    x_clean: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor = None,
) -> tuple:
    """
    OT-CFM 前向过程 (来自 falcon):
        x_t = (1 - t) * x_clean + t * noise
        v_t = noise - x_clean

    Args:
        x_clean: (batch, dim) 干净 embedding
        t: (batch,) 时间步 [0, 1]
        noise: (batch, dim) 噪声

    Returns:
        x_t: (batch, dim) 加噪后状态
        v_t: (batch, dim) 真实速度场
    """
    if noise is None:
        noise = torch.randn_like(x_clean)
    t = t.view(-1, 1)
    x_t = (1.0 - t) * x_clean + t * noise
    v_t = noise - x_clean
    return x_t, v_t


# ──────────────────────────────────────────────────
# 时间编码 (来自 falcon/fm/velocity_net.py)
# ──────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码 (来自 falcon)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ──────────────────────────────────────────────────
# 速度场网络 (改编自 falcon/fm/velocity_net.py)
# ──────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """残差块 (来自 falcon)"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class VelocityNetwork(nn.Module):
    """
    速度场预测网络 v_θ(x_t, t, condition) — MLP 版本 (来自 falcon)。
    """

    def __init__(
        self,
        data_dim: int = 64,
        time_dim: int = 64,
        cond_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(),
        )

        input_dim = data_dim + time_dim + hidden_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, data_dim)])

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        batch_size = x_t.size(0)
        device = x_t.device

        time_emb = self.time_embed(t)
        cond_emb = self.cond_proj(condition) if condition is not None else \
            torch.zeros(batch_size, self.cond_proj[0].out_features, device=device)

        h = torch.cat([x_t, time_emb, cond_emb], dim=-1)
        return self.mlp(h)


class TransformerVelocityNetwork(nn.Module):
    """
    速度场预测网络 v_θ(x_t, t, condition) — Transformer 版本 (来自 falcon)。
    """

    def __init__(
        self,
        data_dim: int = 64,
        time_dim: int = 64,
        cond_dim: int = 64,
        transformer_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.transformer_dim = transformer_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, transformer_dim), nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )

        self.x_proj = nn.Linear(data_dim, transformer_dim)
        self.cond_proj = nn.Linear(cond_dim, transformer_dim)

        self.type_embed = nn.Parameter(torch.randn(3, transformer_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
            norm=nn.LayerNorm(transformer_dim),
        )

        self.norm_out = nn.LayerNorm(transformer_dim)
        self.head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        batch_size = x_t.size(0)
        device = x_t.device

        x_tok = self.x_proj(x_t)
        t_tok = self.time_embed(t)
        c_tok = self.cond_proj(condition) if condition is not None else \
            torch.zeros(batch_size, self.transformer_dim, device=device)

        tokens = torch.stack([x_tok, t_tok, c_tok], dim=1)
        tokens = tokens + self.type_embed.unsqueeze(0)

        h = self.transformer(tokens)
        out = self.norm_out(h[:, 0])
        v_pred = self.head(out)

        return v_pred


# ──────────────────────────────────────────────────
# Flow Matching 模型 (改编自 falcon/fm/model.py)
# ──────────────────────────────────────────────────

class FlowMatchingModel(nn.Module):
    """
    Flow Matching Embedding 优化模型 (改编自 falcon)。
    
    训练: compute_loss(x_clean, condition, time_weight_mode, time_weight_scale)
    推理: sample(condition) / optimize_embedding(condition, num_steps)
    """

    def __init__(
        self,
        data_dim: int = 64,
        time_dim: int = 64,
        cond_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        backbone_type: str = "transformer",
        transformer_dim: int = 256,
        transformer_depth: int = 4,
        transformer_heads: int = 4,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.backbone_type = backbone_type

        if backbone_type == "transformer":
            self.velocity_net = TransformerVelocityNetwork(
                data_dim=data_dim, time_dim=time_dim, cond_dim=cond_dim,
                transformer_dim=transformer_dim, depth=transformer_depth,
                heads=transformer_heads, mlp_ratio=transformer_mlp_ratio,
                dropout=transformer_dropout,
            )
        elif backbone_type == "mlp":
            self.velocity_net = VelocityNetwork(
                data_dim=data_dim, time_dim=time_dim, cond_dim=cond_dim,
                hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

    @staticmethod
    def compute_time_weights(t: torch.Tensor, mode: str = "mid", scale: float = 1.0) -> torch.Tensor:
        """
        计算时间步加权函数 w(t) (来自 falcon)。
        """
        if mode == "none" or scale == 0.0:
            return torch.ones_like(t)

        if mode == "mid":
            w = 1.0 + scale * 4.0 * t * (1.0 - t)
        elif mode == "late":
            w = 1.0 + scale * 2.0 * t
        elif mode == "early":
            w = 1.0 + scale * 2.0 * (1.0 - t)
        else:
            raise ValueError(f"Unknown time_weight_mode: {mode}")

        return w

    def compute_loss(
        self,
        x_clean: torch.Tensor,
        condition: torch.Tensor = None,
        time_weight_mode: str = "mid",
        time_weight_scale: float = 1.0,
    ) -> dict:
        """
        Flow Matching 训练损失: w(t) * ||v_pred - v_t||^2 (来自 falcon)。
        """
        batch_size = x_clean.size(0)
        device = x_clean.device

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(x_clean)
        x_t, v_t = flow_match_forward(x_clean, t, noise)
        v_pred = self.velocity_net(x_t, t, condition)

        per_sample_mse = F.mse_loss(v_pred, v_t, reduction='none').mean(dim=-1)
        mse_raw = per_sample_mse.mean()

        w = self.compute_time_weights(t, mode=time_weight_mode, scale=time_weight_scale)
        mse_weighted = (w * per_sample_mse).mean()

        return {
            'loss': mse_weighted,
            'mse_raw': mse_raw.item(),
            'mse_weighted': mse_weighted.item(),
            'time_weight_mean': w.mean().item(),
            't_mean': t.mean().item(),
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        method: str = 'euler',
        device: str = 'cpu',
    ) -> torch.Tensor:
        """从噪声生成优化 embedding (ODE: t=1→0) (来自 falcon)"""
        batch_size = condition.size(0)
        x = torch.randn(batch_size, self.data_dim, device=device)
        dt = 1.0 / num_steps

        if method == 'euler':
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((batch_size,), t_val, device=device)
                v = self.velocity_net(x, t, condition)
                x = x - v * dt
        elif method == 'rk4':
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((batch_size,), t_val, device=device)
                k1 = self.velocity_net(x, t, condition)
                k2 = self.velocity_net(x - 0.5 * dt * k1, t - 0.5 * dt, condition)
                k3 = self.velocity_net(x - 0.5 * dt * k2, t - 0.5 * dt, condition)
                k4 = self.velocity_net(x - dt * k3, t - dt, condition)
                x = x - (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError(f"Unknown ODE method: {method}")
        return x

    @torch.no_grad()
    def optimize_embedding(
        self,
        condition: torch.Tensor,
        delta_t: float = 0.5,
        num_steps: int = 50,
        method: str = 'euler',
    ) -> torch.Tensor:
        """
        优化 embedding (改编自 falcon 的 predict_future_embedding)。
        
        从精排模型的 embedding (condition) 出发，通过 Flow Matching 生成优化后的 embedding。
        
        Args:
            condition: 精排模型的 embedding, (batch, data_dim)
            delta_t: 流动步长 [0, 1]
            num_steps: ODE 步数
            method: euler / rk4
        
        Returns:
            optimized_emb: (batch, data_dim) 优化后的 embedding
        """
        batch_size = condition.size(0)
        device = condition.device

        noise = torch.randn_like(condition)
        x_t = (1.0 - delta_t) * condition + delta_t * noise

        dt = delta_t / num_steps
        for step in range(num_steps):
            t_val = delta_t - step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.velocity_net(x_t, t, condition)
            x_t = x_t - v * dt
        return x_t


# ──────────────────────────────────────────────────
# FM Optimizer 完整封装
# ──────────────────────────────────────────────────

class FMOptimizer(nn.Module):
    """
    FM (Flow Matching) Embedding 优化器。
    
    完整流程:
    1. 输入: 精排模型的 embedding (condition)
    2. [可选] 输入投影: ranker_dim → data_dim (当精排模型 embedding 维度与 FM data_dim 不匹配时)
    3. FM 模型通过 ODE 求解生成优化 embedding
    4. 预测头将优化 embedding 映射为 CTR/CVR 预测
    
    训练损失:
    L_total = fm_weight * L_FM + (1 - fm_weight) * L_BCE
    
    支持加载预训练 FM checkpoint:
    - 提供 fm_checkpoint 路径时, 加载 falcon 预训练的 FM 权重
    - FM backbone 可选择冻结 (freeze_fm=True), 仅训练投影层和预测头
    """

    def __init__(self, config: FMOptimizerConfig, input_dim: int = None,
                 fm_checkpoint: str = None, freeze_fm: bool = True):
        """
        Args:
            config: FM 优化器配置
            input_dim: 精排模型 embedding 的输出维度 (如果与 data_dim 不同, 自动添加投影层)
            fm_checkpoint: 预训练 FM checkpoint 路径 (如 falcon 的 best_fm.pt)
            freeze_fm: 加载预训练 FM 时是否冻结 FM backbone (仅训练投影层+预测头)
        """
        super().__init__()
        self.config = config
        self.fm_weight = config.fm_weight

        # ── 输入投影层 ──
        # 当精排模型的 embedding 维度 != FM 的 data_dim 时, 需要投影层
        self.input_dim = input_dim if input_dim is not None else config.data_dim
        if self.input_dim != config.data_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, config.data_dim),
                nn.LayerNorm(config.data_dim),
                nn.ReLU(),
            )
            logger.info(f"  输入投影层: {self.input_dim} → {config.data_dim}")
        else:
            self.input_proj = nn.Identity()

        # ── Flow Matching 模型 (改编自 falcon) ──
        self.fm_model = FlowMatchingModel(
            data_dim=config.data_dim,
            time_dim=config.time_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            backbone_type=config.backbone_type,
            transformer_dim=config.transformer_dim,
            transformer_depth=config.transformer_depth,
            transformer_heads=config.transformer_heads,
            transformer_mlp_ratio=config.transformer_mlp_ratio,
            transformer_dropout=config.transformer_dropout,
        )

        # ── 加载预训练 FM checkpoint ──
        self.pretrained_fm = False
        if fm_checkpoint and os.path.exists(fm_checkpoint):
            self._load_fm_checkpoint(fm_checkpoint, freeze_fm)
            self.pretrained_fm = True

        # ── 预测头: 优化 embedding → CTR/CVR ──
        pred_layers = []
        input_dim = config.data_dim
        for hidden_dim in config.pred_hidden_dims:
            pred_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.pred_dropout),
            ])
            input_dim = hidden_dim
        pred_layers.append(nn.Linear(input_dim, 1))
        self.pred_head = nn.Sequential(*pred_layers)

    def _load_fm_checkpoint(self, checkpoint_path: str, freeze_fm: bool = True):
        """
        加载预训练 FM checkpoint (兼容 falcon 格式)。
        
        falcon checkpoint 格式:
        {
            'model_state_dict': {...},  # FlowMatchingModel 的 state_dict
            'epoch': int,
            'loss': float,
            ...
        }
        """
        logger.info(f"  加载预训练 FM checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 兼容不同 checkpoint 格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 加载到 fm_model
        # 需要处理 key 前缀: falcon 保存的是 FlowMatchingModel 的直接 state_dict
        # 而 FMOptimizer 中 fm_model 的 key 前缀是 'fm_model.'
        fm_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('fm_model.'):
                # 如果已经有前缀, 直接使用
                fm_state_dict[key] = value
            else:
                # 如果没有前缀, 添加 'fm_model.' 前缀
                fm_state_dict[f'fm_model.{key}'] = value

        # 尝试加载 (允许部分匹配)
        load_result = self.load_state_dict(fm_state_dict, strict=False)
        loaded_keys = len([k for k in self.state_dict().keys() if k.startswith('fm_model.')])
        matched_keys = loaded_keys - len(load_result.missing_keys) if load_result.missing_keys else loaded_keys
        logger.info(f"  FM checkpoint 加载: {matched_keys} 个匹配参数")

        if load_result.missing_keys:
            missing_fm_keys = [k for k in load_result.missing_keys if k.startswith('fm_model.')]
            if missing_fm_keys:
                logger.warning(f"  FM 缺失 keys: {missing_fm_keys[:5]}...")
        if load_result.unexpected_keys:
            logger.warning(f"  FM 多余 keys: {list(load_result.unexpected_keys)[:5]}...")

        # 冻结 FM backbone
        if freeze_fm:
            for param in self.fm_model.parameters():
                param.requires_grad = False
            logger.info(f"  FM backbone 已冻结 (freeze_fm=True)")
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"  可训练参数量 (仅投影层+预测头): {trainable:,}")
        else:
            logger.info(f"  FM backbone 未冻结, 全参数微调")

    def project_input(self, embeddings: torch.Tensor) -> torch.Tensor:
        """将精排模型 embedding 投影到 FM 的 data_dim"""
        return self.input_proj(embeddings)

    def compute_total_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        fast_train: bool = True,
    ) -> dict:
        """
        计算 FM 优化器的总损失。
        
        L_total = fm_weight * L_FM + (1 - fm_weight) * L_BCE
        
        Args:
            embeddings: 精排模型提取的 embedding (batch, input_dim)
            labels: CTR/CVR 标签 (batch,)
            fast_train: 快速训练模式 (默认 True)
                - True: 训练时跳过 ODE 求解, 用 projected embedding 直接通过 pred_head
                        仅计算 FM velocity loss + 直接预测 BCE loss
                - False: 每步都运行 ODE 求解 (慢, 但更精确)
        
        Returns:
            dict with: loss, fm_loss, bce_loss
        """
        # 投影到 FM data_dim
        projected = self.project_input(embeddings)

        # FM 流匹配损失 (velocity prediction loss, 无需 ODE)
        fm_loss_dict = self.fm_model.compute_loss(
            x_clean=projected,
            condition=projected,
            time_weight_mode=self.config.time_weight_mode,
            time_weight_scale=self.config.time_weight_scale,
        )
        fm_loss = fm_loss_dict['loss']

        if fast_train:
            # ── 快速训练: 跳过 ODE, 直接用 projected embedding 预测 ──
            # 理由: ODE 求解每步需要 N 次 velocity_net forward (N=20)
            #        在 FM backbone 冻结时, ODE 梯度不回传到 FM
            #        pred_head 的梯度通过 projected 直传即可
            logit = self.pred_head(projected).squeeze(-1)
        else:
            # ── 完整训练: 运行 ODE 求解 ──
            optimized_emb = self.fm_model.optimize_embedding(
                projected,
                delta_t=self.config.train_delta_t,
                num_steps=self.config.train_ode_steps,
            )
            logit = self.pred_head(optimized_emb).squeeze(-1)

        # BCE 损失 (使用 with_logits 版本, AMP 安全)
        bce_loss = F.binary_cross_entropy_with_logits(logit, labels)

        # 总损失
        total_loss = self.fm_weight * fm_loss + (1 - self.fm_weight) * bce_loss

        return {
            'loss': total_loss,
            'fm_loss': fm_loss.item(),
            'bce_loss': bce_loss.item(),
            'mse_raw': fm_loss_dict['mse_raw'],
            'mse_weighted': fm_loss_dict['mse_weighted'],
        }

    def forward(
        self,
        embeddings: torch.Tensor,
        num_steps: int = 50,
        delta_t: float = 0.5,
    ) -> torch.Tensor:
        """
        完整推理: embedding → 投影 → FM 优化 → CTR 预测
        
        Args:
            embeddings: 精排模型提取的 embedding (batch, input_dim)
            num_steps: ODE 步数
            delta_t: 流动步长
        
        Returns:
            prediction: (batch,) CTR/CVR 预测概率
        """
        # Step 1: 投影到 FM data_dim
        projected = self.project_input(embeddings)

        # Step 2: FM 优化 embedding
        optimized_emb = self.fm_model.optimize_embedding(
            projected,
            delta_t=delta_t,
            num_steps=num_steps,
        )

        # Step 3: 预测头
        logit = self.pred_head(optimized_emb).squeeze(-1)
        return torch.sigmoid(logit)

    def extract_optimized_embedding(
        self,
        embeddings: torch.Tensor,
        num_steps: int = 50,
        delta_t: float = 0.5,
    ) -> torch.Tensor:
        """
        仅提取 FM 优化后的 embedding (不进行预测)。
        """
        projected = self.project_input(embeddings)
        return self.fm_model.optimize_embedding(
            projected, delta_t=delta_t, num_steps=num_steps
        )
