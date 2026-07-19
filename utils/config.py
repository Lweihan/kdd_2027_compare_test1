"""
config.py — 统一配置管理
=========================
支持从 YAML 文件加载，管理对比实验的所有配置。
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "alimamaTech/MAC"
    target_attribution: str = "last_click"
    max_samples: int = -1
    train_ratio: float = 0.8
    batch_size: int = 4096
    num_workers: int = 4
    sparse_features: List[str] = field(default_factory=list)
    dense_features: List[str] = field(default_factory=list)
    embed_dim: int = 16
    sequence_features: bool = False


@dataclass
class ModelConfig:
    """单个精排模型配置"""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    activation: str = "relu"
    # DCN specific
    num_cross_layers: int = 3
    # AutoInt specific
    num_attention_layers: int = 3
    num_heads: int = 4
    attention_dim: int = 64


@dataclass
class ModelsConfig:
    """所有精排模型配置"""
    dnn: ModelConfig = field(default_factory=ModelConfig)
    wide_deep: ModelConfig = field(default_factory=ModelConfig)
    deepfm: ModelConfig = field(default_factory=ModelConfig)
    dcn: ModelConfig = field(default_factory=lambda: ModelConfig(num_cross_layers=3))
    autoint: ModelConfig = field(default_factory=lambda: ModelConfig(
        num_attention_layers=3, num_heads=4, attention_dim=64
    ))


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 1
    lr: float = 0.003
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    early_stop_patience: int = 2
    save_every: int = 1


@dataclass
class FMOptimizerConfig:
    """FM (Flow Matching) 优化器配置 (改编自 falcon)"""
    # Flow Matching 模型架构
    data_dim: int = 64
    time_dim: int = 64
    cond_dim: int = 64
    backbone_type: str = "transformer"
    # MLP backbone
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    # Transformer backbone
    transformer_dim: int = 256
    transformer_depth: int = 4
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.1
    # ODE 采样
    sample_steps: int = 50
    sample_method: str = "euler"
    # 时间加权策略 (来自 falcon)
    time_weight_mode: str = "mid"
    time_weight_scale: float = 1.0
    # FM 权重 (关键参数: 控制 FM 优化对 embedding 的影响程度)
    fm_weight: float = 1.0
    # FM 优化后的预测头
    pred_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    pred_dropout: float = 0.1
    # FM 训练
    fm_epochs: int = 5
    fm_lr: float = 1e-3
    fm_weight_decay: float = 1e-5
    fm_grad_clip: float = 1.0
    fm_batch_size: int = 4096
    fm_ema_loss_alpha: float = 0.1
    # 预训练 FM checkpoint (来自 falcon)
    fm_checkpoint: str = ""
    freeze_fm: bool = True


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ["auc", "gauc", "logloss", "ndcg"])
    ndcg_k: int = 10


@dataclass
class CompareConfig:
    """对比实验全局配置"""
    seed: int = 42
    output_dir: str = "./output"
    device: str = "auto"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fm_optimizer: FMOptimizerConfig = field(default_factory=FMOptimizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str) -> "CompareConfig":
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        if d is None:
            d = {}
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "CompareConfig":
        dataset = DatasetConfig(**d.get("dataset", {}))
        models_d = d.get("models", {})
        models = ModelsConfig(
            dnn=ModelConfig(**models_d.get("dnn", {})),
            wide_deep=ModelConfig(**models_d.get("wide_deep", {})),
            deepfm=ModelConfig(**models_d.get("deepfm", {})),
            dcn=ModelConfig(**models_d.get("dcn", {})),
            autoint=ModelConfig(**models_d.get("autoint", {})),
        )
        training = TrainingConfig(**d.get("training", {}))
        fm_optimizer = FMOptimizerConfig(**d.get("fm_optimizer", {}))
        evaluation = EvaluationConfig(**d.get("evaluation", {}))
        return cls(
            seed=d.get("seed", 42),
            output_dir=d.get("output_dir", "./output"),
            device=d.get("device", "auto"),
            dataset=dataset,
            models=models,
            training=training,
            fm_optimizer=fm_optimizer,
            evaluation=evaluation,
        )
