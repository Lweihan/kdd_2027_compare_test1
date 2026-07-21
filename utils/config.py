"""
config.py — 统一配置管理
=========================
支持从 YAML 文件加载，管理对比实验的所有配置。
"""

import os
import yaml
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Dict


def _coerce_types(dataclass_cls, kwargs: dict) -> dict:
    """
    将 dict 中的值强制转换为 dataclass 字段声明的类型。
    解决 PyYAML 将 '1e-5' 解析为字符串而非浮点数的问题。
    """
    coerced = {}
    type_map = {int: int, float: float, str: str, bool: bool}
    field_types = {f.name: f.type for f in fields(dataclass_cls)}

    for key, value in kwargs.items():
        if key in field_types:
            target_type = type_map.get(field_types[key])
            if target_type and value is not None:
                try:
                    value = target_type(value)
                except (ValueError, TypeError):
                    pass  # 保留原始值
        coerced[key] = value
    return coerced


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
    # 多卡训练
    multi_gpu: bool = False
    gpu_ids: List[int] = field(default_factory=list)


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
    # 训练加速
    fast_train: bool = True             # 快速训练: 跳过 ODE, 用 projected embedding 直接预测
    train_delta_t: float = 0.3          # 训练时 ODE 步长 (fast_train=False 时生效)
    train_ode_steps: int = 10            # 训练时 ODE 步数 (fast_train=False 时生效)
    eval_interval: int = 1              # 评估间隔 (每 N 个 epoch 评估一次)
    use_amp: bool = False               # 混合精度训练 (AMP)
    compile_velocity: bool = False       # torch.compile 加速 velocity_net
    compile_full: bool = False           # torch.compile 加速整个 FMOptimizer
    grad_accum_steps: int = 1           # 梯度累积步数 (增大等效 batch_size)
    gradient_checkpointing: bool = False # 梯度检查点 (省显存, 慢一点)


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
        dataset = DatasetConfig(**_coerce_types(DatasetConfig, d.get("dataset", {})))
        models_d = d.get("models", {})
        models = ModelsConfig(
            dnn=ModelConfig(**_coerce_types(ModelConfig, models_d.get("dnn", {}))),
            wide_deep=ModelConfig(**_coerce_types(ModelConfig, models_d.get("wide_deep", {}))),
            deepfm=ModelConfig(**_coerce_types(ModelConfig, models_d.get("deepfm", {}))),
            dcn=ModelConfig(**_coerce_types(ModelConfig, models_d.get("dcn", {}))),
            autoint=ModelConfig(**_coerce_types(ModelConfig, models_d.get("autoint", {}))),
        )
        training = TrainingConfig(**_coerce_types(TrainingConfig, d.get("training", {})))
        fm_optimizer = FMOptimizerConfig(**_coerce_types(FMOptimizerConfig, d.get("fm_optimizer", {})))
        evaluation = EvaluationConfig(**_coerce_types(EvaluationConfig, d.get("evaluation", {})))
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
