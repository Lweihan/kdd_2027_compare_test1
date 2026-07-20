#!/usr/bin/env python3
"""
run_compare.py — 对比实验主入口
=================================
用法:
  python run_compare.py --config configs/compare_config.yaml
  python run_compare.py --config configs/compare_config.yaml --fm_weights 0.1 0.5 1.0 2.0
  python run_compare.py --config configs/compare_config.yaml --models dnn deepfm
  python run_compare.py --config configs/compare_config.yaml --target last_click
  python run_compare.py --config configs/compare_config.yaml --data_path /path/to/MAC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import json
import time
from typing import List

from utils.config import CompareConfig
from utils.seed import set_seed
from data.mac_dataset import load_mac_data, get_mac_dataloaders
from models.dnn import DNN
from models.wide_deep import WideDeep
from models.deepfm import DeepFM
from models.dcn import DCN
from models.autoint import AutoInt
from trainers.compare_trainer import CompareTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 可用的精排模型
MODEL_REGISTRY = {
    "dnn": DNN,
    "wide_deep": WideDeep,
    "deepfm": DeepFM,
    "dcn": DCN,
    "autoint": AutoInt,
}

ALL_MODELS = list(MODEL_REGISTRY.keys())


def create_model(model_name: str, config: CompareConfig, sparse_feature_names, feature_voc_sizes):
    """根据名称创建精排模型"""
    model_config = getattr(config.models, model_name)
    model_cls = MODEL_REGISTRY[model_name]

    # 通用参数 (所有模型都接受)
    common_kwargs = {
        "sparse_feature_names": sparse_feature_names,
        "feature_voc_sizes": feature_voc_sizes,
        "embed_dim": config.dataset.embed_dim,
        "hidden_dims": model_config.hidden_dims,
        "dropout": model_config.dropout,
        "activation": model_config.activation,
    }

    # 模型特有参数
    if model_name == "dcn":
        common_kwargs["num_cross_layers"] = model_config.num_cross_layers
    elif model_name == "autoint":
        common_kwargs["num_attention_layers"] = model_config.num_attention_layers
        common_kwargs["num_heads"] = model_config.num_heads
        common_kwargs["attention_dim"] = model_config.attention_dim

    return model_cls(**common_kwargs)


def main():
    parser = argparse.ArgumentParser(description="FM-Optimized Embedding 对比实验")
    parser.add_argument("--config", type=str, default="configs/compare_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=ALL_MODELS,
                        help=f"要测试的模型列表 (默认全部: {ALL_MODELS})")
    parser.add_argument("--fm_weights", nargs="+", type=float, default=None,
                        help="FM 权重列表 (如: 0.1 0.5 1.0 2.0)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace 数据集名称")
    parser.add_argument("--target", type=str, default=None,
                        choices=["last_click", "first_click", "dda", "linear"],
                        help="目标归因机制")
    parser.add_argument("--data_path", type=str, default=None,
                        help="本地数据路径 (git clone 后的目录, 如: ./data)")
    parser.add_argument("--fm_checkpoint", type=str, default=None,
                        help="预训练 FM checkpoint 路径 (如 falcon 的 best_fm.pt)")
    parser.add_argument("--freeze_fm", action="store_true", default=False,
                        help="加载预训练 FM 时冻结 FM backbone (仅训练投影层+预测头)")
    parser.add_argument("--no_freeze_fm", action="store_true", default=False,
                        help="不冻结 FM backbone, 全参数微调")
    parser.add_argument("--no_fast_train", action="store_true", default=False,
                        help="禁用快速训练模式 (每步都运行 ODE, 非常慢)")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="禁用混合精度训练 (AMP)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="启用 torch.compile 加速 velocity_net (需 PyTorch 2.0+)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数 (用于调试)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda:0 / cpu / auto)")
    parser.add_argument("--multi_gpu", action="store_true", default=False,
                        help="启用多卡训练 (DataParallel)")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None,
                        help="指定 GPU IDs (如: --gpu_ids 0 1 2 3)")
    args = parser.parse_args()

    # 加载配置
    config = CompareConfig.from_yaml(args.config)

    # 命令行参数覆盖配置
    if args.models:
        selected_models = args.models
    else:
        selected_models = ALL_MODELS

    if args.dataset:
        config.dataset.name = args.dataset
    if args.target:
        config.dataset.target_attribution = args.target
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples:
        config.dataset.max_samples = args.max_samples
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    if args.multi_gpu:
        config.training.multi_gpu = True
    if args.gpu_ids is not None:
        config.training.gpu_ids = args.gpu_ids
    if args.fm_checkpoint:
        config.fm_optimizer.fm_checkpoint = args.fm_checkpoint
    if args.no_freeze_fm:
        config.fm_optimizer.freeze_fm = False
    elif args.freeze_fm:
        config.fm_optimizer.freeze_fm = True
    if args.no_fast_train:
        config.fm_optimizer.fast_train = False
    if args.no_amp:
        config.fm_optimizer.use_amp = False
    if args.compile:
        config.fm_optimizer.compile_velocity = True

    device = config.resolve_device()
    set_seed(config.seed)

    logger.info("=" * 70)
    logger.info("FM-Optimized Embedding 对比实验")
    logger.info("=" * 70)
    logger.info(f"数据集: {config.dataset.name}")
    logger.info(f"本地数据路径: {args.data_path or '无 (将从 HuggingFace 加载)'}")
    logger.info(f"目标归因: {config.dataset.target_attribution}")
    logger.info(f"精排模型: {selected_models}")
    logger.info(f"FM 权重: {args.fm_weights or '使用配置文件默认值'}")
    logger.info(f"FM checkpoint: {config.fm_optimizer.fm_checkpoint or '无 (从头训练)'}")
    if config.fm_optimizer.fm_checkpoint:
        logger.info(f"FM freeze: {config.fm_optimizer.freeze_fm}")
    logger.info(f"设备: {device}")
    logger.info(f"多卡训练: {config.training.multi_gpu}")
    if config.training.multi_gpu:
        logger.info(f"GPU IDs: {config.training.gpu_ids if config.training.gpu_ids else '全部可见 GPU'}")
    logger.info(f"随机种子: {config.seed}")
    logger.info(f"输出目录: {config.output_dir}")

    # ──────────────────────────────────────────
    # Step 1: 加载数据集
    # ──────────────────────────────────────────
    logger.info("\n[Step 1] 加载 MAC 数据集...")
    t0 = time.time()
    train_data, val_data, test_data = load_mac_data(
        dataset_name=config.dataset.name,
        target_attribution=config.dataset.target_attribution,
        max_samples=config.dataset.max_samples,
        data_path=args.data_path,
    )
    logger.info(f"数据加载耗时: {time.time() - t0:.1f}s")

    # 创建 DataLoader
    train_loader, val_loader, test_loader = get_mac_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
    )

    # 获取特征信息
    sparse_feature_names = train_data["sparse_feature_names"]
    feature_voc_sizes = train_data["feature_voc_sizes"]

    logger.info(f"稀疏特征数: {len(sparse_feature_names)}")
    logger.info(f"Embedding 维度: {config.dataset.embed_dim}")
    logger.info(f"特征词表大小: { {k: v for k, v in feature_voc_sizes.items()} }")

    # 更新 FM 优化器的维度配置
    # 精排模型的 embedding 输出维度 = hidden_dims[-1]
    embed_output_dim = config.models.dnn.hidden_dims[-1]

    if config.fm_optimizer.fm_checkpoint:
        # 使用预训练 FM checkpoint: 保持 FM 原始架构参数 (data_dim, cond_dim 等)
        # FMOptimizer 会自动添加 input_proj 将 ranker_dim → fm.data_dim
        logger.info(f"精排模型 embedding 输出维度: {embed_output_dim}")
        logger.info(f"FM checkpoint data_dim: {config.fm_optimizer.data_dim}")
        logger.info(f"  → 将自动添加投影层: {embed_output_dim} → {config.fm_optimizer.data_dim}")
    else:
        # 从头训练: FM data_dim 匹配精排模型 embedding 维度
        config.fm_optimizer.data_dim = embed_output_dim
        config.fm_optimizer.cond_dim = embed_output_dim
        logger.info(f"精排模型 embedding 输出维度: {embed_output_dim}")
        logger.info(f"FM 优化器 data_dim/cond_dim: {embed_output_dim}")

    # ──────────────────────────────────────────
    # Step 2: 运行对比实验
    # ──────────────────────────────────────────
    trainer = CompareTrainer(config, device=device)

    all_results = {}
    for model_name in selected_models:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# 开始对比实验: {model_name}")
        logger.info(f"{'#'*70}")

        # 创建模型
        model = create_model(model_name, config, sparse_feature_names, feature_voc_sizes)

        # 运行对比
        results = trainer.run_comparison(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            fm_weights=args.fm_weights,
        )
        all_results[model_name] = results

    # ──────────────────────────────────────────
    # Step 3: 汇总所有模型对比结果
    # ──────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("全部模型对比汇总")
    logger.info(f"{'='*70}")

    # 生成对比表格
    summary = {}
    for model_name, results in all_results.items():
        summary[model_name] = {
            'direct': results['direct'],
            'fm_optimized': results['fm_optimized'],
        }

        # 打印每个模型的对比
        logger.info(f"\n--- {model_name} ---")
        logger.info(f"  直接预测: {results['direct']}")
        for fm_key, metrics in results['fm_optimized'].items():
            logger.info(f"  FM优化 ({fm_key}): {metrics}")

    # 保存汇总结果
    summary_path = os.path.join(config.output_dir, "all_comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n汇总结果已保存: {summary_path}")

    # 打印最终对比表
    _print_final_table(all_results, args.fm_weights)


def _print_final_table(all_results: dict, fm_weights: List[float] = None):
    """打印最终对比表格"""
    if fm_weights is None:
        fm_weights = [0.1, 0.5, 1.0, 2.0]

    logger.info(f"\n{'='*80}")
    logger.info("最终对比表 (AUC)")
    logger.info(f"{'='*80}")

    header = f"{'Model':<15} {'Direct':>10}"
    for w in fm_weights:
        header += f" {'FM_w='+str(w):>12}"
    header += f" {'Best FM':>12} {'Improvement':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for model_name, results in all_results.items():
        direct_auc = results['direct'].get('auc', 0)
        row = f"{model_name:<15} {direct_auc:>10.6f}"

        best_fm_auc = 0
        for w in fm_weights:
            fm_key = f"fm_weight_{w}"
            fm_auc = results['fm_optimized'].get(fm_key, {}).get('auc', 0)
            row += f" {fm_auc:>12.6f}"
            if fm_auc > best_fm_auc:
                best_fm_auc = fm_auc

        improvement = best_fm_auc - direct_auc
        sign = "+" if improvement > 0 else ""
        row += f" {best_fm_auc:>12.6f} {sign}{improvement:>11.6f}"
        logger.info(row)

    # LogLoss 对比
    logger.info(f"\n最终对比表 (LogLoss)")
    logger.info(f"{'='*80}")

    header = f"{'Model':<15} {'Direct':>10}"
    for w in fm_weights:
        header += f" {'FM_w='+str(w):>12}"
    header += f" {'Best FM':>12} {'Improvement':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for model_name, results in all_results.items():
        direct_ll = results['direct'].get('logloss', 0)
        row = f"{model_name:<15} {direct_ll:>10.6f}"

        best_fm_ll = float('inf')
        for w in fm_weights:
            fm_key = f"fm_weight_{w}"
            fm_ll = results['fm_optimized'].get(fm_key, {}).get('logloss', 0)
            row += f" {fm_ll:>12.6f}"
            if fm_ll < best_fm_ll:
                best_fm_ll = fm_ll

        improvement = direct_ll - best_fm_ll  # 越低越好, 正数表示提升
        sign = "+" if improvement > 0 else ""
        row += f" {best_fm_ll:>12.6f} {sign}{improvement:>11.6f}"
        logger.info(row)


if __name__ == "__main__":
    main()
