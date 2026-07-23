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
import numpy as np
from typing import List

from utils.config import CompareConfig
from utils.seed import set_seed
from utils.metrics import evaluate_predictions
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


def _json_default(obj):
    """JSON 序列化回退: 处理 numpy / torch 数值类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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
    parser.add_argument("--batch_size", type=int, default=None,
                        help="统一 batch_size (同时覆盖 ranker 和 FM optimizer)")
    parser.add_argument("--fm_batch_size", type=int, default=None,
                        help="FM optimizer batch_size (单独指定, 优先于 --batch_size)")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                        help="梯度累积步数 (增大等效 batch_size 而不增加显存)")
    parser.add_argument("--compile_model", action="store_true", default=False,
                        help="启用 torch.compile 加速整个 FMOptimizer (比 --compile 更激进)")
    parser.add_argument("--joint_train", action="store_true", default=False,
                        help="启用联合训练模式 (ranker + FM 端到端一起训练)")
    parser.add_argument("--skip_baseline", action="store_true", default=False,
                        help="跳过基础模型直接预测阶段 (仅训练联合模型或 FM 优化)")
    parser.add_argument("--joint_alpha", type=float, default=None,
                        help="联合训练中直接预测损失权重 α (0~1, 默认 0.5)")
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
    # batch_size: --batch_size 覆盖两个, --fm_batch_size 只覆盖 FM
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
        config.fm_optimizer.fm_batch_size = args.batch_size
    if args.fm_batch_size is not None:
        config.fm_optimizer.fm_batch_size = args.fm_batch_size
    if args.grad_accum_steps is not None:
        config.fm_optimizer.grad_accum_steps = args.grad_accum_steps
    if args.compile_model:
        config.fm_optimizer.compile_full = True
    # 联合训练配置
    if args.joint_train:
        config.fm_optimizer.joint_train = True
    if args.joint_alpha is not None:
        config.fm_optimizer.joint_alpha = args.joint_alpha
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
    logger.info(f"联合训练: {config.fm_optimizer.joint_train}")
    logger.info(f"跳过基线: {args.skip_baseline}")
    logger.info(f"设备: {device}")
    logger.info(f"多卡训练: {config.training.multi_gpu}")
    if config.training.multi_gpu:
        logger.info(f"GPU IDs: {config.training.gpu_ids if config.training.gpu_ids else '全部可见 GPU'}")
    logger.info(f"Ranker batch_size: {config.dataset.batch_size}")
    logger.info(f"FM batch_size: {config.fm_optimizer.fm_batch_size}")
    if config.fm_optimizer.grad_accum_steps > 1:
        logger.info(f"梯度累积: {config.fm_optimizer.grad_accum_steps} 步 "
                    f"(等效 FM batch_size={config.fm_optimizer.fm_batch_size * config.fm_optimizer.grad_accum_steps})")
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

        # 根据模式选择运行方式
        use_joint = config.fm_optimizer.joint_train
        skip_base = args.skip_baseline

        if use_joint:
            # 联合训练模式: ranker + FM 端到端
            results = trainer.run_joint_comparison(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                fm_weights=args.fm_weights,
            )
        elif skip_base:
            # 跳过基线: 只训练 FM 优化器 (需先训练 ranker 再冻结)
            logger.info(f"跳过基线直接预测, 仅训练 FM 优化器")
            # 仍然需要先训练 ranker 以提取 embeddings
            trainer.train_ranker(model, train_loader, val_loader, model_name)
            # 提取 embeddings
            train_emb, train_labels, train_labels_np, train_user_ids = \
                trainer.extract_embeddings(model, train_loader)
            val_emb, val_labels, val_labels_np, val_user_ids = \
                trainer.extract_embeddings(model, val_loader)
            test_emb, test_labels, test_labels_np, test_user_ids = \
                trainer.extract_embeddings(model, test_loader)

            results = {'model_name': model_name, 'direct': {}, 'fm_optimized': {}}
            # 直接预测结果
            test_metrics_direct = trainer._evaluate_ranker(model, test_loader)
            results['direct'] = test_metrics_direct

            input_dim = train_emb.shape[-1]
            fm_weights = args.fm_weights or [0.1, 0.5, 1.0, 2.0]
            for fm_weight in fm_weights:
                fm_opt, fm_history = trainer.train_fm_optimizer(
                    train_embeddings=train_emb,
                    train_labels=train_labels,
                    val_embeddings=val_emb,
                    val_labels=val_labels,
                    model_name=model_name,
                    fm_weight=fm_weight,
                    input_dim=input_dim,
                )
                # 评估
                fm_opt.eval()
                with torch.no_grad():
                    test_ds = torch.utils.data.TensorDataset(test_emb, test_labels)
                    test_emb_loader = torch.utils.data.DataLoader(
                        test_ds, batch_size=config.fm_optimizer.fm_batch_size,
                        shuffle=False, pin_memory=True,
                    )
                    all_preds = []
                    for emb_batch, _ in test_emb_loader:
                        emb_batch = emb_batch.to(trainer.device, non_blocking=True)
                        pred = fm_opt(emb_batch)
                        all_preds.append(pred.cpu().numpy())
                    y_pred = np.concatenate(all_preds)

                fm_metrics = evaluate_predictions(
                    y_true=test_labels_np, y_pred=y_pred,
                    user_ids=test_user_ids,
                    metrics=config.evaluation.metrics,
                    ndcg_k=config.evaluation.ndcg_k,
                )
                results['fm_optimized'][f'fm_weight_{fm_weight}'] = fm_metrics
        else:
            # 标准模式: 分开训练
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
        json.dump(summary, f, indent=2, default=_json_default)
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
