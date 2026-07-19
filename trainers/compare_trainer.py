"""
compare_trainer.py — 对比实验训练器
=====================================
核心逻辑: 对每个精排模型，分别测试:
1. 直接预测: Features → Model → CTR/CVR
2. FM 优化后预测: Features → Model → Embedding → FM 优化 → 预测

FM 权重可配置，支持多组 fm_weight 对比实验。
"""

import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from utils.config import CompareConfig, FMOptimizerConfig
from utils.metrics import evaluate_predictions
from models.base_ranker import BaseRanker
from models.fm_optimizer import FMOptimizer

logger = logging.getLogger(__name__)


class CompareTrainer:
    """
    对比实验训练器。
    
    流程:
    1. 训练精排模型 (直接预测)
    2. 评估精排模型直接预测效果
    3. 冻结精排模型，提取 embeddings
    4. 训练 FM 优化器 (以精排模型 embedding 为条件)
    5. 评估 FM 优化后的预测效果
    6. 对比两组结果
    """

    def __init__(self, config: CompareConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train_ranker(
        self,
        model: BaseRanker,
        train_loader,
        val_loader,
        model_name: str,
    ) -> Dict:
        """
        训练精排模型 (直接预测)。
        
        Args:
            model: 精排模型
            train_loader: 训练 DataLoader
            val_loader: 验证 DataLoader
            model_name: 模型名称
        
        Returns:
            dict with training history and best metrics
        """
        model = model.to(self.device)
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.training.epochs, eta_min=self.config.training.lr * 0.01)

        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = []

        logger.info(f"=== 训练精排模型: {model_name} ===")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  可训练参数量: {total_params:,}")

        for epoch in range(1, self.config.training.epochs + 1):
            t0 = time.time()

            # 训练
            train_loss = self._train_ranker_epoch(model, train_loader, optimizer)
            scheduler.step()

            # 验证
            val_metrics = self._evaluate_ranker(model, val_loader)

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch {epoch}/{self.config.training.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_auc={val_metrics.get('auc', 0):.6f} | "
                f"val_logloss={val_metrics.get('logloss', 0):.6f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_auc': val_metrics.get('auc', 0),
                'val_logloss': val_metrics.get('logloss', 0),
                'lr': lr,
            })

            # Early stopping
            current_auc = val_metrics.get('auc', 0)
            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳模型
                save_path = os.path.join(self.output_dir, f"{model_name}_best.pt")
                torch.save(model.state_dict(), save_path)
                logger.info(f"  ★ 新最佳模型 (val_auc={best_val_auc:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # 加载最佳模型
        best_path = os.path.join(self.output_dir, f"{model_name}_best.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=self.device))

        # 最终验证
        final_metrics = self._evaluate_ranker(model, val_loader)

        return {
            'model_name': model_name,
            'history': history,
            'best_epoch': best_epoch,
            'best_val_auc': best_val_auc,
            'final_metrics': final_metrics,
        }

    def _train_ranker_epoch(self, model, loader, optimizer) -> float:
        """训练精排模型一个 epoch"""
        model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            sparse_values = {k: v.to(self.device) for k, v in batch["sparse_values"].items()}
            labels = batch["label"].to(self.device)

            pred = model(sparse_values)
            loss = F.binary_cross_entropy(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate_ranker(self, model, loader) -> Dict:
        """评估精排模型"""
        model.eval()
        all_preds = []
        all_labels = []
        all_user_ids = []

        for batch in loader:
            sparse_values = {k: v.to(self.device) for k, v in batch["sparse_values"].items()}
            pred = model(sparse_values)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            if "user_id" in batch:
                all_user_ids.append(batch["user_id"])

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        user_ids = np.concatenate(all_user_ids) if all_user_ids else None

        return evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            user_ids=user_ids,
            metrics=self.config.evaluation.metrics,
            ndcg_k=self.config.evaluation.ndcg_k,
        )

    def extract_embeddings(
        self,
        model: BaseRanker,
        loader,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Optional[np.ndarray]]:
        """
        从精排模型提取 embeddings 和标签。
        
        Returns:
            embeddings: (N, data_dim) 精排模型的 embedding
            labels: (N,) 标签
            labels_np: numpy 标签
            user_ids: numpy 用户 ID (用于 GAUC)
        """
        model.eval()
        model = model.to(self.device)
        all_embeddings = []
        all_labels = []
        all_user_ids = []

        logger.info("提取精排模型 embeddings...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                sparse_values = {k: v.to(self.device) for k, v in batch["sparse_values"].items()}
                emb = model.extract_embedding(sparse_values)  # (batch, data_dim)

                all_embeddings.append(emb.cpu())
                all_labels.append(batch["label"])
                if "user_id" in batch:
                    all_user_ids.append(batch["user_id"])

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        labels_np = labels.numpy()
        user_ids = np.concatenate(all_user_ids) if all_user_ids else None

        logger.info(f"  提取 embeddings: {embeddings.shape}")
        return embeddings, labels, labels_np, user_ids

    def train_fm_optimizer(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: torch.Tensor,
        val_labels: torch.Tensor,
        model_name: str,
        fm_weight: float = 1.0,
    ) -> Tuple[FMOptimizer, Dict]:
        """
        训练 FM 优化器。
        
        Args:
            train_embeddings: 训练集精排模型 embeddings
            train_labels: 训练集标签
            val_embeddings: 验证集精排模型 embeddings
            val_labels: 验证集标签
            model_name: 精排模型名称
            fm_weight: FM 权重
        
        Returns:
            (fm_optimizer, training_history)
        """
        fm_config = self.config.fm_optimizer
        fm_config.fm_weight = fm_weight  # 使用指定的 fm_weight

        fm_optimizer = FMOptimizer(fm_config).to(self.device)
        total_params = sum(p.numel() for p in fm_optimizer.parameters() if p.requires_grad)
        logger.info(f"=== 训练 FM 优化器 (model={model_name}, fm_weight={fm_weight}) ===")
        logger.info(f"  FM 模型可训练参数量: {total_params:,}")
        logger.info(f"  FM backbone: {fm_config.backbone_type}")
        logger.info(f"  FM 权重: {fm_weight}")

        optimizer = AdamW(
            fm_optimizer.parameters(),
            lr=fm_config.fm_lr,
            weight_decay=fm_config.fm_weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=fm_config.fm_epochs)

        best_val_auc = 0.0
        history = []
        batch_size = fm_config.fm_batch_size

        # 创建 embedding DataLoader
        train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_embeddings, val_labels)
        train_emb_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        )
        val_emb_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        )

        for epoch in range(1, fm_config.fm_epochs + 1):
            t0 = time.time()

            # 训练
            train_stats = self._train_fm_epoch(fm_optimizer, train_emb_loader, optimizer)
            scheduler.step()

            # 验证
            val_metrics = self._evaluate_fm_optimizer(fm_optimizer, val_emb_loader)

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            logger.info(
                f"FM Epoch {epoch}/{fm_config.fm_epochs} | "
                f"loss={train_stats['loss']:.6f} "
                f"fm={train_stats['fm_loss']:.6f} "
                f"bce={train_stats['bce_loss']:.6f} | "
                f"val_auc={val_metrics.get('auc', 0):.6f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            history.append({
                'epoch': epoch,
                **train_stats,
                'val_auc': val_metrics.get('auc', 0),
                'val_logloss': val_metrics.get('logloss', 0),
            })

            # 保存最佳 FM 模型
            current_auc = val_metrics.get('auc', 0)
            if current_auc > best_val_auc:
                best_val_auc = current_auc
                save_path = os.path.join(self.output_dir, f"fm_{model_name}_w{fm_weight}_best.pt")
                torch.save(fm_optimizer.state_dict(), save_path)
                logger.info(f"  ★ 新最佳 FM 模型 (val_auc={best_val_auc:.6f})")

        return fm_optimizer, history

    def _train_fm_epoch(self, fm_optimizer, loader, optimizer) -> Dict:
        """训练 FM 优化器一个 epoch"""
        fm_optimizer.train()
        total_loss = 0.0
        total_fm_loss = 0.0
        total_bce_loss = 0.0
        n = 0

        for embeddings, labels in loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            loss_dict = fm_optimizer.compute_total_loss(embeddings, labels)

            optimizer.zero_grad()
            loss_dict['loss'].backward()
            if self.config.fm_optimizer.fm_grad_clip > 0:
                nn.utils.clip_grad_norm_(fm_optimizer.parameters(), self.config.fm_optimizer.fm_grad_clip)
            optimizer.step()

            total_loss += loss_dict['loss'].item()
            total_fm_loss += loss_dict['fm_loss']
            total_bce_loss += loss_dict['bce_loss']
            n += 1

        return {
            'loss': total_loss / max(n, 1),
            'fm_loss': total_fm_loss / max(n, 1),
            'bce_loss': total_bce_loss / max(n, 1),
        }

    @torch.no_grad()
    def _evaluate_fm_optimizer(self, fm_optimizer, loader) -> Dict:
        """评估 FM 优化器"""
        fm_optimizer.eval()
        all_preds = []
        all_labels = []

        for embeddings, labels in loader:
            embeddings = embeddings.to(self.device)
            pred = fm_optimizer(embeddings)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        return evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            metrics=self.config.evaluation.metrics,
            ndcg_k=self.config.evaluation.ndcg_k,
        )

    def run_comparison(
        self,
        model: BaseRanker,
        model_name: str,
        train_loader,
        val_loader,
        test_loader,
        fm_weights: List[float] = None,
    ) -> Dict:
        """
        对单个精排模型运行完整对比实验。
        
        Args:
            model: 精排模型
            model_name: 模型名称
            train_loader: 训练集
            val_loader: 验证集
            test_loader: 测试集
            fm_weights: FM 权重列表 (默认 [0.1, 0.5, 1.0, 2.0])
        
        Returns:
            dict with all comparison results
        """
        if fm_weights is None:
            fm_weights = [0.1, 0.5, 1.0, 2.0]

        results = {
            'model_name': model_name,
            'direct': {},
            'fm_optimized': {},
        }

        # ──────────────────────────────────────
        # Phase 1: 直接预测 (训练精排模型)
        # ──────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 1: 直接预测 — {model_name}")
        logger.info(f"{'='*60}")

        ranker_history = self.train_ranker(model, train_loader, val_loader, model_name)
        test_metrics_direct = self._evaluate_ranker(model, test_loader)
        results['direct'] = test_metrics_direct

        logger.info(f"\n{model_name} 直接预测结果:")
        for metric, value in test_metrics_direct.items():
            logger.info(f"  {metric}: {value:.6f}")

        # ──────────────────────────────────────
        # Phase 2: FM 优化后预测
        # ──────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 2: FM 优化后预测 — {model_name}")
        logger.info(f"{'='*60}")

        # 提取 embeddings
        train_emb, train_labels, train_labels_np, train_user_ids = \
            self.extract_embeddings(model, train_loader)
        val_emb, val_labels, val_labels_np, val_user_ids = \
            self.extract_embeddings(model, val_loader)
        test_emb, test_labels, test_labels_np, test_user_ids = \
            self.extract_embeddings(model, test_loader)

        # 对每个 fm_weight 进行实验
        for fm_weight in fm_weights:
            logger.info(f"\n--- FM weight = {fm_weight} ---")

            # 训练 FM 优化器
            fm_optimizer, fm_history = self.train_fm_optimizer(
                train_embeddings=train_emb,
                train_labels=train_labels,
                val_embeddings=val_emb,
                val_labels=val_labels,
                model_name=model_name,
                fm_weight=fm_weight,
            )

            # 评估 FM 优化后的预测效果
            fm_optimizer.eval()
            with torch.no_grad():
                # 使用测试集 embedding
                test_dataset = torch.utils.data.TensorDataset(test_emb, test_labels)
                test_emb_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=self.config.fm_optimizer.fm_batch_size,
                    shuffle=False, pin_memory=True,
                )

                all_preds = []
                for emb_batch, _ in test_emb_loader:
                    emb_batch = emb_batch.to(self.device)
                    pred = fm_optimizer(emb_batch)
                    all_preds.append(pred.cpu().numpy())

                y_pred = np.concatenate(all_preds)

            fm_test_metrics = evaluate_predictions(
                y_true=test_labels_np,
                y_pred=y_pred,
                user_ids=test_user_ids,
                metrics=self.config.evaluation.metrics,
                ndcg_k=self.config.evaluation.ndcg_k,
            )

            results['fm_optimized'][f'fm_weight_{fm_weight}'] = fm_test_metrics

            logger.info(f"\n{model_name} + FM (weight={fm_weight}) 测试结果:")
            for metric, value in fm_test_metrics.items():
                logger.info(f"  {metric}: {value:.6f}")

        # ──────────────────────────────────────
        # 汇总对比
        # ──────────────────────────────────────
        self._print_comparison_summary(model_name, results)

        # 保存结果
        result_path = os.path.join(self.output_dir, f"{model_name}_comparison.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"结果已保存: {result_path}")

        return results

    def _print_comparison_summary(self, model_name: str, results: Dict):
        """打印对比结果汇总"""
        logger.info(f"\n{'='*60}")
        logger.info(f"对比汇总: {model_name}")
        logger.info(f"{'='*60}")

        # 表头
        header = f"{'Method':<30}"
        for metric in results['direct'].keys():
            header += f"{metric:>12}"
        logger.info(header)
        logger.info("-" * len(header))

        # 直接预测
        direct_row = f"{'Direct Prediction':<30}"
        for metric, value in results['direct'].items():
            direct_row += f"{value:>12.6f}"
        logger.info(direct_row)

        # FM 优化后
        for fm_key, metrics in results['fm_optimized'].items():
            fm_row = f"{'FM (' + fm_key + ')':<30}"
            for metric, value in metrics.items():
                fm_row += f"{value:>12.6f}"
            logger.info(fm_row)

        # 计算提升
        for fm_key, metrics in results['fm_optimized'].items():
            improvements = {}
            for metric in results['direct'].keys():
                direct_val = results['direct'][metric]
                fm_val = metrics[metric]
                # AUC 越高越好, LogLoss 越低越好
                if metric == "logloss":
                    improvement = direct_val - fm_val  # 正数表示提升
                else:
                    improvement = fm_val - direct_val  # 正数表示提升
                improvements[metric] = improvement

            logger.info(f"\n  {fm_key} 提升量:")
            for metric, improvement in improvements.items():
                sign = "+" if improvement > 0 else ""
                logger.info(f"    {metric}: {sign}{improvement:.6f}")
