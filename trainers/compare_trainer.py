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
from models.joint_fm_ranker import JointFMRanker

logger = logging.getLogger(__name__)


def _json_default(obj):
    """JSON 序列化回退: 处理 numpy / torch 数值类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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

        # 多卡配置
        self.use_multi_gpu = False
        self.gpu_ids = []
        if (
            config.training.multi_gpu
            and torch.cuda.is_available()
            and str(device).startswith("cuda")
        ):
            available_gpus = torch.cuda.device_count()
            requested = config.training.gpu_ids if config.training.gpu_ids else list(range(available_gpus))
            self.gpu_ids = [gid for gid in requested if 0 <= gid < available_gpus]
            if len(self.gpu_ids) >= 2:
                self.use_multi_gpu = True
                # DataParallel 要求模型在 device_ids[0] 上, 同时输入数据也必须发往该设备
                self.device = f"cuda:{self.gpu_ids[0]}"
                logger.info(f"启用多卡训练 DataParallel, GPU IDs: {self.gpu_ids}, 主设备: {self.device}")
            else:
                logger.info("multi_gpu=True 但可用 GPU < 2, 退化为单卡")

    def _maybe_wrap_data_parallel(self, model: nn.Module) -> nn.Module:
        """按配置将模型包装为 DataParallel。
        DataParallel 要求模型参数已在 device_ids[0] 上,
        所以先将模型移至主设备再包装。
        """
        if self.use_multi_gpu and not isinstance(model, nn.DataParallel):
            model = model.to(f"cuda:{self.gpu_ids[0]}")
            return nn.DataParallel(model, device_ids=self.gpu_ids)
        return model

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """获取原始模型 (去除 DataParallel 包装)"""
        return model.module if isinstance(model, nn.DataParallel) else model

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
        model = self._maybe_wrap_data_parallel(model)
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
                torch.save(self._unwrap_model(model).state_dict(), save_path)
                logger.info(f"  ★ 新最佳模型 (val_auc={best_val_auc:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # 加载最佳模型
        best_path = os.path.join(self.output_dir, f"{model_name}_best.pt")
        if os.path.exists(best_path):
            self._unwrap_model(model).load_state_dict(torch.load(best_path, map_location=self.device))

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

        pbar = tqdm(loader, desc="Ranker train", leave=False,
                     unit="batch", mininterval=0.5)
        for batch in pbar:
            sparse_values = {k: v.to(self.device, non_blocking=True) for k, v in batch["sparse_values"].items()}
            labels = batch["label"].to(self.device, non_blocking=True)

            pred = model(sparse_values)  # logits (AMP-safe)
            loss = F.binary_cross_entropy_with_logits(pred, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate_ranker(self, model, loader) -> Dict:
        """评估精排模型"""
        model.eval()
        all_preds = []
        all_labels = []
        all_user_ids = []

        for batch in tqdm(loader, desc="Ranker eval", leave=False,
                          unit="batch", mininterval=0.5):
            sparse_values = {k: v.to(self.device, non_blocking=True) for k, v in batch["sparse_values"].items()}
            logit = model(sparse_values)  # logits
            pred = torch.sigmoid(logit)    # → probability

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
        model = self._maybe_wrap_data_parallel(model)
        all_embeddings = []
        all_labels = []
        all_user_ids = []

        logger.info("提取精排模型 embeddings...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                sparse_values = {k: v.to(self.device) for k, v in batch["sparse_values"].items()}
                base_model = self._unwrap_model(model)
                emb = base_model.extract_embedding(sparse_values)  # (batch, data_dim)

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
        input_dim: int = None,
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
            input_dim: 精排模型 embedding 维度 (若与 FM data_dim 不同, 自动添加投影层)
        
        Returns:
            (fm_optimizer, training_history)
        """
        fm_config = self.config.fm_optimizer
        fm_config.fm_weight = fm_weight  # 使用指定的 fm_weight

        fm_optimizer = FMOptimizer(
            fm_config,
            input_dim=input_dim,
            fm_checkpoint=fm_config.fm_checkpoint if fm_config.fm_checkpoint else None,
            freeze_fm=fm_config.freeze_fm,
        ).to(self.device)
        fm_optimizer = self._maybe_wrap_data_parallel(fm_optimizer)
        total_params = sum(
            p.numel() for p in self._unwrap_model(fm_optimizer).parameters() if p.requires_grad
        )
        logger.info(f"=== 训练 FM 优化器 (model={model_name}, fm_weight={fm_weight}) ===")
        logger.info(f"  FM 模型可训练参数量: {total_params:,}")
        logger.info(f"  FM backbone: {fm_config.backbone_type}")
        logger.info(f"  FM 权重: {fm_weight}")
        logger.info(f"  快速训练模式: {fm_config.fast_train}")
        logger.info(f"  AMP: {fm_config.use_amp}")
        logger.info(f"  梯度累积: {fm_config.grad_accum_steps} 步")
        if fm_config.grad_accum_steps > 1:
            logger.info(f"  等效 batch_size: {fm_config.fm_batch_size * fm_config.grad_accum_steps}")
        logger.info(f"  compile_velocity: {fm_config.compile_velocity}")
        logger.info(f"  compile_full: {fm_config.compile_full}")
        logger.info(f"  gradient_checkpointing: {fm_config.gradient_checkpointing}")

        # torch.compile 加速 (PyTorch 2.0+)
        if fm_config.compile_velocity:
            try:
                base_fm = self._unwrap_model(fm_optimizer)
                base_fm.fm_model.velocity_net = torch.compile(
                    base_fm.fm_model.velocity_net, mode="reduce-overhead"
                )
                logger.info(f"  torch.compile (velocity_net): 已启用")
            except Exception as e:
                logger.warning(f"  torch.compile 失败: {e}, 跳过")

        # 编译整个 FMOptimizer (更激进)
        if fm_config.compile_full:
            try:
                base_fm = self._unwrap_model(fm_optimizer)
                base_fm.compute_total_loss = torch.compile(
                    base_fm.compute_total_loss, mode="reduce-overhead"
                )
                logger.info(f"  torch.compile (full model): 已启用")
            except Exception as e:
                logger.warning(f"  torch.compile (full) 失败: {e}, 跳过")

        # 梯度检查点 (省显存, 适合 no_freeze_fm 大模型)
        if fm_config.gradient_checkpointing:
            base_fm = self._unwrap_model(fm_optimizer)
            if hasattr(base_fm.fm_model.velocity_net, 'transformer'):
                # Transformer 版本: 对 TransformerEncoder 启用检查点
                base_fm.fm_model.velocity_net.transformer.gradient_checkpointing_enable()
            logger.info(f"  梯度检查点: 已启用 (省显存, 略慢)")

        optimizer = AdamW(
            fm_optimizer.parameters(),
            lr=fm_config.fm_lr,
            weight_decay=fm_config.fm_weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=fm_config.fm_epochs)

        # AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=fm_config.use_amp)

        best_val_auc = 0.0
        best_epoch = 0
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

            # 训练 (fast_train + AMP)
            train_stats = self._train_fm_epoch(
                fm_optimizer, train_emb_loader, optimizer, scaler
            )
            scheduler.step()

            # 评估 (按 eval_interval 控制频率, 减少耗时)
            need_eval = (epoch % fm_config.eval_interval == 0) or (epoch == fm_config.fm_epochs)
            if need_eval:
                val_metrics = self._evaluate_fm_optimizer(fm_optimizer, val_emb_loader)
            else:
                # 轻量代理评估: 跳过 ODE, 直接用 projected embedding 评估
                val_metrics = self._evaluate_fm_optimizer_fast(fm_optimizer, val_emb_loader)

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            eval_tag = "ODE" if need_eval else "fast"
            logger.info(
                f"FM Epoch {epoch}/{fm_config.fm_epochs} | "
                f"loss={train_stats['loss']:.6f} "
                f"fm={train_stats['fm_loss']:.6f} "
                f"bce={train_stats['bce_loss']:.6f} | "
                f"val_auc={val_metrics.get('auc', 0):.6f} [{eval_tag}] | "
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
                best_epoch = epoch
                save_path = os.path.join(self.output_dir, f"fm_{model_name}_w{fm_weight}_best.pt")
                torch.save(self._unwrap_model(fm_optimizer).state_dict(), save_path)
                logger.info(f"  ★ 新最佳 FM 模型 (val_auc={best_val_auc:.6f}, epoch={best_epoch})")

        # 最终用完整 ODE 评估一次 best model
        logger.info(f"  加载最佳模型 (epoch={best_epoch}) 进行完整 ODE 评估...")
        save_path = os.path.join(self.output_dir, f"fm_{model_name}_w{fm_weight}_best.pt")
        if os.path.exists(save_path):
            self._unwrap_model(fm_optimizer).load_state_dict(torch.load(save_path, map_location=self.device))
        final_val_metrics = self._evaluate_fm_optimizer(fm_optimizer, val_emb_loader)
        logger.info(f"  最终 ODE 评估: val_auc={final_val_metrics.get('auc', 0):.6f}")

        return fm_optimizer, history

    def _train_fm_epoch(self, fm_optimizer, loader, optimizer, scaler=None) -> Dict:
        """训练 FM 优化器一个 epoch (支持 fast_train + AMP + 梯度累积)"""
        fm_optimizer.train()
        use_amp = self.config.fm_optimizer.use_amp and scaler is not None
        fast_train = self.config.fm_optimizer.fast_train
        accum_steps = self.config.fm_optimizer.grad_accum_steps

        total_loss = 0.0
        total_fm_loss = 0.0
        total_bce_loss = 0.0
        n = 0

        pbar = tqdm(loader, desc="FM train", leave=False,
                     unit="batch", mininterval=0.5)
        for step, (embeddings, labels) in enumerate(pbar):
            embeddings = embeddings.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                base_fm = self._unwrap_model(fm_optimizer)
                loss_dict = base_fm.compute_total_loss(
                    embeddings, labels, fast_train=fast_train
                )
                # 梯度累积: 缩放损失
                scaled_loss = loss_dict['loss'] / accum_steps

            # 反向传播 (累积梯度)
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # 每 accum_steps 步更新一次参数
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if use_amp:
                    if self.config.fm_optimizer.fm_grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(fm_optimizer.parameters(), self.config.fm_optimizer.fm_grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if self.config.fm_optimizer.fm_grad_clip > 0:
                        nn.utils.clip_grad_norm_(fm_optimizer.parameters(), self.config.fm_optimizer.fm_grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss_dict['loss'].item()
            total_fm_loss += loss_dict['fm_loss']
            total_bce_loss += loss_dict['bce_loss']
            n += 1

            pbar.set_postfix(
                loss=f"{loss_dict['loss'].item():.4f}",
                fm=f"{loss_dict['fm_loss']:.4f}",
                bce=f"{loss_dict['bce_loss']:.4f}",
            )

        return {
            'loss': total_loss / max(n, 1),
            'fm_loss': total_fm_loss / max(n, 1),
            'bce_loss': total_bce_loss / max(n, 1),
        }

    @torch.no_grad()
    def _evaluate_fm_optimizer_fast(self, fm_optimizer, loader) -> Dict:
        """快速评估: 跳过 ODE, 直接用 projected embedding 评估 (用于中间 epoch)"""
        fm_optimizer.eval()
        all_preds = []
        all_labels = []

        for embeddings, labels in tqdm(loader, desc="FM eval (fast)", leave=False,
                                        unit="batch", mininterval=0.5):
            embeddings = embeddings.to(self.device, non_blocking=True)
            # 跳过 ODE, 直接用 projected embedding → pred_head
            projected = self._unwrap_model(fm_optimizer).project_input(embeddings)
            logit = self._unwrap_model(fm_optimizer).pred_head(projected).squeeze(-1)
            pred = torch.sigmoid(logit)

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

    @torch.no_grad()
    def _evaluate_fm_optimizer(self, fm_optimizer, loader) -> Dict:
        """评估 FM 优化器"""
        fm_optimizer.eval()
        all_preds = []
        all_labels = []

        for embeddings, labels in tqdm(loader, desc="FM eval (ODE)", leave=False,
                                        unit="batch", mininterval=0.5):
            embeddings = embeddings.to(self.device, non_blocking=True)
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
        # 获取精排模型 embedding 输出维度
        input_dim = train_emb.shape[-1]
        logger.info(f"  精排模型 embedding 维度: {input_dim}")

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
                input_dim=input_dim,
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
                for emb_batch, _ in tqdm(test_emb_loader, desc="FM test", leave=False,
                                          unit="batch", mininterval=0.5):
                    emb_batch = emb_batch.to(self.device, non_blocking=True)
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
            json.dump(results, f, indent=2, default=_json_default)
        logger.info(f"结果已保存: {result_path}")

        return results

    # ──────────────────────────────────────────
    # 联合训练: ranker + FM 端到端
    # ──────────────────────────────────────────

    def run_joint_comparison(
        self,
        model: BaseRanker,
        model_name: str,
        train_loader,
        val_loader,
        test_loader,
        fm_weights: List[float] = None,
    ) -> Dict:
        """
        联合训练对比实验: ranker + FM 一起训练。

        训练 3 种状态的对比:
        1. Direct: 只训练 ranker 直接预测
        2. Joint: ranker + FM 联合训练 (两条路径同时优化)
        3. FM-only: 冻结 ranker, 单独训练 FM 优化器 (作为参照)

        Args:
            model: 精排模型
            model_name: 模型名称
            train_loader, val_loader, test_loader: 数据加载器
            fm_weights: FM 权重列表

        Returns:
            dict with: direct, joint, fm_optimized results
        """
        if fm_weights is None:
            fm_weights = [0.1, 0.5, 1.0, 2.0]

        results = {
            'model_name': model_name,
            'direct': {},
            'joint': {},
            'fm_optimized': {},
        }

        fm_config = self.config.fm_optimizer

        # ──────────────────────────────────────
        # Phase 1: 直接预测 (训练 ranker)
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

        # 提取 embeddings (用于后续 fm_optimized)
        train_emb, train_labels, train_labels_np, train_user_ids = \
            self.extract_embeddings(model, train_loader)
        val_emb, val_labels, val_labels_np, val_user_ids = \
            self.extract_embeddings(model, val_loader)
        test_emb, test_labels, test_labels_np, test_user_ids = \
            self.extract_embeddings(model, test_loader)

        # ──────────────────────────────────────
        # Phase 2: 联合训练 (ranker + FM 一起训)
        # ──────────────────────────────────────
        for fm_weight in fm_weights:
            logger.info(f"\n{'='*60}")
            logger.info(f"Phase 2: 联合训练 — {model_name} + FM (weight={fm_weight})")
            logger.info(f"{'='*60}")

            joint_model = self.train_joint(model, train_loader, val_loader,
                                           model_name, fm_weight=fm_weight)

            # 评估联合训练的两条路径
            direct_metrics, fm_metrics = self._evaluate_joint(joint_model, test_loader)
            results['joint'][f'fm_weight_{fm_weight}'] = {
                'direct': direct_metrics,
                'fm': fm_metrics,
            }

            logger.info(f"\n{model_name} + FM joint (weight={fm_weight}) 测试结果:")
            logger.info(f"  直接预测:")
            for m, v in direct_metrics.items():
                logger.info(f"    {m}: {v:.6f}")
            logger.info(f"  FM 优化后:")
            for m, v in fm_metrics.items():
                logger.info(f"    {m}: {v:.6f}")

        # ──────────────────────────────────────
        # Phase 3: 分开训练 FM (冻结 ranker)
        # ──────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 3: 分开训练 FM — {model_name}")
        logger.info(f"{'='*60}")

        input_dim = train_emb.shape[-1]
        for fm_weight in fm_weights:
            logger.info(f"\n--- FM weight = {fm_weight} (分开训练) ---")

            fm_optimizer, fm_history = self.train_fm_optimizer(
                train_embeddings=train_emb,
                train_labels=train_labels,
                val_embeddings=val_emb,
                val_labels=val_labels,
                model_name=model_name,
                fm_weight=fm_weight,
                input_dim=input_dim,
            )

            # 评估分开训练的 FM
            fm_optimizer.eval()
            with torch.no_grad():
                test_dataset = torch.utils.data.TensorDataset(test_emb, test_labels)
                test_emb_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=self.config.fm_optimizer.fm_batch_size,
                    shuffle=False, pin_memory=True,
                )

                all_preds = []
                for emb_batch, _ in tqdm(test_emb_loader, desc="FM test", leave=False,
                                          unit="batch", mininterval=0.5):
                    emb_batch = emb_batch.to(self.device, non_blocking=True)
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

            logger.info(f"\n{model_name} + FM (分开训练, weight={fm_weight}) 测试结果:")
            for metric, value in fm_test_metrics.items():
                logger.info(f"  {metric}: {value:.6f}")

        # ──────────────────────────────────────
        # 汇总对比
        # ──────────────────────────────────────
        self._print_joint_comparison_summary(model_name, results)

        # 保存结果
        result_path = os.path.join(self.output_dir, f"{model_name}_joint_comparison.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=_json_default)
        logger.info(f"结果已保存: {result_path}")

        return results

    def train_joint(
        self,
        ranker: BaseRanker,
        train_loader,
        val_loader,
        model_name: str,
        fm_weight: float = 1.0,
    ) -> JointFMRanker:
        """
        联合训练 ranker + FM。

        Args:
            ranker: 已训练的精排模型 (权重会被覆盖)
            train_loader, val_loader: 数据加载器
            model_name: 模型名称
            fm_weight: FM 速度损失权重

        Returns:
            训练好的 JointFMRanker
        """
        fm_config = self.config.fm_optimizer
        fm_config.fm_weight = fm_weight

        # 重新加载最佳 ranker 权重
        best_path = os.path.join(self.output_dir, f"{model_name}_best.pt")
        if os.path.exists(best_path):
            ranker.load_state_dict(torch.load(best_path, map_location=self.device))

        # 创建联合模型
        joint_model = JointFMRanker(
            ranker=ranker,
            fm_config=fm_config,
            input_dim=ranker.hidden_dims[-1],
            fm_checkpoint=fm_config.fm_checkpoint if fm_config.fm_checkpoint else None,
            freeze_fm=fm_config.freeze_fm,
            joint_alpha=fm_config.joint_alpha if hasattr(fm_config, 'joint_alpha') else 0.5,
        ).to(self.device)
        joint_model = self._maybe_wrap_data_parallel(joint_model)

        total_params = sum(
            p.numel() for p in self._unwrap_model(joint_model).parameters() if p.requires_grad
        )
        ranker_params = sum(
            p.numel() for p in self._unwrap_model(joint_model).ranker.parameters() if p.requires_grad
        )
        fm_params = sum(
            p.numel() for p in self._unwrap_model(joint_model).fm_optimizer.parameters() if p.requires_grad
        )

        logger.info(f"=== 联合训练: {model_name} + FM (fm_weight={fm_weight}) ===")
        logger.info(f"  总可训练参数量: {total_params:,}")
        logger.info(f"    ranker 参数量: {ranker_params:,}")
        logger.info(f"    FM 参数量: {fm_params:,}")
        logger.info(f"  joint_alpha: {fm_config.joint_alpha if hasattr(fm_config, 'joint_alpha') else 0.5}")
        logger.info(f"  快速训练: {fm_config.fast_train}")
        logger.info(f"  AMP: {fm_config.use_amp}")

        # 编译
        if fm_config.compile_full:
            try:
                base_joint = self._unwrap_model(joint_model)
                base_joint.compute_joint_loss = torch.compile(
                    base_joint.compute_joint_loss, mode="reduce-overhead"
                )
                logger.info(f"  torch.compile (joint): 已启用")
            except Exception as e:
                logger.warning(f"  torch.compile 失败: {e}, 跳过")

        # 分组优化器: ranker 用较小 lr, FM 用配置中的 lr
        base_model = self._unwrap_model(joint_model)
        optimizer = AdamW([
            {'params': base_model.ranker.parameters(), 'lr': self.config.training.lr},
            {'params': base_model.fm_optimizer.parameters(), 'lr': fm_config.fm_lr},
        ], weight_decay=self.config.training.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=fm_config.fm_epochs)

        # AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=fm_config.use_amp)

        best_val_auc = 0.0
        best_epoch = 0
        accum_steps = fm_config.grad_accum_steps

        for epoch in range(1, fm_config.fm_epochs + 1):
            t0 = time.time()

            # 训练
            train_stats = self._train_joint_epoch(
                joint_model, train_loader, optimizer, scaler, accum_steps
            )
            scheduler.step()

            # 评估
            need_eval = (epoch % fm_config.eval_interval == 0) or (epoch == fm_config.fm_epochs)
            if need_eval:
                direct_metrics, fm_metrics = self._evaluate_joint(
                    joint_model, val_loader, full_ode=True
                )
            else:
                direct_metrics, fm_metrics = self._evaluate_joint(
                    joint_model, val_loader, full_ode=False
                )

            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            eval_tag = "ODE" if need_eval else "fast"
            logger.info(
                f"Joint Epoch {epoch}/{fm_config.fm_epochs} | "
                f"loss={train_stats['loss']:.6f} "
                f"direct={train_stats['bce_direct']:.6f} "
                f"fm_vel={train_stats['fm_velocity_loss']:.6f} "
                f"bce_fm={train_stats['bce_fm']:.6f} | "
                f"val_direct_auc={direct_metrics.get('auc', 0):.6f} "
                f"val_fm_auc={fm_metrics.get('auc', 0):.6f} [{eval_tag}] | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # 以 FM 预测 AUC 为选择标准
            current_auc = fm_metrics.get('auc', 0)
            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_epoch = epoch
                save_path = os.path.join(self.output_dir, f"joint_{model_name}_w{fm_weight}_best.pt")
                torch.save(self._unwrap_model(joint_model).state_dict(), save_path)
                logger.info(f"  ★ 新最佳联合模型 (val_fm_auc={best_val_auc:.6f}, epoch={best_epoch})")

        # 加载最佳模型
        save_path = os.path.join(self.output_dir, f"joint_{model_name}_w{fm_weight}_best.pt")
        if os.path.exists(save_path):
            self._unwrap_model(joint_model).load_state_dict(
                torch.load(save_path, map_location=self.device)
            )

        return joint_model

    def _train_joint_epoch(self, joint_model, loader, optimizer, scaler=None,
                            accum_steps: int = 1) -> Dict:
        """训练联合模型一个 epoch"""
        joint_model.train()
        use_amp = self.config.fm_optimizer.use_amp and scaler is not None
        fast_train = self.config.fm_optimizer.fast_train

        total_loss = 0.0
        total_direct = 0.0
        total_fm_vel = 0.0
        total_bce_fm = 0.0
        n = 0

        pbar = tqdm(loader, desc="Joint train", leave=False,
                     unit="batch", mininterval=0.5)
        for step, batch in enumerate(pbar):
            sparse_values = {k: v.to(self.device, non_blocking=True)
                            for k, v in batch["sparse_values"].items()}
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                base_joint = self._unwrap_model(joint_model)
                loss_dict = base_joint.compute_joint_loss(
                    sparse_values, labels, fast_train=fast_train
                )
                scaled_loss = loss_dict['loss'] / accum_steps

            # 反向传播 (累积)
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # 每 accum_steps 步更新参数
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if use_amp:
                    if self.config.fm_optimizer.fm_grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(joint_model.parameters(),
                                                  self.config.fm_optimizer.fm_grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if self.config.fm_optimizer.fm_grad_clip > 0:
                        nn.utils.clip_grad_norm_(joint_model.parameters(),
                                                  self.config.fm_optimizer.fm_grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss_dict['loss'].item()
            total_direct += loss_dict['bce_direct']
            total_fm_vel += loss_dict['fm_velocity_loss']
            total_bce_fm += loss_dict['bce_fm']
            n += 1

            pbar.set_postfix(
                loss=f"{loss_dict['loss'].item():.4f}",
                d=f"{loss_dict['bce_direct']:.4f}",
                fm=f"{loss_dict['fm_velocity_loss']:.4f}",
            )

        return {
            'loss': total_loss / max(n, 1),
            'bce_direct': total_direct / max(n, 1),
            'fm_velocity_loss': total_fm_vel / max(n, 1),
            'bce_fm': total_bce_fm / max(n, 1),
        }

    @torch.no_grad()
    def _evaluate_joint(self, joint_model, loader, full_ode: bool = True) -> Tuple[Dict, Dict]:
        """
        评估联合模型的两条路径。

        Returns:
            (direct_metrics, fm_metrics)
        """
        joint_model.eval()
        all_direct_preds = []
        all_fm_preds = []
        all_labels = []
        all_user_ids = []

        desc = "Joint eval (ODE)" if full_ode else "Joint eval (fast)"
        for batch in tqdm(loader, desc=desc, leave=False,
                          unit="batch", mininterval=0.5):
            sparse_values = {k: v.to(self.device, non_blocking=True)
                            for k, v in batch["sparse_values"].items()}

            base_joint = self._unwrap_model(joint_model)

            # 直接预测
            direct_pred = base_joint.predict_direct(sparse_values)

            # FM 优化后预测
            if full_ode:
                fm_pred = base_joint.predict_fm(
                    sparse_values,
                    num_steps=self.config.fm_optimizer.sample_steps,
                    delta_t=0.5,
                )
            else:
                # 快速评估: 跳过 ODE
                embeddings = base_joint.ranker.extract_embedding(sparse_values)
                projected = base_joint.fm_optimizer.project_input(embeddings)
                logit = base_joint.fm_optimizer.pred_head(projected).squeeze(-1)
                fm_pred = torch.sigmoid(logit)

            all_direct_preds.append(direct_pred.cpu().numpy())
            all_fm_preds.append(fm_pred.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            if "user_id" in batch:
                all_user_ids.append(batch["user_id"])

        y_direct = np.concatenate(all_direct_preds)
        y_fm = np.concatenate(all_fm_preds)
        y_true = np.concatenate(all_labels)
        user_ids = np.concatenate(all_user_ids) if all_user_ids else None

        direct_metrics = evaluate_predictions(
            y_true=y_true, y_pred=y_direct, user_ids=user_ids,
            metrics=self.config.evaluation.metrics,
            ndcg_k=self.config.evaluation.ndcg_k,
        )
        fm_metrics = evaluate_predictions(
            y_true=y_true, y_pred=y_fm, user_ids=user_ids,
            metrics=self.config.evaluation.metrics,
            ndcg_k=self.config.evaluation.ndcg_k,
        )

        return direct_metrics, fm_metrics

    def _print_joint_comparison_summary(self, model_name: str, results: Dict):
        """打印联合训练对比结果汇总"""
        logger.info(f"\n{'='*70}")
        logger.info(f"联合训练对比汇总: {model_name}")
        logger.info(f"{'='*70}")

        # 表头
        metrics_keys = list(results['direct'].keys())
        header = f"{'Method':<30}"
        for m in metrics_keys:
            header += f"{m:>12}"
        logger.info(header)
        logger.info("-" * len(header))

        # 直接预测
        if results.get('direct'):
            row = f"{'Direct Prediction':<30}"
            for m, v in results['direct'].items():
                row += f"{v:>12.6f}"
            logger.info(row)

        # 联合训练
        for fw_key, joint_result in results.get('joint', {}).items():
            row = f"{'Joint-direct (' + fw_key + ')':<30}"
            for m, v in joint_result['direct'].items():
                row += f"{v:>12.6f}"
            logger.info(row)

            row = f"{'Joint-FM (' + fw_key + ')':<30}"
            for m, v in joint_result['fm'].items():
                row += f"{v:>12.6f}"
            logger.info(row)

        # 分开训练 FM
        for fw_key, metrics in results.get('fm_optimized', {}).items():
            row = f"{'Separate-FM (' + fw_key + ')':<30}"
            for m, v in metrics.items():
                row += f"{v:>12.6f}"
            logger.info(row)

        # 提升量
        if results.get('direct'):
            for fw_key, joint_result in results.get('joint', {}).items():
                logger.info(f"\n  Joint-FM ({fw_key}) vs Direct 提升:")
                for m in metrics_keys:
                    direct_val = results['direct'][m]
                    fm_val = joint_result['fm'][m]
                    if m == "logloss":
                        imp = direct_val - fm_val
                    else:
                        imp = fm_val - direct_val
                    sign = "+" if imp > 0 else ""
                    logger.info(f"    {m}: {sign}{imp:.6f}")

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
