"""
metrics.py — 评估指标
=======================
支持 AUC, GAUC, LogLoss, NDCG 等指标。
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, List, Optional


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 AUC"""
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0


def compute_gauc(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray) -> float:
    """
    计算 GAUC (Group AUC)
    
    GAUC = sum(#clicks_u * AUC_u) / sum(#clicks_u)
    """
    unique_users = np.unique(user_ids)
    total_weighted_auc = 0.0
    total_clicks = 0

    for user in unique_users:
        mask = user_ids == user
        user_y_true = y_true[mask]
        user_y_pred = y_pred[mask]

        # 需要同时有正负样本才能计算 AUC
        if len(np.unique(user_y_true)) < 2:
            continue

        user_auc = compute_auc(user_y_true, user_y_pred)
        user_clicks = mask.sum()
        total_weighted_auc += user_clicks * user_auc
        total_clicks += user_clicks

    if total_clicks == 0:
        return 0.0
    return total_weighted_auc / total_clicks


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """计算 LogLoss (Binary Cross-Entropy)"""
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """计算 NDCG@K"""
    # 按 y_pred 降序排列
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order]

    # DCG@K
    k = min(k, len(y_sorted))
    gains = 2 ** y_sorted[:k] - 1
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum(gains / discounts)

    # Ideal DCG@K
    ideal_order = np.argsort(y_true)[::-1]
    ideal_sorted = y_true[ideal_order]
    ideal_gains = 2 ** ideal_sorted[:k] - 1
    ideal_discounts = np.log2(np.arange(k) + 2)
    idcg = np.sum(ideal_gains / ideal_discounts)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_ids: Optional[np.ndarray] = None,
    metrics: List[str] = ["auc", "gauc", "logloss", "ndcg"],
    ndcg_k: int = 10,
) -> Dict[str, float]:
    """
    统一评估接口
    
    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测概率
        user_ids: 用户 ID (GAUC 计算需要)
        metrics: 需要计算的指标列表
        ndcg_k: NDCG 的 K 值
    
    Returns:
        dict of metric_name -> value
    """
    results = {}
    
    for metric in metrics:
        if metric == "auc":
            results["auc"] = compute_auc(y_true, y_pred)
        elif metric == "gauc":
            if user_ids is not None:
                results["gauc"] = compute_gauc(y_true, y_pred, user_ids)
            else:
                results["gauc"] = 0.0
        elif metric == "logloss":
            results["logloss"] = compute_logloss(y_true, y_pred)
        elif metric == "ndcg":
            results[f"ndcg@{ndcg_k}"] = compute_ndcg(y_true, y_pred, k=ndcg_k)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return results
