"""
mac_dataset.py — Ali MAC 数据集加载器
=========================================
从 HuggingFace (alimamaTech/MAC) 加载多归因转化率预测数据集。

MAC 数据集特征 (来自 PyMAL):
- 7 个用户稀疏特征: user_id, user_feat0~5
- 10 个广告/上下文稀疏特征: ad_feat0~5, context_feat0~3
- 2 个多模态特征: mm_feat0_seq, mm_feat1_seq
- 3 个行为序列特征: ad_feat0_seq, ad_feat1_seq, ad_feat2_seq
- 4 种归因标签: first, last, mta(dda), linear

参考: https://arxiv.org/pdf/2603.02184
      https://github.com/alimama-tech/PyMAL
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# MAC 数据集特征定义 (与 PyMAL 一致)
# ──────────────────────────────────────────────────

USER_COLS = ['user_id', 'user_feat0', 'user_feat1', 'user_feat2', 'user_feat3', 'user_feat4', 'user_feat5']
AD_COLS = ['ad_feat0', 'ad_feat1', 'ad_feat2', 'ad_feat3', 'ad_feat4', 'ad_feat5',
           'context_feat0', 'context_feat1', 'context_feat2', 'context_feat3']
MM_COLS = ['mm_feat0_seq', 'mm_feat1_seq']
SEQ_COLS = ['ad_feat0_seq', 'ad_feat1_seq', 'ad_feat2_seq']
LABEL_COLS = ['first', 'last', 'mta', 'linear']

# 稀疏特征 (用于精排模型)
SPARSE_FEATURES = USER_COLS + AD_COLS

# 归因标签映射
ATTRIBUTION_MAP = {
    "last_click": "last",
    "first_click": "first",
    "dda": "mta",
    "linear": "linear",
}

TRAIN_FILES = [f'train-{i:02d}.parquet' for i in range(20)]
TEST_FILES = ['test.parquet']


class MACDataset(Dataset):
    """
    Ali MAC 数据集 Dataset。
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        user_ids: np.ndarray = None,
    ):
        self.features = features
        self.labels = labels
        self.user_ids = user_ids
        self.sparse_feature_names = list(features.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sparse_values = {}
        for feat_name in self.sparse_feature_names:
            sparse_values[feat_name] = torch.tensor(self.features[feat_name][idx], dtype=torch.long)

        item = {
            "sparse_values": sparse_values,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.user_ids is not None:
            item["user_id"] = self.user_ids[idx]
        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义 collate 函数"""
    sparse_values = {}
    for feat_name in batch[0]["sparse_values"].keys():
        sparse_values[feat_name] = torch.stack([item["sparse_values"][feat_name] for item in batch])

    result = {
        "sparse_values": sparse_values,
        "label": torch.stack([item["label"] for item in batch]),
    }

    if "user_id" in batch[0]:
        result["user_id"] = np.array([item["user_id"] for item in batch])

    return result


def get_feature_info() -> Dict:
    """返回 MAC 数据集特征信息"""
    return {
        "sparse_features": SPARSE_FEATURES,
        "label_columns": LABEL_COLS,
        "attribution_map": ATTRIBUTION_MAP,
    }


def get_vocabs(path_vocabs: str) -> Dict:
    """
    加载词表映射 (与 PyMAL 一致)。
    
    每个稀疏特征有一个 JSON 文件，将原始值映射为整数索引。
    """
    vocabs = {}
    for col in SPARSE_FEATURES:
        vocab_path = os.path.join(path_vocabs, f"{col}.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocabs[col] = json.load(f)
        else:
            logger.warning(f"词表文件不存在: {vocab_path}")
            vocabs[col] = {}
    return vocabs


def df_to_dict(df: pd.DataFrame, vocabs: Dict) -> Dict:
    """
    将 DataFrame 转为字典格式 (与 PyMAL 一致)。
    
    使用词表将原始值映射为整数索引。
    """
    result = {}

    for col in SPARSE_FEATURES:
        if col in df.columns:
            vocab = vocabs.get(col, {})
            if vocab:
                mapper = np.vectorize(lambda x: vocab.get(str(x), 0))
                result[col] = mapper(df[col].values).astype(np.int64)
            else:
                # 没有词表时自动编码
                unique_vals = df[col].dropna().unique()
                val_to_idx = {str(val): idx for idx, val in enumerate(sorted(unique_vals.astype(str)))}
                mapper = np.vectorize(lambda x: val_to_idx.get(str(x), 0))
                result[col] = mapper(df[col].values).astype(np.int64)

    return result


def load_mac_data(
    dataset_name: str = "alimamaTech/MAC",
    target_attribution: str = "last_click",
    max_samples: int = -1,
    data_path: str = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    加载 MAC 数据集。
    
    支持两种模式:
    1. 从本地文件加载 (data_path 指向 git clone 后的目录)
    2. 从 HuggingFace datasets API 加载 (自动下载)
    
    Args:
        dataset_name: HuggingFace 数据集名称
        target_attribution: 目标归因机制 (last_click/first_click/dda/linear)
        max_samples: 最大样本数 (-1 表示全部加载)
        data_path: 本地数据路径 (如果提供, 从本地文件加载)
    
    Returns:
        (train_data, val_data, test_data) — 各为 dict:
            features: {feat_name: np.array(int64)}
            labels: np.array(float32)
            user_ids: np.array(int64)
            feature_voc_sizes: {feat_name: int}
    """
    # 确定标签列
    label_col = ATTRIBUTION_MAP.get(target_attribution, "last")
    logger.info(f"目标归因: {target_attribution} → 标签列: {label_col}")

    if data_path and os.path.isdir(data_path):
        return _load_from_local(data_path, label_col, max_samples)
    else:
        return _load_from_huggingface(dataset_name, label_col, max_samples, target_attribution)


def _load_from_local(
    data_path: str,
    label_col: str,
    max_samples: int,
) -> Tuple[Dict, Dict, Dict]:
    """从本地 parquet 文件加载 MAC 数据集 (与 PyMAL 格式一致)"""
    logger.info(f"从本地加载 MAC 数据集: {data_path}")

    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    vocabs_dir = os.path.join(data_path, "vocabs")

    # 加载词表
    vocabs = get_vocabs(vocabs_dir) if os.path.isdir(vocabs_dir) else {}
    if vocabs:
        logger.info(f"加载词表: {len(vocabs)} 个特征")

    # 加载训练数据
    train_dfs = []
    for fname in TRAIN_FILES:
        fpath = os.path.join(train_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            train_dfs.append(df)
            logger.info(f"  加载 {fname}: {len(df)} 行")

    if not train_dfs:
        # 尝试其他文件格式
        for f in sorted(os.listdir(train_dir)):
            if f.endswith('.parquet'):
                df = pd.read_parquet(os.path.join(train_dir, f))
                train_dfs.append(df)
                logger.info(f"  加载 {f}: {len(df)} 行")

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    logger.info(f"训练集总大小: {len(train_df)}")

    # 加载测试数据
    test_dfs = []
    for fname in TEST_FILES:
        fpath = os.path.join(test_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            test_dfs.append(df)
            logger.info(f"  加载 {fname}: {len(df)} 行")

    if not test_dfs:
        for f in sorted(os.listdir(test_dir)):
            if f.endswith('.parquet'):
                df = pd.read_parquet(os.path.join(test_dir, f))
                test_dfs.append(df)
                logger.info(f"  加载 {f}: {len(df)} 行")

    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    logger.info(f"测试集总大小: {len(test_df)}")

    # 限制样本数
    if max_samples > 0:
        train_df = train_df.head(min(max_samples, len(train_df)))
        test_df = test_df.head(min(max_samples, len(test_df)))

    return _process_dataframes(train_df, test_df, vocabs, label_col)


def _load_from_huggingface(
    dataset_name: str,
    label_col: str,
    max_samples: int,
    target_attribution: str,
) -> Tuple[Dict, Dict, Dict]:
    """从 HuggingFace datasets API 加载 MAC 数据集"""
    from datasets import load_dataset

    logger.info(f"从 HuggingFace 加载: {dataset_name}")

    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"加载 HuggingFace 数据集失败: {e}")
        logger.info("请尝试先下载数据集: git clone https://huggingface.co/datasets/alimamaTech/MAC data/")
        raise

    train_df = ds["train"].to_pandas() if "train" in ds else None
    test_df = ds["test"].to_pandas() if "test" in ds else None

    if train_df is None or test_df is None:
        raise ValueError(f"数据集缺少 train/test split, 可用 splits: {list(ds.keys())}")

    logger.info(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")

    # 限制样本数
    if max_samples > 0:
        train_df = train_df.head(min(max_samples, len(train_df)))
        test_df = test_df.head(min(max_samples, len(test_df)))

    # 自动构建词表 (从训练数据)
    vocabs = {}
    for col in SPARSE_FEATURES:
        if col in train_df.columns:
            unique_vals = train_df[col].dropna().unique()
            vocabs[col] = {str(val): idx for idx, val in enumerate(sorted(unique_vals.astype(str)))}

    return _process_dataframes(train_df, test_df, vocabs, label_col)


def _process_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocabs: Dict,
    label_col: str,
) -> Tuple[Dict, Dict, Dict]:
    """
    处理 DataFrame, 提取特征和标签, 并返回标准化的数据格式。
    """
    # 检查可用的稀疏特征
    available_features = [f for f in SPARSE_FEATURES if f in train_df.columns]
    logger.info(f"可用稀疏特征: {available_features} ({len(available_features)}/{len(SPARSE_FEATURES)})")

    # 计算词表大小
    feature_voc_sizes = {}
    for feat_name in available_features:
        vocab = vocabs.get(feat_name, {})
        feature_voc_sizes[feat_name] = len(vocab) + 1  # +1 for unknown/padding

    # 处理特征
    def prepare_features_and_labels(df, available_features, vocabs, label_col):
        features = df_to_dict(df, vocabs)

        # 标签
        if label_col in df.columns:
            labels = (df[label_col].values > 0).astype(np.float32)
        else:
            # 尝试映射
            available_labels = [c for c in LABEL_COLS if c in df.columns]
            if available_labels:
                actual_label = available_labels[0]
                labels = (df[actual_label].values > 0).astype(np.float32)
                logger.warning(f"标签列 {label_col} 不存在, 使用 {actual_label}")
            else:
                logger.error("找不到任何标签列!")
                labels = np.zeros(len(df), dtype=np.float32)

        # user_id
        user_ids = None
        if 'user_id' in df.columns:
            vocab = vocabs.get('user_id', {})
            if vocab:
                mapper = np.vectorize(lambda x: vocab.get(str(x), 0))
                user_ids = mapper(df['user_id'].values).astype(np.int64)
            else:
                unique_users = df['user_id'].dropna().unique()
                user_map = {str(val): idx for idx, val in enumerate(sorted(unique_users.astype(str)))}
                mapper = np.vectorize(lambda x: user_map.get(str(x), 0))
                user_ids = mapper(df['user_id'].values).astype(np.int64)

        return features, labels, user_ids

    train_features, train_labels, train_user_ids = prepare_features_and_labels(
        train_df, available_features, vocabs, label_col)
    test_features, test_labels, test_user_ids = prepare_features_and_labels(
        test_df, available_features, vocabs, label_col)

    # 从训练集分割验证集
    n_train = len(train_labels)
    n_val = int(n_train * 0.1)
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    val_features = {k: v[val_indices] for k, v in train_features.items()}
    val_labels = train_labels[val_indices]
    val_user_ids = train_user_ids[val_indices] if train_user_ids is not None else None

    train_features = {k: v[train_indices] for k, v in train_features.items()}
    train_labels = train_labels[train_indices]
    train_user_ids = train_user_ids[train_indices] if train_user_ids is not None else None

    # 统计
    for name, labels in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        pos_rate = labels.mean()
        logger.info(f"{name}: {len(labels)} 样本, 正样本比例: {pos_rate:.4f}")

    def make_data_dict(features, labels, user_ids):
        return {
            "features": features,
            "labels": labels,
            "user_ids": user_ids,
            "feature_voc_sizes": feature_voc_sizes,
            "sparse_feature_names": available_features,
        }

    return (
        make_data_dict(train_features, train_labels, train_user_ids),
        make_data_dict(val_features, val_labels, val_user_ids),
        make_data_dict(test_features, test_labels, test_user_ids),
    )


def get_mac_dataloaders(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    batch_size: int = 4096,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 MAC 数据集的 DataLoader。
    """
    train_dataset = MACDataset(
        features=train_data["features"],
        labels=train_data["labels"],
        user_ids=train_data.get("user_ids"),
    )
    val_dataset = MACDataset(
        features=val_data["features"],
        labels=val_data["labels"],
        user_ids=val_data.get("user_ids"),
    )
    test_dataset = MACDataset(
        features=test_data["features"],
        labels=test_data["labels"],
        user_ids=test_data.get("user_ids"),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
