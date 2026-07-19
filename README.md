# FM-Optimized Embedding 对比实验

对比 5 种精排模型的直接预测与 FM (Flow Matching) 优化 embedding 后预测的效果。

## 核心思想

改编自 [falcon](../falcon) 的 Flow Matching Teacher 模型，将 FM 用于优化精排模型的 embedding，使其更适合下游 CTR/CVR 预测任务。

```
直接预测:   Features → 精排模型 → CTR/CVR 预测
FM 优化后:  Features → 精排模型 → Embedding → FM 优化 → 优化 Embedding → CTR/CVR 预测
```

FM 权重 (`fm_weight`) 控制流匹配损失与预测损失的平衡:
- `fm_weight ↑`: 更关注 embedding 空间的流形结构
- `fm_weight ↓`: 更关注下游预测任务

## 5 种精排模型

| 模型 | 说明 |
|------|------|
| **DNN** | 基本 Embedding + MLP 模型 |
| **Wide & Deep** | Wide (线性记忆) + Deep (泛化) 组合 |
| **DeepFM** | FM (二阶交叉) + DNN 组合 |
| **DCN** | Cross Network (显式交叉) + Deep 组合 |
| **AutoInt** | Multi-Head Self-Attention 自动特征交互 |

## 数据集

### Ali MAC (alimamaTech/MAC)

首个多归因转化率预测公开数据集:
- 20 个稀疏特征 (7 用户 + 10 物品 + 3 上下文)
- 4 种归因标签 (last_click, first_click, dda, linear)
- 79M 点击, 0.8M 用户

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行对比实验

```bash
# 默认配置 (5 个模型全部测试)
python run_compare.py --config configs/compare_config.yaml

# 指定模型和 FM 权重
python run_compare.py --models dnn deepfm --fm_weights 0.1 0.5 1.0 2.0

# 指定数据集和归因机制
python run_compare.py --dataset alimamaTech/MAC --target last_click

# 调试模式 (限制样本数)
python run_compare.py --max_samples 10000 --models dnn
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/compare_config.yaml` |
| `--models` | 模型列表 | 全部 5 个 |
| `--fm_weights` | FM 权重列表 | `[0.1, 0.5, 1.0, 2.0]` |
| `--dataset` | 数据集名称 | `alimamaTech/MAC` |
| `--target` | 目标归因机制 | `last_click` |
| `--max_samples` | 最大样本数 (调试用) | -1 |
| `--output_dir` | 输出目录 | `./output` |
| `--seed` | 随机种子 | 42 |
| `--device` | 设备 | auto |

## 项目结构

```
kdd_compare_test1/
├── configs/
│   └── compare_config.yaml       # 实验配置
├── data/
│   ├── __init__.py
│   └── mac_dataset.py            # MAC 数据集加载器
├── models/
│   ├── __init__.py
│   ├── base_ranker.py            # 精排模型基类
│   ├── dnn.py                    # DNN
│   ├── wide_deep.py              # Wide & Deep
│   ├── deepfm.py                 # DeepFM
│   ├── dcn.py                    # DCN
│   ├── autoint.py                # AutoInt
│   └── fm_optimizer.py           # FM 优化器 (改编自 falcon)
├── trainers/
│   ├── __init__.py
│   └── compare_trainer.py        # 对比实验训练器
├── utils/
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── metrics.py                # 评估指标 (AUC, GAUC, LogLoss, NDCG)
│   └── seed.py                   # 随机种子
├── run_compare.py                # 主入口
├── requirements.txt
└── README.md
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **AUC** | Area Under ROC Curve |
| **GAUC** | Group AUC (按用户分组) |
| **LogLoss** | Binary Cross-Entropy |
| **NDCG@K** | Normalized Discounted Cumulative Gain |

## FM 优化器原理

FM (Flow Matching) 优化器改编自 falcon 的 Flow Matching Teacher 模型:

1. **前向过程** (OT-CFM): `x_t = (1-t) * x_clean + t * ε`, 学习从干净 embedding 到噪声的速度场
2. **反向过程** (ODE 求解): 从精排模型的 embedding 出发，通过 ODE 求解生成优化的 embedding
3. **训练损失**: `L = fm_weight * L_FM + (1 - fm_weight) * L_BCE`

其中 `L_FM` 是流匹配一致性损失，`L_BCE` 是下游预测的交叉熵损失。
