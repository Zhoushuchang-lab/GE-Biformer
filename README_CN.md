# Bidirectional Cross Attention Integrates Genomic and Enviromic Data for Improved Genotype by Environment Prediction in Maize

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

这是一个基于注意力机制和Mixture of Experts (MoE)的基因-环境交互预测模型，用于预测复杂性状的表现。

## 📋 目录

- [项目简介](#项目简介)
- [模型架构](#模型架构)
- [安装指南](#安装指南)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [超参数配置](#超参数配置)
- [输出说明](#输出说明)
- [模型变体](#模型变体)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🌟 项目简介

本模型旨在通过整合基因型数据（SNP）和环境协变量，预测作物的复杂性状。模型采用以下核心技术：

- **注意力机制**：捕捉SNP之间和环境变量之间的复杂交互
- **MoE架构**：动态选择专家网络处理不同的基因-环境组合
- **跨模态融合**：有效整合基因型和环境信息

## 🏗️ 模型架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gene-Environment MoE Model                   │
├─────────────────────────────────────────────────────────────────┤
│  Genotype Input │    Environment Input                          │
│       │         │         │                                     │
│       ▼         │         ▼                                     │
│  SNP Attention  │  Environment Attention                        │
│  (Ind+Coop)     │  (Ind+Coop)                                   │
│       │         │         │                                     │
│       └────┬────┘         │                                     │
│            │              │                                     │
│            ▼              ▼                                     │
│        Token-wise Cross-Modal Fusion                            │
│            │                                                    │
│            ▼                                                    │
│        Mixture of Experts (Top-K)                               │
│            │                                                    │
│            ▼                                                    │
│        Feature Network                                          │
│            │                                                    │
│            ▼                                                    │
│        Phenotype Prediction                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 功能 |
|------|------|
| **SNP Attention Module** | 处理基因型数据，学习SNP间的独立效应和协同效应 |
| **Environment Attention Module** | 处理环境数据，学习环境变量间的独立效应和协同效应 |
| **Token-wise Cross-Modal Fusion** | 将SNP和环境特征进行跨模态注意力融合 |
| **MoE Layer** | 使用Top-K门控机制动态选择专家网络 |
| **Feature Network** | 特征转换和整合 |
| **Predictor** | 最终性状预测层 |

## 🛠️ 安装指南

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8 (推荐使用GPU训练)

### 使用conda安装

```bash
# 克隆仓库
git clone <repository-url>
cd <repository-name>

# 创建并激活环境
conda env create -f code/environment.yml
conda activate gene-env-moe

# 进入代码目录
cd code
```

### 使用pip安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate    # Windows

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn matplotlib tqdm
```

## 📊 数据准备

### 项目目录结构

```
project/
├── code/                    # 源代码目录
│   ├── config.json          # 配置文件
│   ├── config.py            # 配置加载
│   ├── dataset.py           # 数据处理
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练脚本
│   ├── utils.py             # 工具函数
│   └── environment.yml      # 环境配置
├── data/                    # 数据目录（需自行创建）
│   ├── genotype.tsv         # 基因型数据（0-1-2编码）
│   ├── Phenotypes.csv       # 表型数据（训练+验证）
│   ├── Environment_data.csv # 环境协变量数据
│   └── test.csv             # 测试集数据（可选）
├── gene_env_moe_model/      # 模型保存目录（自动创建）
└── moe_analysis/            # MoE分析结果目录（自动创建）
```

### 数据格式要求

#### 1. 基因型数据 (`genotype.tsv`)

| SNP_ID | Hybrid_1 | Hybrid_2 | ... | Hybrid_N |
|--------|----------|----------|-----|----------|
| SNP_001 | 0 | 1 | ... | 2 |
| SNP_002 | 1 | 2 | ... | 0 |
| ... | ... | ... | ... | ... |

- 第一列为SNP标识符
- 其他列为杂交种的基因型
- 基因型编码：0（纯合参考）、1（杂合）、2（纯合替代）
- 缺失值用 `-1` 表示

#### 2. 表型数据 (`Phenotypes.csv`)

| Environment | Hybrid | Yield | Grain Moisture | Pollen_DAP_days | ... |
|-------------|--------|-------|----------------|-----------------|-----|
| ENV_001 | Hybrid_1 | 10.5 | 18.2 | 65 | ... |
| ENV_001 | Hybrid_2 | 12.3 | 19.1 | 67 | ... |
| ... | ... | ... | ... | ... | ... |

#### 3. 环境数据 (`Environment_data.csv`)

| Variable | ENV_001 | ENV_002 | ... | ENV_M |
|----------|---------|---------|-----|-------|
| Var_1 | 25.3 | 26.1 | ... | 24.8 |
| Var_2| 65 | 72 | ... | 68 |
| Var_3| 10.5 | 8.2 | ... | 12.3 |
| ... | ... | ... | ... | ... |
#### 4. 测试集数据 (`test.csv`) - 可选

测试集格式与Phenotypes.csv相同，用于独立评估模型性能。

| Environment | Hybrid | Yield | Grain Moisture | ... |
|-------------|--------|-------|----------------|-----|
| ENV_TEST_01 | Hybrid_101 | 11.2 | 17.8 | ... |
| ENV_TEST_01 | Hybrid_102 | 13.1 | 18.5 | ... |

### 数据划分

- **训练集**: 80% 的数据用于模型训练
- **验证集**: 10% 的数据用于超参数调优和早停
- **测试集**: 10% 的数据（或外部测试集）用于最终模型评估

如果提供了 `test.csv` 文件，模型将使用外部测试集进行评估。否则，测试集将从主数据集按比例划分。

## 🚀 使用方法

### 训练模型

```bash
cd code

# 训练完整模型（trait1-6）
python train.py --traits 1-6 --model full

# 训练特定性状
python train.py --traits 1,3,5 --model full

# 训练无MoE版本进行对比
python train.py --traits 1-6 --model no_moe
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--traits` | str | `3,4,5,6` | 要训练的性状编号，用逗号分隔或范围表示 |
| `--model` | str | `full` | 模型类型：`full`, `no_moe`, `no_effect_sep`, `no_token_fusion` |

### 训练示例

```bash
# 示例1：训练所有性状
python train.py --traits 1-7 --model full

# 示例2：只训练产量和株高
python train.py --traits 1,5 --model full

# 示例3：训练消融实验模型
python train.py --traits 1-6 --model no_effect_sep
```

### 训练输出

训练过程中会显示：

```
选择的性状: ['trait1', 'trait2', 'trait3', 'trait4', 'trait5', 'trait6']
模型类型: full
使用GPU: NVIDIA GeForce RTX 4090
测试集样本数: 156

========== 训练Gene-Environment Interaction Attention Model with MoE - Yield ==========
Epoch 1/300 | Train Loss: 5.2341 | Val Loss: 5.1923 | Train R²: 0.1234 | Val R²: 0.1185 | LR: 3.00e-04
Epoch 2/300 | Train Loss: 4.8723 | Val Loss: 4.8210 | Train R²: 0.1876 | Val R²: 0.1821 | LR: 3.00e-04
...

评估Yield测试集...
测试集 R²: 0.6523, PCC: 0.8077
```

## ⚙️ 超参数配置

配置文件位于 `config.json`：

```json
{
    "num_experts": 8,           // MoE专家数量
    "top_k": 2,                 // Top-K选择的专家数
    "num_heads": 8,             // 注意力头数
    "batch_size": 64,           // 批次大小
    "learning_rate": 0.0003,    // 学习率
    "epochs": 300,              // 训练轮数
    "dropout": 0.3,             // Dropout率
    "moe_hidden_dim": 128,      // MoE隐藏层维度
    "fusion_attention_dim": 256,// 融合注意力维度
    "early_patience": 30,       // 早停耐心值
    "cuda_device": 0,           // GPU设备编号
    "test_size": 0.1,           // 测试集比例
    "random_state": 42          // 随机种子
}
```

### 关键超参数说明

| 参数 | 说明 | 建议范围 |
|------|------|----------|
| `num_experts` | MoE专家数量 | 4-16 |
| `top_k` | 每个样本选择的专家数 | 1-4 |
| `num_heads` | 注意力头数 | 4-16 |
| `batch_size` | 批次大小 | 32-256 |
| `learning_rate` | 初始学习率 | 1e-4-1e-3 |
| `epochs` | 训练轮数 | 100-500 |

## 📈 输出说明

### 输出文件结构

```
project/
├── gene_env_moe_model/
│   ├── trait1_full_moe_model_03_07_1139.pt    # 模型权重
│   ├── trait1_full_moe_model_03_07_1139.txt    # 模型信息
│   ├── trait1_full_history.csv                  # 训练历史
│   └── trait1_full_summary.json                 # 训练摘要（含测试集指标）
└── moe_analysis/
    ├── trait1_full_train_moe_weights.csv        # 训练集MoE权重
    ├── trait1_full_val_moe_weights.csv          # 验证集MoE权重
    └── trait1_full_test_moe_weights.csv         # 测试集MoE权重（如有）
```

### 训练历史CSV格式

| epoch | train_loss | val_loss | learning_rate | train_r2 | val_r2 | train_pcc | val_pcc | aux_loss |
|-------|-----------|----------|---------------|----------|--------|-----------|---------|----------|
| 1 | 5.2341 | 5.1923 | 0.0003 | 0.1234 | 0.1185 | 0.3512 | 0.3445 | 0.0012 |
| 2 | 4.8723 | 4.8210 | 0.0003 | 0.1876 | 0.1821 | 0.4332 | 0.4267 | 0.0008 |

### 训练摘要JSON格式

```json
{
    "best_epoch": 156,
    "best_val_loss": 2.3456,
    "train_r2": 0.7890,
    "val_r2": 0.6789,
    "test_r2": 0.6523,
    "test_pcc": 0.8077,
    "model_path": "../gene_env_moe_model/trait1_full_moe_model_03_07_1139.pt"
}
```

### MoE权重CSV格式

| sample_index | hybrid_id | env_id | prediction | expert_0_prob | expert_1_prob | ... | selected_expert_0 | gating_weight_0 | ... |
|--------------|-----------|--------|------------|---------------|---------------|-----|-------------------|-----------------|-----|
| 0 | Hybrid_1 | ENV_001 | 10.5 | 0.1 | 0.8 | ... | 1 | 0.9 | ... |
| 1 | Hybrid_2 | ENV_001 | 12.3 | 0.05 | 0.05 | ... | 5 | 0.85 | ... |

## 🔬 模型变体

本项目提供四种模型变体用于消融实验：

| 模型 | 命令参数 | 说明 |
|------|----------|------|
| **完整模型** | `--model full` | 包含所有组件的完整模型 |
| **无MoE** | `--model no_moe` | 移除MoE层，使用单一专家 |
| **无效应分离** | `--model no_effect_sep` | 移除独立效应和协同效应的分离机制 |
| **无Token融合** | `--model no_token_fusion` | 使用简单拼接代替Token化跨模态融合 |

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/your-feature`)
3. 提交修改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

### 代码规范

- 遵循PEP 8编码规范
- 使用类型提示
- 添加适当的注释
- 保持代码简洁清晰

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。



## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至：[sy1302498577@163.com]

---

**注意**：本模型仅供研究使用，使用前请确保数据使用符合相关法规和伦理要求
