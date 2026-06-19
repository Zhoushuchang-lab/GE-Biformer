# GEBiformer: Gene-Environment Bridge Transformer with Mixture of Experts 🧬

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GEBiformer** is a deep learning framework for genomic prediction that models gene-environment interactions (G×E) using learnable clustering tokenization, cross-modal attention, and Mixture of Experts (MoE). The model is trained on the **G2F dataset** (Genomes to Fields).

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Data Splitting Strategies (CV1-CV4)](#data-splitting-strategies-cv1-cv4)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Introduction

GEBiformer addresses the challenge of predicting complex traits by integrating genotype data (SNP markers) and environmental covariates. The model leverages advanced deep learning techniques to capture intricate gene-environment interactions, enabling more accurate predictions across different environmental conditions and novel genotypes.

---

## ✨ Key Features

- **🧩 Learnable Clustering Tokenization**: Softly clusters SNP markers and environment variables into semantically meaningful tokens with entropy-diversity regularization
- **🔗 Token-wise Cross-Modal Attention**: SNP tokens and environment tokens attend to each other, capturing fine-grained G×E interactions
- **👥 Top-K Mixture of Experts**: 8 expert networks with sparse Top-2 routing for specialized processing of different gene-environment patterns
- **⚡ Dual Attention with GLU Gating**: Fuses independent and cooperative effects for both SNP and environment modalities

---

## 🏗️ Model Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         GEBiformer                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Genotype (N × S)                  Environment (N × E)           │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  SNP Attention Module              Env Attention Module          │
│  ├─ Independent Attention (MLP)    ├─ Independent Attention      │
│  ├─ Cooperative Attention (MHA)    ├─ Cooperative Attention      │
│  └─ GLU Dynamic Gate               └─ GLU Dynamic Gate           │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  Learnable Cluster Tokenizer       Learnable Cluster Tokenizer   │
│  (S → 32 SNP Tokens)                (E → 16 Env Tokens)            │
│       │                                   │                      │
│       └───────────────┬───────────────────┘                      │
│                       ▼                                          │
│           Token-wise Cross-Modal Attention                       │
│            (SNP Tokens ⟷ Env Tokens)                            │
│                       │                                          │
│                       ▼                                          │
│           Mixture of Experts (Top-2 of 8)                        │
│                       │                                          │
│                       ▼                                          │
│              Feature Network [512, 256, 128]                     │
│                       │                                          │
│                       ▼                                          │
│                Phenotype Prediction                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Splitting Strategies (CV1-CV4)

| Strategy | Name | Description | Folds |
|----------|------|-------------|-------|
| **CV1** | Standard 5-fold CV | Random stratified splits | 5 |
| **CV2** | Unseen Environment | Test environments never seen during training | 5 |
| **CV3** | Unseen Genotype | Test genotypes never seen during training | 5 |
| **CV4** | Unseen Both | Simultaneously unseen environments AND genotypes | 5 |

---

## 📁 Project Structure

```
GEBiformer/
├── algorithm/              # Core algorithm
│   ├── config.py           # Hyperparameter configuration
│   ├── model.py            # Model definitions
│   ├── dataset.py          # Data loading & preprocessing
│   ├── train.py            # 5-fold CV training script
│   └── utils.py            # Utilities (EarlyStopping, metrics)
├── code/                   # Experiment scripts
│   ├── create_generalization_splits.py    # Generate CV2/CV3/CV4 splits
│   └── run_generalization_experiments.py  # Run generalization experiments
└── data/                   # Data directory (not tracked)
    ├── genotype.tsv                    # SNP matrix (markers × hybrids)
    ├── Environment_data.csv            # Environment covariates
    ├── Phenotypes.csv                  # Phenotype records
    ├── unseen_environment_data1~5/     # CV2: Unseen environment (5 folds)
    ├── unseen_genotype_data1~5/        # CV3: Unseen genotype (5 folds)
    └── unseen_both_data1~5/            # CV4: Unseen both (5 folds)
```

---

## 🔧 Installation

### Create Environment

```bash
conda create -n gebiformer python=3.12
conda activate gebiformer

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```

---

## 📁 Data Preparation

### Dataset Overview

The **G2F dataset** (Genomes to Fields) consists of:
- **1 Phenotype dataset**: Multi-environment trial data for 6 traits
- **6 Environmental covariates**: Weather and soil conditions across locations
- **5 Genotype datasets**: SNP markers for hybrid lines

### Required Files

Place the following files in the `data/` directory:

| File | Format | Description |
|------|--------|-------------|
| `genotype.tsv` | TSV | SNP matrix with rows as markers and columns as hybrids (0/1/2 encoding, missing=-1) |
| `Environment_data.csv` | CSV | Environment covariates with rows as variables and columns as environments |
| `Phenotypes.csv` | CSV | Phenotype records with columns: `Environment`, `Hybrid`, and 6 trait columns |

### Data Format Examples

#### genotype.tsv
```
SNP_ID	Hybrid_1	Hybrid_2	Hybrid_3
SNP_001	0	1	2
SNP_002	1	2	0
SNP_003	2	0	1
```

#### Environment_data.csv
```
Variable	ENV_001	ENV_002	ENV_003
Temperature	25.3	26.1	24.8
Rainfall	65	72	68
Humidity	10.5	8.2	12.3
```

#### Phenotypes.csv
```
Environment	Hybrid	Yield	Grain Moisture	Pollen_DAP_days	Silk_DAP_days	Plant_Height_cm	Ear_Height_cm
ENV_001	Hybrid_1	10.5	18.2	65	68	180	95
ENV_001	Hybrid_2	12.3	19.1	67	70	185	98
ENV_002	Hybrid_1	11.2	17.8	66	69	178	93
```

**Traits**: `Yield`, `Grain Moisture`, `Pollen_DAP_days`, `Silk_DAP_days`, `Plant_Height_cm`, `Ear_Height_cm`

---

## 🚀 Usage

### 1. Standard 5-fold Cross-Validation (CV1)

Train the model with 5-fold cross-validation on all traits:

```bash
cd algorithm
python train.py --traits 1-6
```

Train specific traits:

```bash
# Train only Yield (trait1) and Plant Height (trait5)
python train.py --traits 1,5
```

### 2. Generalization Experiments (CV2/CV3/CV4)

**Step 1: Generate data splits**

```bash
# Generate splits for all scenarios
python code/create_generalization_splits.py --experiment all

# Or specific scenario
python code/create_generalization_splits.py --experiment unseen_env
```

**Step 2: Run experiments**

```bash
# Run all scenarios
python code/run_generalization_experiments.py --experiment all

# Or specific scenario
python code/run_generalization_experiments.py --experiment unseen_both
```

### 3. Output Directories

| Directory | Description |
|-----------|-------------|
| `results/cv_results/` | Model weights and summaries for CV1 |
| `results/history/` | Training curves per fold |
| `results/predictions/` | Cross-fold predictions |
| `results/generalization/` | Results for CV2/CV3/CV4 |

---

## ⚙️ Hyperparameters

All hyperparameters are configured in `algorithm/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_snp_tokens` | 32 | Number of SNP cluster tokens |
| `num_env_tokens` | 16 | Number of environment cluster tokens |
| `num_experts` | 8 | Number of MoE experts |
| `top_k` | 2 | Number of experts selected per sample |
| `num_heads` | 8 | Number of attention heads |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 3e-4 | Initial learning rate |
| `epochs` | 100 | Maximum training epochs |
| `early_patience` | 30 | Early stopping patience |
| `dropout` | 0.3 | General dropout rate |

---

## 📄 License

MIT License.

---

## 📧 Contact

- Email: sy1302498577@163.com
