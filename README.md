# Gene-Environment Interaction Attention Model with Mixture of Experts

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This is a gene-environment interaction prediction model based on attention mechanisms and Mixture of Experts (MoE) for predicting complex trait performance.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Hyperparameter Configuration](#hyperparameter-configuration)
- [Output Description](#output-description)
- [Model Variants](#model-variants)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Introduction

This model aims to predict complex crop traits by integrating genotype data (SNP) and environmental covariates. The model adopts the following core technologies:

- **Attention Mechanism**: Captures complex interactions between SNPs and environmental variables
- **MoE Architecture**: Dynamically selects expert networks to handle different gene-environment combinations
- **Cross-Modal Fusion**: Effectively integrates genotype and environment information

## 🏗️ Model Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gene-Environment MoE Model                   │
├─────────────────────────────────────────────────────────────────┤
│  Genotype Input │    Environment Input                         │
│       │         │         │                                    │
│       ▼         │         ▼                                    │
│  SNP Attention  │  Environment Attention                       │
│  (Ind+Coop)     │  (Ind+Coop)                                  │
│       │         │         │                                    │
│       └────┬────┘         │                                    │
│            │               │                                    │
│            ▼               ▼                                    │
│        Token-wise Cross-Modal Fusion                          │
│            │                                                  │
│            ▼                                                  │
│        Mixture of Experts (Top-K)                             │
│            │                                                  │
│            ▼                                                  │
│        Feature Network                                        │
│            │                                                  │
│            ▼                                                  │
│        Phenotype Prediction                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Function |
|-----------|----------|
| **SNP Attention Module** | Processes genotype data, learns independent and cooperative effects between SNPs |
| **Environment Attention Module** | Processes environment data, learns independent and cooperative effects between environmental variables |
| **Token-wise Cross-Modal Fusion** | Performs cross-modal attention fusion on SNP and environment features |
| **MoE Layer** | Dynamically selects expert networks using Top-K gating mechanism |
| **Feature Network** | Feature transformation and integration |
| **Predictor** | Final trait prediction layer |

## 🛠️ Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8 (GPU training recommended)

### Using Conda

```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Create and activate environment
conda env create -f code/environment.yml
conda activate gene-env-moe

# Enter code directory
cd code
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Or
venv\Scripts\activate    # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn matplotlib tqdm
```

## 📊 Data Preparation

### Project Structure

```
project/
├── code/                    # Source code directory
│   ├── config.json          # Configuration file
│   ├── config.py            # Configuration loader
│   ├── dataset.py           # Data processing
│   ├── model.py             # Model definitions
│   ├── train.py             # Training script
│   ├── utils.py             # Utility functions
│   └── environment.yml      # Environment configuration
├── data/                    # Data directory (create manually)
│   ├── genotype.tsv         # Genotype data (0-1-2 encoding)
│   ├── Phenotypes.csv       # Phenotype data (train + validation)
│   ├── Environment_data.csv # Environmental covariate data
│   └── test.csv             # Test set data (optional)
├── gene_env_moe_model/      # Model save directory (auto-created)
└── moe_analysis/            # MoE analysis results directory (auto-created)
```

### Data Format Requirements

#### 1. Genotype Data (`genotype.tsv`)

| SNP_ID | Hybrid_1 | Hybrid_2 | ... | Hybrid_N |
|--------|----------|----------|-----|----------|
| SNP_001 | 0 | 1 | ... | 2 |
| SNP_002 | 1 | 2 | ... | 0 |
| ... | ... | ... | ... | ... |

- First column contains SNP identifiers
- Other columns contain genotype data for each hybrid
- Genotype encoding: 0 (homozygous reference), 1 (heterozygous), 2 (homozygous alternative)
- Missing values should be represented as `-1`

#### 2. Phenotype Data (`Phenotypes.csv`)

| Environment | Hybrid | Yield | Grain Moisture | Pollen_DAP_days | ... |
|-------------|--------|-------|----------------|-----------------|-----|
| ENV_001 | Hybrid_1 | 10.5 | 18.2 | 65 | ... |
| ENV_001 | Hybrid_2 | 12.3 | 19.1 | 67 | ... |
| ... | ... | ... | ... | ... | ... |

#### 3. Environment Data (`Environment_data.csv`)

| Variable | ENV_001 | ENV_002 | ... | ENV_M |
|----------|---------|---------|-----|-------|
| Var_1 | 25.3 | 26.1 | ... | 24.8 |
| Var_2| 65 | 72 | ... | 68 |
| Var_3| 10.5 | 8.2 | ... | 12.3 |
| ... | ... | ... | ... | ... |

#### 4. Test Set Data (`test.csv`) - Optional

The test set format is the same as Phenotypes.csv, used for independent model evaluation.

| Environment | Hybrid | Yield | Grain Moisture | ... |
|-------------|--------|-------|----------------|-----|
| ENV_TEST_01 | Hybrid_101 | 11.2 | 17.8 | ... |
| ENV_TEST_01 | Hybrid_102 | 13.1 | 18.5 | ... |

### Data Splitting

- **Training Set**: 80% of data for model training
- **Validation Set**: 10% of data for hyperparameter tuning and early stopping
- **Test Set**: 10% of data (or external test set) for final model evaluation

If a `test.csv` file is provided, the model will use the external test set for evaluation. Otherwise, the test set will be split from the main dataset.

## 🚀 Usage

### Training the Model

```bash
cd code

# Train full model (traits 1-6)
python train.py --traits 1-6 --model full

# Train specific traits
python train.py --traits 1,3,5 --model full

# Train without MoE for comparison
python train.py --traits 1-6 --model no_moe
```

### Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--traits` | str | `3,4,5,6` | Trait numbers to train, separated by commas or ranges |
| `--model` | str | `full` | Model type: `full`, `no_moe`, `no_effect_sep`, `no_token_fusion` |

### Training Examples

```bash
# Example 1: Train all traits
python train.py --traits 1-7 --model full

# Example 2: Train only Yield and Plant Height
python train.py --traits 1,5 --model full

# Example 3: Train ablation model
python train.py --traits 1-6 --model no_effect_sep
```

### Training Output

During training, the following will be displayed:

```
Selected traits: ['trait1', 'trait2', 'trait3', 'trait4', 'trait5', 'trait6']
Model type: full
Using GPU: NVIDIA GeForce RTX 4090
Test set samples: 156

========== Training Gene-Environment Interaction Attention Model with MoE - Yield ==========
Epoch 1/300 | Train Loss: 5.2341 | Val Loss: 5.1923 | Train R²: 0.1234 | Val R²: 0.1185 | LR: 3.00e-04
Epoch 2/300 | Train Loss: 4.8723 | Val Loss: 4.8210 | Train R²: 0.1876 | Val R²: 0.1821 | LR: 3.00e-04
...

Evaluating Yield test set...
Test R²: 0.6523, PCC: 0.8077
```

## ⚙️ Hyperparameter Configuration

The configuration file is located at `config.json`:

```json
{
    "num_experts": 8,           // Number of MoE experts
    "top_k": 2,                 // Number of experts selected per sample
    "num_heads": 8,             // Number of attention heads
    "batch_size": 64,           // Batch size
    "learning_rate": 0.0003,    // Learning rate
    "epochs": 300,              // Number of training epochs
    "dropout": 0.3,             // Dropout rate
    "moe_hidden_dim": 128,      // MoE hidden dimension
    "fusion_attention_dim": 256,// Fusion attention dimension
    "early_patience": 30,       // Early stopping patience
    "cuda_device": 0,           // GPU device ID
    "test_size": 0.1,           // Test set ratio
    "random_state": 42          // Random seed
}
```

### Key Hyperparameter Description

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `num_experts` | Number of MoE experts | 4-16 |
| `top_k` | Number of experts selected per sample | 1-4 |
| `num_heads` | Number of attention heads | 4-16 |
| `batch_size` | Batch size | 32-256 |
| `learning_rate` | Initial learning rate | 1e-4-1e-3 |
| `epochs` | Number of training epochs | 100-500 |

## 📈 Output Description

### Output File Structure

```
project/
├── gene_env_moe_model/
│   ├── trait1_full_moe_model_03_07_1139.pt    # Model weights
│   ├── trait1_full_moe_model_03_07_1139.txt    # Model information
│   ├── trait1_full_history.csv                  # Training history
│   └── trait1_full_summary.json                 # Training summary (includes test metrics)
└── moe_analysis/
    ├── trait1_full_train_moe_weights.csv        # Training set MoE weights
    ├── trait1_full_val_moe_weights.csv          # Validation set MoE weights
    └── trait1_full_test_moe_weights.csv         # Test set MoE weights (if available)
```

### Training History CSV Format

| epoch | train_loss | val_loss | learning_rate | train_r2 | val_r2 | train_pcc | val_pcc | aux_loss |
|-------|-----------|----------|---------------|----------|--------|-----------|---------|----------|
| 1 | 5.2341 | 5.1923 | 0.0003 | 0.1234 | 0.1185 | 0.3512 | 0.3445 | 0.0012 |
| 2 | 4.8723 | 4.8210 | 0.0003 | 0.1876 | 0.1821 | 0.4332 | 0.4267 | 0.0008 |

### Training Summary JSON Format

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

### MoE Weights CSV Format

| sample_index | hybrid_id | env_id | prediction | expert_0_prob | expert_1_prob | ... | selected_expert_0 | gating_weight_0 | ... |
|--------------|-----------|--------|------------|---------------|---------------|-----|-------------------|-----------------|-----|
| 0 | Hybrid_1 | ENV_001 | 10.5 | 0.1 | 0.8 | ... | 1 | 0.9 | ... |
| 1 | Hybrid_2 | ENV_001 | 12.3 | 0.05 | 0.05 | ... | 5 | 0.85 | ... |

## 🔬 Model Variants

This project provides four model variants for ablation experiments:

| Model | Command Parameter | Description |
|-------|------------------|-------------|
| **Full Model** | `--model full` | Complete model with all components |
| **Without MoE** | `--model no_moe` | Model without MoE layer (single expert) |
| **Without Effect Separation** | `--model no_effect_sep` | Model without independent/cooperative effect separation |
| **Without Token Fusion** | `--model no_token_fusion` | Uses simple concatenation instead of token-wise cross-modal fusion |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

### Code Style Guidelines

- Follow PEP 8 coding standards
- Use type hints
- Add appropriate comments
- Keep code clean and concise

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📧 Contact

For questions or suggestions, please contact us via:

- Submit an Issue
- Email: [sy1302498577@163.com]

---

**Note**: This model is for research purposes only. Please ensure compliance with relevant regulations and ethical requirements before using the data.