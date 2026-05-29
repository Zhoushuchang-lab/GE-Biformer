# GEBiformer: Gene-Environment Bridge Transformer with Mixture of Experts for Genomic Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GEBiformer** is a deep learning framework for genomic prediction that models gene-environment interactions (G×E) using learnable clustering tokenization, cross-modal attention, and Mixture of Experts (MoE).

---

## Key Innovations

1. **Learnable Clustering Tokenization** — Instead of fixed segment-based tokenization, we use learnable prototype vectors that softly cluster SNP markers and environment variables into compact, semantically meaningful tokens. An entropy-diversity regularization ensures balanced and discriminative token assignments.

2. **Token-wise Cross-Modal Attention** — SNP tokens and environment tokens attend to each other via multi-head cross-attention, capturing fine-grained G×E interactions at the token level.

3. **Top-K Mixture of Experts (MoE)** — 8 expert networks with sparse Top-2 routing and load-balancing auxiliary loss, enabling specialized processing of different gene-environment patterns.

4. **Dual Attention with GLU Gating** — Independent (per-feature MLP) and cooperative (multi-head self-attention) effects are fused via a learnable GLU gate for both SNP and environment modalities.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         GEBiformer                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Genotype (N × S)                  Environment (N × E)           │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  SNP Attention Module              Env Attention Module          │
│  ├─ Independent Attention (MLP)    ├─ Independent Attention       │
│  ├─ Cooperative Attention (MHA)    ├─ Cooperative Attention       │
│  └─ GLU Dynamic Gate               └─ GLU Dynamic Gate           │
│       │                                   │                      │
│       ▼                                   ▼                      │
│  Learnable Cluster Tokenizer       Learnable Cluster Tokenizer   │
│  (S → 8 SNP Tokens)                (E → 4 Env Tokens)            │
│       │                                   │                      │
│       └───────────────┬───────────────────┘                      │
│                       ▼                                          │
│           Token-wise Cross-Modal Attention                       │
│            (SNP Tokens ⟷ Env Tokens)                             │
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

## Project Structure

```
GEBiformer/
│
├── algorithm/                          # Core algorithm module
│   ├── config.py                       # Hyperparameter configuration
│   ├── model.py                        # Model definitions (10 classes)
│   ├── dataset.py                      # Data loading & preprocessing
│   ├── train.py                        # 5-fold CV training script
│   ├── utils.py                        # Utilities (EarlyStopping, metrics, etc.)
│   ├── environment.yml                 # Conda environment
│   └── README_CN.md                    # Chinese documentation
│
├── compared_algorithm/                 # Baseline comparison algorithms
│   ├── gblup.py                        # GBLUP implementation
│   ├── random_forest.py                # Random Forest
│   ├── knn.py                          # K-Nearest Neighbors
│   ├── gbt.py                          # Gradient Boosting Trees
│   ├── data_utils.py                   # Shared data loading utilities
│   ├── train_comparisons.py            # 5-fold CV training for baselines
│   └── train_comparisons_generalization.py  # Generalization experiment training
│
├── code/                               # Auxiliary experiment scripts
│   ├── create_generalization_splits.py     # Generate train/test splits for unseen scenarios
│   ├── run_generalization_experiments.py   # Run GEBiformer on generalization splits
│   ├── convert_external_data.py            # Convert external dataset to compatible format
│   ├── run_external_validation.py          # External dataset validation
│   ├── generate_tokenization_diagram.py    # Generate DrawIO tokenization diagram
│   └── analysis_token_biology.py           # Biological interpretation of learned tokens
│
├── data/                               # Data directory (not tracked by git)
│   ├── genotype.tsv                    # Genotype matrix (SNPs × hybrids)
│   ├── Environment_data.csv            # Environment covariates
│   ├── Phenotypes.csv                  # Phenotype records (6 traits)
│   ├── other_data/                     # External validation dataset
│   ├── unseen_environment_data1~5/     # Generalization: unseen environments (5 folds)
│   ├── unseen_genotype_data1~5/        # Generalization: unseen genotypes (5 folds)
│   └── unseen_both_data1~3/            # Generalization: unseen environments + genotypes (3 folds)
│
├── results/                            # GEBiformer results
│   ├── cv_results/                     # 5-fold CV model weights & summaries
│   ├── history/                        # Training curves per fold
│   ├── predictions/                    # Cross-fold predictions
│   ├── token_analysis/                 # Token biology analysis figures
│   ├── figures/                        # Paper figures
│   ├── generalization/                 # Generalization experiment outputs
│   └── external_validation/            # External validation outputs
│
├── results_comparison/                 # Baseline comparison results
│   ├── cv_results/                     # Per-algorithm per-trait summaries
│   ├── predictions/                    # Per-algorithm predictions
│   ├── external/                       # External validation results
│   └── generalization/                 # Generalization experiment outputs
│
├── .trae/specs/                        # Experiment specification documents
│   ├── external-dataset-validation/
│   ├── generalization-experiments/
│   └── tokenization-diagram/
│
├── .gitignore
└── README.md
```

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (recommended for GPU training)

### Using Conda

```bash
git clone <repository-url>
cd GEBiformer

conda env create -f algorithm/environment.yml
conda activate gebiformer
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn matplotlib tqdm
```

---

## Data Preparation

Place the following files in the `data/` directory:

| File | Format | Description |
|------|--------|-------------|
| `genotype.tsv` | TSV | SNP matrix: rows = markers, columns = hybrids. Values: 0/1/2, missing = -1 |
| `Environment_data.csv` | CSV | Environment covariates: rows = variables, columns = environments |
| `Phenotypes.csv` | CSV | Columns: `Environment`, `Hybrid`, and 6 trait columns |

**Traits**: `Yield`, `Grain Moisture`, `Pollen_DAP_days`, `Silk_DAP_days`, `Plant_Height_cm`, `Ear_Height_cm`

---

## Usage

### 1. Standard 5-fold Cross-Validation

```bash
cd algorithm

# Train all 6 traits with 5-fold CV
python train.py --traits 1-6

# Train specific traits
python train.py --traits 1,3,5
```

Outputs saved to `results/cv_results/`, `results/history/`, `results/predictions/`.

### 2. Generalization Experiments

GEBiformer supports three generalization scenarios designed to rigorously test model robustness:

| Scenario | Description | Folds |
|----------|-------------|-------|
| `unseen_env` | Test on completely held-out environments | 5 |
| `unseen_geno` | Test on completely held-out genotypes | 5 |
| `unseen_both` | Test on simultaneously unseen environments AND genotypes | 3 |

**Step 1: Generate data splits**

```bash
python code/create_generalization_splits.py --experiment all
```

**Step 2: Run GEBiformer on generalization splits**

```bash
# All three scenarios
nohup python -u code/run_generalization_experiments.py --experiment all \
    > results/generalization/gebiformer_all.log 2>&1 &

# Or individually
nohup python -u code/run_generalization_experiments.py --experiment unseen_env \
    > results/generalization/gebiformer_env.log 2>&1 &
```

**Step 3: Run comparison algorithms**

```bash
# All scenarios (GBLUP + RF + KNN + GBT)
nohup python -u compared_algorithm/train_comparisons_generalization.py --experiment all \
    > results_comparison/generalization/comparisons_all.log 2>&1 &
```

### 3. External Dataset Validation

```bash
# Convert external data to compatible format
python code/convert_external_data.py

# Run GEBiformer external validation
python code/run_external_validation.py

# Run comparison algorithms on external data
python compared_algorithm/train_comparisons_external.py
```

### 4. Token Biology Analysis

```bash
# Analyze biological meaning of learned tokens
python code/analysis_token_biology.py
```

Outputs include permutation test p-values, token-SNP chromosome distributions, and enrichment results in `results/token_analysis/`.

---

## Hyperparameters

Configured in [algorithm/config.py](algorithm/config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_snp_tokens` | 8 | Number of SNP cluster tokens |
| `num_env_tokens` | 4 | Number of environment cluster tokens |
| `clustering_temperature` | 0.1 | Softness of token assignment |
| `num_experts` | 8 | Number of MoE experts |
| `top_k` | 2 | Top-K routing in MoE |
| `num_heads` | 8 | Multi-head attention heads |
| `fusion_attention_dim` | 256 | Cross-modal fusion dimension |
| `dropout` | 0.3 | Dropout rate |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 3e-4 | Initial learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `early_patience` | 30 | Early stopping patience |
| `epochs` | 100 | Maximum training epochs |

---

## Comparison Algorithms

Four classical genomic prediction methods are implemented for benchmarking:

| Algorithm | Implementation | Key Params |
|-----------|---------------|------------|
| **GBLUP** | Ridge regression on genomic relationship matrix | α = 10.0 |
| **Random Forest** | 100 trees with environment features | n_estimators=100, max_depth=30 |
| **KNN** | Distance-weighted k-nearest neighbors | n_neighbors=20 |
| **GBT** | Gradient boosting trees | n_estimators=100, lr=0.1 |

All baselines use identical data splits as GEBiformer for fair comparison.

---

## Results

### Standard 5-fold CV (full dataset)

| Trait | Train R² | Val R² | PCC |
|-------|----------|--------|-----|
| Yield | ~0.89 | ~0.59 | ~0.77 |
| Grain Moisture | ~0.93 | ~0.76 | ~0.87 |
| Pollen_DAP_days | ~0.92 | ~0.73 | ~0.85 |
| Silk_DAP_days | ~0.91 | ~0.73 | ~0.85 |
| Plant_Height_cm | ~0.83 | ~0.66 | ~0.81 |
| Ear_Height_cm | ~0.88 | ~0.68 | ~0.80 |

### Generalization: Unseen Environments (GEformer from GE/ archive)

| Trait | GEBiformer R² | GBT R² | RF R² |
|-------|:------------:|:------:|:-----:|
| Yield | 0.33 | 0.21 | 0.17 |
| Grain Moisture | 0.46 | 0.08 | -0.09 |
| Pollen_DAP_days | 0.49 | 0.96 | 0.95 |
| Silk_DAP_days | 0.48 | 0.96 | 0.95 |
| Plant_Height_cm | 0.18 | 0.59 | 0.57 |
| Ear_Height_cm | 0.08 | 0.57 | 0.57 |

*Full generalization results pending on the new experiment framework.*

---

## Model Variants (Ablation)

Three ablated variants are available for component analysis:

| Variant | `model.py` Class | Description |
|---------|-----------------|-------------|
| **Full** | `GeneEnvAttentionModelWithMoE` | Complete model with all components |
| **No MoE** | `GeneEnvAttentionModelWithoutMoE` | Removes the MoE layer |
| **No Token Fusion** | `GeneEnvAttentionModelWithoutTokenFusion` | Simple concatenation instead of cross-modal attention |
| **No EffectSep** | `GeneEnvAttentionModelWithoutEffectSeparation` | Removes independent/cooperative attention separation |

---

## Citation

If you use this work, please cite:

```bibtex
@article{GEBiformer2025,
  title={GEBiformer: Gene-Environment Bridge Transformer with Mixture of Experts for Genomic Prediction},
  author={},
  journal={},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Submit issues via GitHub Issues
- Email: sy1302498577@163.com
