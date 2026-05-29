# Comparison Algorithms

This directory contains implementations of four comparison algorithms:

1. **GBLUP** - Genomic Best Linear Unbiased Prediction
2. **RF** - Random Forest
3. **KNN** - K-Nearest Neighbors
4. **GBT** - Gradient Boosting Trees

## Directory Structure

```
compared_algorithm/
├── __init__.py
├── data_utils.py        # Data loading and cross-validation utilities
├── gblup.py            # GBLUP implementation
├── random_forest.py    # Random Forest implementation
├── knn.py              # KNN implementation
├── gbt.py              # GBT implementation
└── train_comparisons.py # Main training script
```

## Usage

### On Server (Recommended)

```bash
cd fsas/compared_algorithm

# Train all 4 models on traits 1-6 (SNP only)
nohup python -u train_comparisons.py --traits 1-6 > comparisons.log 2>&1 &

# Train with environment features
nohup python -u train_comparisons.py --traits 1-6 --use_env > comparisons_env.log 2>&1 &

# Train specific traits
nohup python -u train_comparisons.py --traits 1,3,5 > comparisons.log 2>&1 &
```

### Monitor Training

```bash
# View real-time log
tail -f comparisons.log

# View full log
cat comparisons.log
```

## Output Files

Results are saved to `../results_comparison/`:

```
results_comparison/
├── cv_results/
│   ├── {model}_{trait}_summary.json      # Model summary for each trait
│   ├── {model}_{trait}_fold_timing.csv  # Fold timing information
│   └── overall_summary.json              # Overall results summary
└── predictions/
    └── {model}_{trait}_predictions.csv   # All fold predictions
```

## Algorithm Details

### GBLUP
- Uses genomic relationship matrix (GRM)
- Supports environment features integration
- Uses heritability parameter (h² = 0.5 by default)

### Random Forest
- 100 trees by default
- Supports environment features integration
- Parallel training (`n_jobs=-1`)

### KNN
- 5 neighbors by default
- Distance-based weighting
- Euclidean distance metric
- Supports environment features integration

### GBT
- 100 boosting stages by default
- Learning rate = 0.1
- Max tree depth = 3
- Supports environment features integration

## Key Features

- **Same 5-fold splits** for all models (random_state=42)
- **Same data preprocessing** as main algorithm
- **Predictions recorded** for all samples across folds
- **Timing information** for each fold
- **Comprehensive metrics**: MSE, R², Pearson correlation coefficient
