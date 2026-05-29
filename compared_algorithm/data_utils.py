# -*- coding: utf-8 -*-
"""
Data utilities for comparison algorithms
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_and_prepare_data(trait_name, data_dir=None):
    """
    Load and prepare data for a specific trait
    
    Args:
        trait_name: Name of the trait (e.g., 'trait1')
        data_dir: Path to data directory (optional, auto-detect if None)
    
    Returns:
        X_snp: SNP features (n_samples, n_snps)
        X_env: Environment features (n_samples, n_env)
        y: Target values (n_samples,)
    """
    # Auto-detect data directory if not provided
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
    
    print(f"Data directory: {data_dir}")
    
    # Data paths - use genotype.tsv as genotype file
    genotype_path = os.path.join(data_dir, "genotype.tsv")
    pheno_path = os.path.join(data_dir, "Phenotypes.csv")
    env_path = os.path.join(data_dir, "Environment_data.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    # Check if files exist
    for fpath in [genotype_path, pheno_path, env_path]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    # Load dataset using shared dataset module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithm'))
    from dataset import prepare_dataset
    
    dataset_dict = prepare_dataset(genotype_path, env_path, pheno_path, test_path)
    
    # Combine train and val data
    train_val_data = dataset_dict[trait_name]['train'] + dataset_dict[trait_name]['val']
    
    # Extract features and targets
    X_snp = []
    X_env = []
    y = []
    
    for sample in train_val_data:
        X_snp.append(sample['snp'])
        X_env.append(sample['env'])
        y.append(sample['trait'])
    
    X_snp = np.array(X_snp)
    X_env = np.array(X_env)
    y = np.array(y)
    
    print(f"Loaded {len(y)} samples, {X_snp.shape[1]} SNPs, {X_env.shape[1]} env features")
    
    return X_snp, X_env, y


def get_5fold_split_indices(n_samples, random_state=42):
    """
    Get 5-fold cross-validation indices
    
    Returns:
        list of (train_idx, val_idx) tuples
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = list(kfold.split(range(n_samples)))
    return folds


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics
    
    Returns:
        dict with mse, r2, pcc
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Compute Pearson correlation coefficient
    pcc = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    return {
        'mse': mse,
        'r2': r2,
        'pcc': pcc
    }
