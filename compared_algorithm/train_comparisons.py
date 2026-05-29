# -*- coding: utf-8 -*-
"""
Main training script for comparison algorithms (GBLUP, RF, KNN, GBT)
"""
import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from data_utils import load_and_prepare_data, get_5fold_split_indices, compute_metrics
from gblup import GBLUP
from random_forest import RandomForest
from knn import KNN
from gbt import GBT


def parse_args():
    parser = argparse.ArgumentParser(description='Train comparison algorithms')
    parser.add_argument('--traits', type=str, default='1-6',
                        help='Traits to train (e.g., "1-6" or "1,3,5")')
    parser.add_argument('--use_env', action='store_true', default=True,
                        help='Use environment features')
    parser.add_argument('--no_env', action='store_true',
                        help='Do not use environment features')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='GBLUP regularization parameter')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run only 1 fold for testing')
    return parser.parse_args()


def train_single_algorithm(model_class, model_name, X_snp, X_env, y, folds, 
                          trait_name, results_dir, use_env=False, alpha=1.0, test_mode=False):
    """
    Train a single algorithm with 5-fold cross validation
    """
    fold_results = []
    all_predictions = []
    fold_timing_info = []
    
    num_folds = 1 if test_mode else 5
    for fold in range(1, num_folds + 1):
        print(f"\n=== {model_name} - Fold {fold}/5 ===")
        
        train_idx, val_idx = folds[fold - 1]
        
        X_snp_train = X_snp[train_idx]
        X_snp_val = X_snp[val_idx]
        X_env_train = X_env[train_idx] if X_env is not None else None
        X_env_val = X_env[val_idx] if X_env is not None else None
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        fold_start_time = time.time()
        
        # Initialize model with tuned parameters
        if model_name == 'gblup':
            model = model_class(use_env=use_env, alpha=alpha)
        elif model_name == 'rf':
            model = model_class(use_env=use_env, n_estimators=100, max_depth=30, 
                               min_samples_split=5, random_state=42)
        elif model_name == 'knn':
            model = model_class(use_env=use_env, n_neighbors=20, weights='distance')
        elif model_name == 'gbt':
            model = model_class(use_env=use_env, n_estimators=100, learning_rate=0.1, 
                               max_depth=5, random_state=42)
        
        if use_env:
            model.fit(X_snp_train, y_train, X_env_train)
        else:
            model.fit(X_snp_train, y_train)
        
        if use_env:
            y_train_pred = model.predict(X_snp_train, X_env_train)
            y_val_pred = model.predict(X_snp_val, X_env_val)
        else:
            y_train_pred = model.predict(X_snp_train)
            y_val_pred = model.predict(X_snp_val)
        
        train_metrics = compute_metrics(y_train, y_train_pred)
        val_metrics = compute_metrics(y_val, y_val_pred)
        
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        
        fold_result = {
            'fold': fold,
            'train_mse': train_metrics['mse'],
            'train_r2': train_metrics['r2'],
            'train_pcc': train_metrics['pcc'],
            'val_mse': val_metrics['mse'],
            'val_r2': val_metrics['r2'],
            'val_pcc': val_metrics['pcc'],
            'duration_seconds': fold_duration,
            'n_train_samples': len(train_idx),
            'n_val_samples': len(val_idx)
        }
        fold_results.append(fold_result)
        
        fold_timing_info.append({
            'fold': fold,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fold_start_time)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fold_end_time)),
            'duration_seconds': fold_duration,
            'duration_minutes': fold_duration / 60
        })
        
        for i, idx in enumerate(train_idx):
            all_predictions.append({
                'sample_index': int(idx),
                'fold': fold,
                'set': 'train',
                'true_value': float(y_train[i]),
                'predicted_value': float(y_train_pred[i])
            })
        
        for i, idx in enumerate(val_idx):
            all_predictions.append({
                'sample_index': int(idx),
                'fold': fold,
                'set': 'val',
                'true_value': float(y_val[i]),
                'predicted_value': float(y_val_pred[i])
            })
        
        print(f"Train R2: {train_metrics['r2']:.4f}, PCC: {train_metrics['pcc']:.4f}")
        print(f"Val   R2: {val_metrics['r2']:.4f}, PCC: {val_metrics['pcc']:.4f}")
        print(f"Fold {fold} time: {fold_duration:.2f}s")
    
    # Save results
    all_predictions_df = pd.DataFrame(all_predictions)
    predictions_output_path = os.path.join(
        results_dir['predictions'], 
        f"{model_name}_{trait_name}_predictions.csv"
    )
    all_predictions_df.to_csv(predictions_output_path, index=False)
    print(f"\nAll predictions saved: {predictions_output_path}")
    
    timing_df = pd.DataFrame(fold_timing_info)
    timing_output_path = os.path.join(
        results_dir['cv_results'], 
        f"{model_name}_{trait_name}_fold_timing.csv"
    )
    timing_df.to_csv(timing_output_path, index=False)
    print(f"Fold timing saved: {timing_output_path}")
    
    summary = {
        'model': model_name,
        'trait': trait_name,
        'use_env': use_env,
        'num_folds': 5,
        'fold_results': fold_results,
        'average_train_mse': np.mean([r['train_mse'] for r in fold_results]),
        'average_train_r2': np.mean([r['train_r2'] for r in fold_results]),
        'average_train_pcc': np.mean([r['train_pcc'] for r in fold_results]),
        'average_val_mse': np.mean([r['val_mse'] for r in fold_results]),
        'average_val_r2': np.mean([r['val_r2'] for r in fold_results]),
        'average_val_pcc': np.mean([r['val_pcc'] for r in fold_results]),
        'std_val_r2': np.std([r['val_r2'] for r in fold_results]),
        'std_val_pcc': np.std([r['val_pcc'] for r in fold_results]),
        'total_duration_seconds': sum([r['duration_seconds'] for r in fold_results]),
        'fold_timing': fold_timing_info
    }
    
    summary_path = os.path.join(
        results_dir['cv_results'], 
        f"{model_name}_{trait_name}_summary.json"
    )
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Model summary saved: {summary_path}")
    
    return summary


def main():
    args = parse_args()
    
    # Determine if we should use environment features
    use_env = args.use_env and not args.no_env
    
    if '-' in args.traits:
        start, end = map(int, args.traits.split('-'))
        trait_list = [f'trait{i}' for i in range(start, end + 1)]
    else:
        trait_list = [f'trait{i}' for i in map(int, args.traits.split(','))]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = {
        'base': os.path.join(base_dir, "results_comparison"),
        'cv_results': os.path.join(base_dir, "results_comparison", "cv_results"),
        'predictions': os.path.join(base_dir, "results_comparison", "predictions")
    }
    
    os.makedirs(results_dir['base'], exist_ok=True)
    os.makedirs(results_dir['cv_results'], exist_ok=True)
    os.makedirs(results_dir['predictions'], exist_ok=True)
    
    models = {
        'gblup': GBLUP,
        'rf': RandomForest,
        'knn': KNN,
        'gbt': GBT
    }
    
    all_summaries = []
    
    for trait in trait_list:
        print(f"\n{'='*60}")
        print(f"Processing Trait: {trait}")
        print(f"Use Environment: {use_env}")
        print(f"GBLUP Alpha: {args.alpha}")
        print(f"{'='*60}")
        
        X_snp, X_env, y = load_and_prepare_data(trait)
        print(f"Final data: {len(y)} samples, {X_snp.shape[1]} SNPs")
        
        folds = get_5fold_split_indices(len(y), random_state=42)
        
        for model_name, model_class in models.items():
            print(f"\n{'-'*60}")
            print(f"Training model: {model_name}")
            print(f"{'-'*60}")
            
            summary = train_single_algorithm(
                model_class=model_class,
                model_name=model_name,
                X_snp=X_snp,
                X_env=X_env,
                y=y,
                folds=folds,
                trait_name=trait,
                results_dir=results_dir,
                use_env=use_env,
                alpha=args.alpha,
                test_mode=args.test_mode
            )
            all_summaries.append(summary)
    
    overall_summary = {
        'use_env': use_env,
        'gblup_alpha': args.alpha,
        'traits_trained': trait_list,
        'models_trained': list(models.keys()),
        'all_summaries': all_summaries,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    env_suffix = '_with_env' if use_env else ''
    overall_summary_path = os.path.join(
        results_dir['cv_results'], 
        f'overall_summary{env_suffix}.json'
    )
    
    with open(overall_summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"All training completed!")
    print(f"Overall summary saved: {overall_summary_path}")
    print(f"{'='*60}")
    
    print(f"\nOverall Results (Average Validation R2):")
    for trait in trait_list:
        print(f"\nTrait {trait}:")
        for model_name in models.keys():
            summary = next(s for s in all_summaries 
                          if s['model'] == model_name and s['trait'] == trait)
            print(f"  {model_name.upper()}: {summary['average_val_r2']:.4f} +/- {summary['std_val_r2']:.4f}")


if __name__ == '__main__':
    main()
