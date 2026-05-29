# -*- coding: utf-8 -*-
"""
对比算法泛化实验脚本
仿照 train_comparisons.py，适配预划分的训练/测试集（无五折）
"""
import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_utils import compute_metrics
from gblup import GBLUP
from random_forest import RandomForest
from knn import KNN
from gbt import GBT

EXP_DIR_MAP = {
    'unseen_env': ['unseen_environment_data%d' % i for i in range(1, 6)],
    'unseen_geno': ['unseen_genotype_data%d' % i for i in range(1, 6)],
    'unseen_both': ['unseen_both_data%d' % i for i in range(1, 4)],
}

TRAIT_COL_MAP = {
    'trait1': 'Yield', 'trait2': 'Grain Moisture', 'trait3': 'Pollen_DAP_days',
    'trait4': 'Silk_DAP_days', 'trait5': 'Plant_Height_cm', 'trait6': 'Ear_Height_cm',
}


def parse_args():
    parser = argparse.ArgumentParser(description='对比算法泛化实验')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['unseen_env', 'unseen_geno', 'unseen_both', 'all'])
    parser.add_argument('--use_env', action='store_true', default=True)
    parser.add_argument('--no_env', action='store_true')
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--test_mode', action='store_true')
    return parser.parse_args()


def load_split(genotype_path, env_path, train_csv, test_csv, trait_col):
    """加载单个划分，返回 (X_snp_train, X_env_train, y_train, X_snp_test, X_env_test, y_test)"""
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    algo_dir = os.path.join(base_dir, 'algorithm')
    sys.path.insert(0, algo_dir)
    from dataset import load_genotype_data, load_environment_data

    hybrid_to_genotype = load_genotype_data(genotype_path)
    env_features = load_environment_data(env_path)

    train_df = pd.read_csv(train_csv, sep=',', dtype=str)
    test_df = pd.read_csv(test_csv, sep=',', dtype=str)

    def build_arrays(df):
        X_snp_list, X_env_list, y_list = [], [], []
        for _, row in df.iterrows():
            env_id = row['Environment']
            hybrid_id = row['Hybrid']
            if env_id not in env_features or hybrid_id not in hybrid_to_genotype:
                continue
            if trait_col not in df.columns or pd.isna(row[trait_col]):
                continue
            X_snp_list.append(hybrid_to_genotype[hybrid_id])
            X_env_list.append(env_features[env_id])
            y_list.append(float(row[trait_col]))
        return np.array(X_snp_list), np.array(X_env_list), np.array(y_list)

    X_snp_train, X_env_train, y_train = build_arrays(train_df)
    X_snp_test, X_env_test, y_test = build_arrays(test_df)
    return X_snp_train, X_env_train, y_train, X_snp_test, X_env_test, y_test


def train_single_model(model_class, model_name, X_snp_train, X_env_train, y_train,
                       X_snp_test, X_env_test, y_test, trait_name, split_name,
                       results_dir, use_env=False, alpha=10.0):
    """对单个划分训练单个模型"""
    start_time = time.time()

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
        y_train_pred = model.predict(X_snp_train, X_env_train)
        y_test_pred = model.predict(X_snp_test, X_env_test)
    else:
        model.fit(X_snp_train, y_train)
        y_train_pred = model.predict(X_snp_train)
        y_test_pred = model.predict(X_snp_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    duration = time.time() - start_time

    # Save predictions (仿照原格式)
    all_predictions = []
    for i in range(len(y_train)):
        all_predictions.append({
            'sample_index': i, 'set': 'train',
            'true_value': float(y_train[i]), 'predicted_value': float(y_train_pred[i])
        })
    for i in range(len(y_test)):
        all_predictions.append({
            'sample_index': i, 'set': 'test',
            'true_value': float(y_test[i]), 'predicted_value': float(y_test_pred[i])
        })

    predictions_df = pd.DataFrame(all_predictions)
    pred_path = os.path.join(results_dir['predictions'],
                             f"{model_name}_{trait_name}_{split_name}_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)

    timing_info = {
        'split': split_name,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        'duration_seconds': duration,
        'duration_minutes': duration / 60
    }

    timing_df = pd.DataFrame([timing_info])
    timing_path = os.path.join(results_dir['cv_results'],
                               f"{model_name}_{trait_name}_{split_name}_timing.csv")
    timing_df.to_csv(timing_path, index=False)

    summary = {
        'model': model_name,
        'trait': trait_name,
        'split': split_name,
        'use_env': use_env,
        'train_mse': train_metrics['mse'],
        'train_r2': train_metrics['r2'],
        'train_pcc': train_metrics['pcc'],
        'test_mse': test_metrics['mse'],
        'test_r2': test_metrics['r2'],
        'test_pcc': test_metrics['pcc'],
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'duration_seconds': duration,
        'timing': timing_info
    }

    summary_path = os.path.join(results_dir['cv_results'],
                                f"{model_name}_{trait_name}_{split_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"  Train R2: {train_metrics['r2']:.4f}, PCC: {train_metrics['pcc']:.4f}")
    print(f"  Test  R2: {test_metrics['r2']:.4f}, PCC: {test_metrics['pcc']:.4f}")
    print(f"  Time: {duration:.2f}s")

    return summary


def main():
    args = parse_args()
    use_env = args.use_env and not args.no_env

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    genotype_path = os.path.join(data_dir, 'genotype.tsv')
    env_path = os.path.join(data_dir, 'Environment_data.csv')

    results_dir = {
        'base': os.path.join(base_dir, 'results_comparison', 'generalization'),
        'cv_results': os.path.join(base_dir, 'results_comparison', 'generalization', 'cv_results'),
        'predictions': os.path.join(base_dir, 'results_comparison', 'generalization', 'predictions'),
    }
    for d in results_dir.values():
        os.makedirs(d, exist_ok=True)

    models = {
        'gblup': GBLUP,
        'rf': RandomForest,
        'knn': KNN,
        'gbt': GBT,
    }

    experiments = list(EXP_DIR_MAP.keys()) if args.experiment == 'all' else [args.experiment]
    all_summaries = []

    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Use Environment: {use_env}")
        print(f"{'='*60}")

        for split_dir_name in EXP_DIR_MAP[exp_name]:
            split_path = os.path.join(data_dir, split_dir_name)
            train_csv = os.path.join(split_path, 'train.csv')
            test_csv = os.path.join(split_path, 'test.csv')

            if not os.path.exists(train_csv) or not os.path.exists(test_csv):
                print(f"  SKIP {split_dir_name}: missing files")
                continue

            for trait_name, trait_col in TRAIT_COL_MAP.items():
                print(f"\n  [{split_dir_name}] [{trait_name}] {trait_col}")

                X_snp_train, X_env_train, y_train, X_snp_test, X_env_test, y_test = load_split(
                    genotype_path, env_path, train_csv, test_csv, trait_col
                )

                if len(y_train) == 0 or len(y_test) == 0:
                    print(f"    EMPTY (train={len(y_train)}, test={len(y_test)})")
                    continue

                for model_name, model_class in models.items():
                    print(f"    --- {model_name.upper()} ---")
                    summary = train_single_model(
                        model_class=model_class, model_name=model_name,
                        X_snp_train=X_snp_train, X_env_train=X_env_train, y_train=y_train,
                        X_snp_test=X_snp_test, X_env_test=X_env_test, y_test=y_test,
                        trait_name=trait_name, split_name=split_dir_name,
                        results_dir=results_dir, use_env=use_env, alpha=args.alpha,
                    )
                    all_summaries.append(summary)

    # Overall summary
    overall_summary = {
        'use_env': use_env,
        'gblup_alpha': args.alpha,
        'experiments': experiments,
        'models_trained': list(models.keys()),
        'all_summaries': all_summaries,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    overall_path = os.path.join(results_dir['cv_results'],
                                f'overall_summary{"_with_env" if use_env else ""}.json')
    with open(overall_path, 'w') as f:
        json.dump(overall_summary, f, indent=4)

    print(f"\n{'='*60}")
    print(f"All training completed!")
    print(f"Overall summary saved: {overall_path}")
    print(f"{'='*60}")

    # Print results summary
    print(f"\nOverall Results (Test R2):")
    for exp_name in experiments:
        print(f"\n[{exp_name}]")
        for trait_name in TRAIT_COL_MAP:
            print(f"  {trait_name}:")
            for model_name in models:
                summaries = [s for s in all_summaries
                            if s['model'] == model_name and s['trait'] == trait_name]
                if summaries:
                    r2s = [s['test_r2'] for s in summaries]
                    print(f"    {model_name.upper():>6s}: avg R2={np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")


if __name__ == '__main__':
    main()
