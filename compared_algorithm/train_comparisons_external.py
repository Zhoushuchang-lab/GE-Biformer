"""
对比算法外部数据集验证脚本
将 GBLUP、RF、KNN、GBT 四个对比算法应用于外部数据集
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import get_5fold_split_indices, compute_metrics
from gblup import GBLUP
from random_forest import RandomForest
from knn import KNN
from gbt import GBT


def parse_args():
    parser = argparse.ArgumentParser(description='对比算法外部数据集验证')
    parser.add_argument('--use_env', action='store_true', default=True)
    parser.add_argument('--no_env', action='store_true')
    parser.add_argument('--alpha', type=float, default=10.0, help='GBLUP regularization')
    parser.add_argument('--test_mode', action='store_true', help='Run only 1 fold')
    return parser.parse_args()


def load_external_data():
    """Load the converted external dataset"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    converted_dir = os.path.join(base_dir, 'data', 'other_data', 'converted')
    algo_dir = os.path.join(base_dir, 'algorithm')
    sys.path.insert(0, algo_dir)

    from sklearn.preprocessing import RobustScaler, StandardScaler

    print('Loading external data...')

    # Genotype
    genotype_path = os.path.join(converted_dir, 'genotype.tsv')
    genotype_data = pd.read_csv(genotype_path, sep='\t', dtype=str, low_memory=False)
    hybrid_ids = genotype_data.columns[1:].tolist()
    genotype_values = genotype_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32).T
    scaler_g = RobustScaler()
    genotype_scaled = scaler_g.fit_transform(genotype_values)
    hybrid_to_genotype = {hid: genotype_scaled[i] for i, hid in enumerate(hybrid_ids)}

    # Environment
    env_path = os.path.join(converted_dir, 'Environment_data.csv')
    env_data = pd.read_csv(env_path, sep=',', dtype=str)
    env_ids = env_data.columns[1:]
    env_values = env_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32).T
    scaler_e = StandardScaler()
    env_scaled = scaler_e.fit_transform(env_values)
    env_features = {eid: env_scaled[i] for i, eid in enumerate(env_ids)}

    # Phenotype
    pheno_path = os.path.join(converted_dir, 'Phenotypes.csv')
    pheno_data = pd.read_csv(pheno_path, sep=',', dtype=str)
    pheno_data['BLUEs_wtn'] = pd.to_numeric(pheno_data['BLUEs_wtn'], errors='coerce')
    pheno_data = pheno_data.dropna(subset=['BLUEs_wtn'])

    X_snp_list = []
    X_env_list = []
    y_list = []

    for _, row in pheno_data.iterrows():
        env_id = row['Environment']
        hybrid_id = row['Hybrid']
        if env_id in env_features and hybrid_id in hybrid_to_genotype:
            X_snp_list.append(hybrid_to_genotype[hybrid_id])
            X_env_list.append(env_features[env_id])
            y_list.append(float(row['BLUEs_wtn']))

    X_snp = np.array(X_snp_list)
    X_env = np.array(X_env_list)
    y = np.array(y_list)

    print(f'  {len(y)} samples, {X_snp.shape[1]} SNPs, {X_env.shape[1]} env vars')
    return X_snp, X_env, y


def train_single_algorithm(model_class, model_name, X_snp, X_env, y, folds,
                           results_dir, use_env, alpha, test_mode):
    fold_results = []
    all_predictions = []

    num_folds = 1 if test_mode else 5
    for fold in range(1, num_folds + 1):
        print(f'\n=== {model_name.upper()} - Fold {fold}/{num_folds} ===')

        train_idx, val_idx = folds[fold - 1]
        X_snp_train, X_snp_val = X_snp[train_idx], X_snp[val_idx]
        X_env_train, X_env_val = X_env[train_idx], X_env[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_start = time.time()

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
            y_val_pred = model.predict(X_snp_val, X_env_val)
        else:
            model.fit(X_snp_train, y_train)
            y_train_pred = model.predict(X_snp_train)
            y_val_pred = model.predict(X_snp_val)

        train_metrics = compute_metrics(y_train, y_train_pred)
        val_metrics = compute_metrics(y_val, y_val_pred)
        fold_time = time.time() - fold_start

        fold_results.append({
            'fold': fold, 'train_r2': train_metrics['r2'],
            'train_pcc': train_metrics['pcc'], 'val_r2': val_metrics['r2'],
            'val_pcc': val_metrics['pcc'], 'val_mse': val_metrics['mse'],
            'time_seconds': fold_time
        })

        for i, idx in enumerate(val_idx):
            all_predictions.append({
                'sample_index': int(idx), 'fold': fold, 'set': 'val',
                'true_value': float(y_val[i]), 'predicted_value': float(y_val_pred[i])
            })

        print(f'  Train R2: {train_metrics["r2"]:.4f} | Val R2: {val_metrics["r2"]:.4f} | '
              f'Val PCC: {val_metrics["pcc"]:.4f} | Time: {fold_time:.1f}s')

    predictions_df = pd.DataFrame(all_predictions)
    pred_path = os.path.join(results_dir, f'{model_name}_external_predictions.csv')
    predictions_df.to_csv(pred_path, index=False)

    summary = {
        'model': model_name, 'dataset': 'external',
        'use_env': use_env, 'num_folds': num_folds,
        'fold_results': fold_results,
        'average_val_r2': float(np.mean([r['val_r2'] for r in fold_results])),
        'std_val_r2': float(np.std([r['val_r2'] for r in fold_results])),
        'average_val_pcc': float(np.mean([r['val_pcc'] for r in fold_results])),
        'total_time_seconds': float(sum(r['time_seconds'] for r in fold_results))
    }

    summary_path = os.path.join(results_dir, f'{model_name}_external_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'  Summary saved: {summary_path}')

    return summary


def main():
    args = parse_args()
    use_env = args.use_env and not args.no_env

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results_comparison', 'external')
    os.makedirs(results_dir, exist_ok=True)

    print('=' * 60)
    print('对比算法 - 外部数据集泛化能力验证')
    print('=' * 60)

    X_snp, X_env, y = load_external_data()
    print(f'\nData: {len(y)} samples, {X_snp.shape[1]} SNPs, {X_env.shape[1]} env')
    print(f'Trait stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}, '
          f'min={np.min(y):.2f}, max={np.max(y):.2f}')

    folds = get_5fold_split_indices(len(y), random_state=42)

    models = {
        'gblup': GBLUP,
        'rf': RandomForest,
        'knn': KNN,
        'gbt': GBT
    }

    all_summaries = []
    for model_name, model_class in models.items():
        print(f'\n{"-" * 60}')
        print(f'Training: {model_name.upper()}')
        print(f'{"-" * 60}')
        summary = train_single_algorithm(
            model_class, model_name, X_snp, X_env, y,
            folds, results_dir, use_env, args.alpha, args.test_mode
        )
        all_summaries.append(summary)

    print('\n' + '=' * 60)
    print('外部数据集验证结果汇总')
    print('=' * 60)
    for s in all_summaries:
        print(f"  {s['model'].upper():5s}: Val R2={s['average_val_r2']:.4f} +/- "
              f"{s['std_val_r2']:.4f}, Val PCC={s['average_val_pcc']:.4f}, "
              f"Time={s['total_time_seconds']:.0f}s")

    overall_path = os.path.join(results_dir, 'overall_external_summary.json')
    with open(overall_path, 'w') as f:
        json.dump({'models': all_summaries, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=4)
    print(f'\nOverall summary: {overall_path}')


if __name__ == '__main__':
    main()
