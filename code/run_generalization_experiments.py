# -*- coding: utf-8 -*-
"""
GEBiformer 泛化实验脚本
直接复用 train.py 中的完整训练逻辑，仅替换数据加载为预划分的 train/test
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO_DIR = os.path.join(BASE_DIR, 'algorithm')
sys.path.insert(0, ALGO_DIR)

from dataset import GeneEnvDataset, load_genotype_data, load_environment_data
from model import GeneEnvAttentionModelWithMoE
from config import config
from utils import EarlyStopping

TRAIT_COL_MAP = {
    'trait1': 'Yield', 'trait2': 'Grain Moisture', 'trait3': 'Pollen_DAP_days',
    'trait4': 'Silk_DAP_days', 'trait5': 'Plant_Height_cm', 'trait6': 'Ear_Height_cm',
}
EXP_DIR_MAP = {
    'unseen_env': ['unseen_environment_data%d' % i for i in range(1, 6)],
    'unseen_geno': ['unseen_genotype_data%d' % i for i in range(1, 6)],
    'unseen_both': ['unseen_both_data%d' % i for i in range(1, 4)],
}


def load_experiment_data(genotype_path, env_path, train_csv, test_csv, trait_key):
    """完全复用 prepare_dataset 的数据加载逻辑，但使用预划分的 train/test"""
    hybrid_to_genotype = load_genotype_data(genotype_path)
    env_features = load_environment_data(env_path)
    col_name = TRAIT_COL_MAP[trait_key]

    train_data = []
    test_data = []

    import pandas as pd
    train_df = pd.read_csv(train_csv, sep=',', dtype=str)
    test_df = pd.read_csv(test_csv, sep=',', dtype=str)

    for df, target_list in [(train_df, train_data), (test_df, test_data)]:
        for _, row in df.iterrows():
            env_id = row['Environment']
            hybrid_id = row['Hybrid']
            if env_id not in env_features or hybrid_id not in hybrid_to_genotype:
                continue
            if col_name in df.columns and not pd.isna(row[col_name]):
                target_list.append({
                    'hybrid_id': hybrid_id, 'env_id': env_id,
                    'snp': hybrid_to_genotype[hybrid_id],
                    'env': env_features[env_id],
                    'trait': float(row[col_name])
                })

    num_snps = next(iter(hybrid_to_genotype.values())).shape[0]
    num_env_vars = next(iter(env_features.values())).shape[0]
    return train_data, test_data, num_snps, num_env_vars


def train_with_test(train_data, test_data, num_snps, num_env_vars, trait_name, device):
    """直接复制 train.py 中 train_fold 的训练逻辑，验证集替换为测试集"""
    model = GeneEnvAttentionModelWithMoE(num_snps, num_env_vars, num_traits=1).to(device)

    train_ds = GeneEnvDataset(train_data, is_train=True)
    test_ds = GeneEnvDataset(test_data, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_reduce_factor'],
        patience=config['lr_reduce_patience'], min_lr=config['min_lr']
    )
    early_stopper = EarlyStopping(
        patience=config['early_patience'], min_delta=config['early_min_delta'], verbose=True
    )

    best_test_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_clustering_entropy = 0.0
        all_train_t, all_train_p = [], []

        for batch in train_loader:
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)
            targets = batch['trait'].to(device)
            preds, aux_loss = model(snp, env, hard_clustering=False)

            main_loss = criterion(preds, targets.squeeze())
            clustering_info = model.get_clustering_info()
            clustering_loss = 0.0
            if clustering_info is not None:
                entropy = clustering_info.get('entropy', torch.tensor(0.0))
                diversity = clustering_info.get('diversity', torch.tensor(0.0))
                clustering_loss = (
                    config.get('clustering_entropy_weight', 0.05) * entropy +
                    config.get('clustering_diversity_weight', 0.01) * diversity
                )
                epoch_clustering_entropy += entropy.item() * targets.size(0)

            loss = main_loss + config['aux_loss_coef'] * aux_loss + clustering_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += main_loss.detach().item() * targets.size(0)
            epoch_aux_loss += aux_loss.detach().item() * targets.size(0)
            all_train_t.extend(targets.detach().cpu().numpy())
            all_train_p.extend(preds.detach().cpu().numpy())

        train_loss = epoch_train_loss / len(train_ds)
        avg_aux = epoch_aux_loss / len(train_ds)
        avg_entropy = epoch_clustering_entropy / len(train_ds)

        from sklearn.metrics import r2_score
        train_r2 = r2_score(all_train_t, all_train_p)

        model.eval()
        test_loss_total = 0.0
        all_test_t, all_test_p = [], []

        with torch.no_grad():
            for batch in test_loader:
                snp = batch['snp'].to(device)
                env = batch['env'].to(device)
                targets = batch['trait'].to(device)
                preds, _ = model(snp, env, hard_clustering=True)
                loss = criterion(preds, targets.squeeze())
                test_loss_total += loss.item() * targets.size(0)
                all_test_t.extend(targets.cpu().numpy())
                all_test_p.extend(preds.cpu().numpy())

        test_loss = test_loss_total / len(test_ds)
        test_r2 = r2_score(all_test_t, all_test_p)
        test_pcc = np.corrcoef(np.array(all_test_t).flatten(), np.array(all_test_p).flatten())[0, 1]

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_loss)

        if epoch % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Test R2: {test_r2:.4f} | Test PCC: {test_pcc:.4f} | LR: {current_lr:.2e}")

        if test_loss < best_test_loss - config['early_min_delta']:
            best_test_loss = test_loss
            best_epoch = epoch + 1

        if early_stopper.step(test_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    print(f"  Done. Best epoch: {best_epoch}, Best test loss: {best_test_loss:.6f}, "
          f"Time: {elapsed:.1f}s")

    return {
        'best_test_loss': float(best_test_loss),
        'best_epoch': best_epoch,
        'training_time_seconds': float(elapsed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['unseen_env', 'unseen_geno', 'unseen_both', 'all'])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    data_dir = os.path.join(BASE_DIR, 'data')
    results_dir = os.path.join(BASE_DIR, 'results', 'generalization')
    os.makedirs(results_dir, exist_ok=True)

    genotype_path = os.path.join(data_dir, 'genotype.tsv')
    env_path = os.path.join(data_dir, 'Environment_data.csv')

    experiments = list(EXP_DIR_MAP.keys()) if args.experiment == 'all' else [args.experiment]
    all_results = {}

    for exp_name in experiments:
        print(f"\n{'='*60}\n# {exp_name}\n{'='*60}")
        exp_results = []

        for split_dir_name in EXP_DIR_MAP[exp_name]:
            split_path = os.path.join(data_dir, split_dir_name)
            train_csv = os.path.join(split_path, 'train.csv')
            test_csv = os.path.join(split_path, 'test.csv')
            if not os.path.exists(train_csv) or not os.path.exists(test_csv):
                print(f"  SKIP {split_dir_name}: missing train.csv or test.csv")
                continue

            split_results = {}
            for trait_key in ['trait1', 'trait2', 'trait3', 'trait4', 'trait5', 'trait6']:
                print(f"\n  [{split_dir_name}] [{trait_key}] {TRAIT_COL_MAP[trait_key]}")
                train_data, test_data, ns, ne = load_experiment_data(
                    genotype_path, env_path, train_csv, test_csv, trait_key
                )
                print(f"  train={len(train_data)}, test={len(test_data)}, SNPs={ns}, Env={ne}")

                result = train_with_test(train_data, test_data, ns, ne, trait_key, device)
                split_results[trait_key] = result

            exp_results.append({'split': split_dir_name, 'results': split_results})

        all_results[exp_name] = exp_results
        out_path = os.path.join(results_dir, f'{exp_name}_results.json')
        with open(out_path, 'w') as f:
            json.dump(exp_results, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    summary = {}
    for exp_name, exp_results in all_results.items():
        trait_r2s = {}
        for split_result in exp_results:
            for trait_key, r in split_result['results'].items():
                if trait_key not in trait_r2s:
                    trait_r2s[trait_key] = []
                trait_r2s[trait_key].append(r['best_test_loss'])
        summary[exp_name] = {k: {'avg_loss': np.mean(v)} for k, v in trait_r2s.items()}

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary: {summary_path}')


if __name__ == '__main__':
    main()
