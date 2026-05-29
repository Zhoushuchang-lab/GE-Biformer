"""
外部数据集泛化能力验证脚本
使用GEBiformer算法对外部数据集进行训练和评估
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO_DIR = os.path.join(BASE_DIR, 'algorithm')
sys.path.insert(0, ALGO_DIR)

from model import GeneEnvAttentionModelWithMoE
from config import config

CONVERTED_DIR = os.path.join(BASE_DIR, 'data', 'other_data', 'converted')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'external_validation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'PyTorch version: {torch.__version__}')


class ExternalGeneEnvDataset(Dataset):
    def __init__(self, data_list, is_train=True):
        self.data = data_list
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        result = {
            'snp': torch.tensor(item['snp'], dtype=torch.float32),
            'env': torch.tensor(item['env'], dtype=torch.float32),
            'hybrid_id': item['hybrid_id'],
            'env_id': item['env_id']
        }
        if self.is_train:
            result['trait'] = torch.tensor(item['trait'], dtype=torch.float32)
        return result


def load_external_data():
    from sklearn.preprocessing import RobustScaler, StandardScaler

    print('Loading external data...')

    # Load genotype
    genotype_path = os.path.join(CONVERTED_DIR, 'genotype.tsv')
    print(f'  Genotype: {genotype_path}')
    genotype_data = pd.read_csv(genotype_path, sep='\t', dtype=str, low_memory=False)
    hybrid_ids = genotype_data.columns[1:].tolist()
    genotype_values = genotype_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32).T
    scaler_g = RobustScaler()
    genotype_scaled = scaler_g.fit_transform(genotype_values)
    hybrid_to_genotype = {hid: genotype_scaled[i] for i, hid in enumerate(hybrid_ids)}
    print(f'  {len(hybrid_ids)} genotypes, {hybrid_to_genotype[hybrid_ids[0]].shape[0]} markers')

    # Load environment
    env_path = os.path.join(CONVERTED_DIR, 'Environment_data.csv')
    print(f'  Environment: {env_path}')
    env_data = pd.read_csv(env_path, sep=',', dtype=str)
    env_ids = env_data.columns[1:]
    env_values = env_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32).T
    scaler_e = StandardScaler()
    env_scaled = scaler_e.fit_transform(env_values)
    env_features = {eid: env_scaled[i] for i, eid in enumerate(env_ids)}
    print(f'  {len(env_ids)} environments, {env_features[env_ids[0]].shape[0]} env vars')

    # Load phenotype
    pheno_path = os.path.join(CONVERTED_DIR, 'Phenotypes.csv')
    print(f'  Phenotype: {pheno_path}')
    pheno_data = pd.read_csv(pheno_path, sep=',', dtype=str)
    pheno_data['BLUEs_wtn'] = pd.to_numeric(pheno_data['BLUEs_wtn'], errors='coerce')
    pheno_data = pheno_data.dropna(subset=['BLUEs_wtn'])
    print(f'  {len(pheno_data)} valid phenotype records')

    # Build dataset
    samples = []
    skipped = 0
    for _, row in pheno_data.iterrows():
        env_id = row['Environment']
        hybrid_id = row['Hybrid']
        if env_id in env_features and hybrid_id in hybrid_to_genotype:
            samples.append({
                'hybrid_id': hybrid_id,
                'env_id': env_id,
                'snp': hybrid_to_genotype[hybrid_id],
                'env': env_features[env_id],
                'trait': float(row['BLUEs_wtn'])
            })
        else:
            skipped += 1

    print(f'  {len(samples)} samples built ({skipped} skipped)')

    num_snps = hybrid_to_genotype[hybrid_ids[0]].shape[0]
    num_env_vars = env_features[env_ids[0]].shape[0]
    return samples, num_snps, num_env_vars


def run_validation():
    print('=' * 60)
    print('外部数据集泛化能力验证')
    print('=' * 60)

    samples, num_snps, num_env_vars = load_external_data()
    print(f'\nSNP features: {num_snps}')
    print(f'Env features: {num_env_vars}')
    print(f'Total samples: {len(samples)}')

    target_values = [s['trait'] for s in samples]
    print(f'Trait stats: mean={np.mean(target_values):.2f}, std={np.std(target_values):.2f}, '
          f'min={np.min(target_values):.2f}, max={np.max(target_values):.2f}')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples)):
        fold += 1
        print(f'\n{"=" * 40}')
        print(f'Fold {fold}/5')
        print(f'{"=" * 40}')

        train_data = [samples[i] for i in train_idx]
        val_data = [samples[i] for i in val_idx]
        print(f'Train: {len(train_data)}, Val: {len(val_data)}')

        train_ds = ExternalGeneEnvDataset(train_data, is_train=True)
        val_ds = ExternalGeneEnvDataset(val_data, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

        model = GeneEnvAttentionModelWithMoE(num_snps, num_env_vars, num_traits=1)
        model = model.to(DEVICE)
        print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

        criterion = torch.nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                       weight_decay=config.get('weight_decay', 1e-5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config['lr_reduce_factor'],
            patience=config['lr_reduce_patience'], min_lr=config['min_lr'])

        best_val_loss = float('inf')
        best_train_r2 = 0
        best_val_r2 = 0
        patience_counter = 0
        fold_start_time = time.time()

        for epoch in range(config['epochs']):
            model.train()
            epoch_train_loss = 0.0
            all_train_t = []
            all_train_p = []

            for batch in train_loader:
                snp = batch['snp'].to(DEVICE)
                env = batch['env'].to(DEVICE)
                targets = batch['trait'].to(DEVICE)

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

                loss = main_loss + config['aux_loss_coef'] * aux_loss + clustering_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += main_loss.detach().item() * targets.size(0)
                all_train_t.extend(targets.detach().cpu().numpy())
                all_train_p.extend(preds.detach().cpu().numpy())

            train_loss = epoch_train_loss / len(train_loader.dataset)
            train_r2 = r2_score(all_train_t, all_train_p)

            model.eval()
            val_loss_total = 0.0
            all_val_t = []
            all_val_p = []
            with torch.no_grad():
                for batch in val_loader:
                    snp = batch['snp'].to(DEVICE)
                    env = batch['env'].to(DEVICE)
                    targets = batch['trait'].to(DEVICE)
                    preds, _ = model(snp, env, hard_clustering=True)
                    l = criterion(preds, targets.squeeze())
                    val_loss_total += l.item() * targets.size(0)
                    all_val_t.extend(targets.cpu().numpy())
                    all_val_p.extend(preds.cpu().numpy())

            val_loss = val_loss_total / len(val_loader.dataset)
            val_r2 = r2_score(all_val_t, all_val_p)
            val_pcc = np.corrcoef(np.array(all_val_t).flatten(), np.array(all_val_p).flatten())[0, 1]
            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:3d} | Val R2: {val_r2:.4f} | Val PCC: {val_pcc:.4f} | Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_train_r2 = train_r2
                best_val_r2 = val_r2
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.get('early_patience', 30):
                print(f'  Early stopping at epoch {epoch+1}')
                break

        fold_time = time.time() - fold_start_time
        result = {
            'fold': fold,
            'best_val_loss': float(best_val_loss),
            'best_train_r2': float(best_train_r2),
            'best_val_r2': float(best_val_r2),
            'time_seconds': fold_time
        }
        fold_results.append(result)
        print(f'  Fold {fold} done | Best Val R2: {best_val_r2:.4f} | Time: {fold_time:.1f}s')

    # Summary
    val_r2s = [r['best_val_r2'] for r in fold_results]
    times = [r['time_seconds'] for r in fold_results]
    summary = {
        'dataset': 'external_data',
        'n_samples': len(samples),
        'n_snps': num_snps,
        'n_env_vars': num_env_vars,
        'n_envs': len(set(s['env_id'] for s in samples)),
        'n_genotypes': len(set(s['hybrid_id'] for s in samples)),
        'fold_results': fold_results,
        'mean_val_r2': float(np.mean(val_r2s)),
        'std_val_r2': float(np.std(val_r2s)),
        'mean_time_seconds': float(np.mean(times)),
        'total_time_seconds': float(np.sum(times))
    }

    output_path = os.path.join(OUTPUT_DIR, 'external_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 60)
    print('外部验证结果汇总')
    print('=' * 60)
    print(f'Mean Val R2: {summary["mean_val_r2"]:.4f} +/- {summary["std_val_r2"]:.4f}')
    print(f'Per-fold Val R2: {[f"{r["best_val_r2"]:.4f}" for r in fold_results]}')
    print(f'Total time: {summary["total_time_seconds"]:.0f}s')
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    run_validation()
