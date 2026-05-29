# -*- coding: utf-8 -*-
"""
Gene-Environment MoE Attention Model Training Script
Learnable Clustering + 5-fold Cross Validation
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from config import config
from dataset import GeneEnvDataset, prepare_dataset
from model import GeneEnvAttentionModelWithMoE
from utils import EarlyStopping


def train_fold(model, train_loader, val_loader, device, trait_name, fold):
    """Train a single fold"""
    # Move model to device (CRITICAL FIX)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_reduce_factor'],
        patience=config['lr_reduce_patience'],
        min_lr=config['min_lr']
    )

    criterion = nn.HuberLoss()
    early_stopper = EarlyStopping(
        patience=config['early_patience'],
        min_delta=config['early_min_delta'],
        mode='min',
        verbose=True
    )

    history = {
        'train_loss': [], 'val_loss': [], 'learning_rates': [],
        'train_r2': [], 'val_r2': [], 'train_pcc': [], 'val_pcc': [],
        'aux_loss': [], 'clustering_entropy': [],
        'best_epoch': 0, 'best_val_loss': float('inf')
    }

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_aux = 0.0
        epoch_clustering_entropy = 0.0
        all_targets = []
        all_preds = []

        for batch in train_loader:
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)
            targets = batch['trait'].to(device)

            # Forward pass - returns (pred, aux_loss) by default
            preds, aux_loss = model(snp, env, hard_clustering=False)
            main_loss = criterion(preds, targets.squeeze())

            # Clustering auxiliary loss
            clustering_info = model.get_clustering_info()
            clustering_loss = 0.0
            if clustering_info is not None:
                entropy = clustering_info.get('entropy', torch.tensor(0.0))
                diversity = clustering_info.get('diversity', torch.tensor(0.0))
                
                entropy_loss = entropy
                diversity_loss = diversity
                
                clustering_loss = (
                    config.get('clustering_entropy_weight', 0.05) * entropy_loss +
                    config.get('clustering_diversity_weight', 0.01) * diversity_loss
                )
                epoch_clustering_entropy += entropy.item() * targets.size(0)

            loss = main_loss + config['aux_loss_coef'] * aux_loss + clustering_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += main_loss.detach().item() * targets.size(0)
            epoch_aux += aux_loss.detach().item() * targets.size(0)
            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

        train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_aux = epoch_aux / len(train_loader.dataset)
        avg_clustering_entropy = epoch_clustering_entropy / len(train_loader.dataset)
        train_r2 = r2_score(all_targets, all_preds)
        train_pcc = np.corrcoef(np.array(all_targets).flatten(), np.array(all_preds).flatten())[0, 1]

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        all_val_t = []
        all_val_p = []

        with torch.no_grad():
            for batch in val_loader:
                snp = batch['snp'].to(device)
                env = batch['env'].to(device)
                targets = batch['trait'].to(device)
                preds, _ = model(snp, env, hard_clustering=True)
                loss = criterion(preds, targets.squeeze())
                val_loss_total += loss.item() * targets.size(0)
                all_val_t.extend(targets.detach().cpu().numpy())
                all_val_p.extend(preds.detach().cpu().numpy())

        val_loss = val_loss_total / len(val_loader.dataset)
        val_r2 = r2_score(all_val_t, all_val_p)
        val_pcc = np.corrcoef(np.array(all_val_t).flatten(), np.array(all_val_p).flatten())[0, 1]

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['train_pcc'].append(train_pcc)
        history['val_pcc'].append(val_pcc)
        history['aux_loss'].append(avg_aux)
        history['clustering_entropy'].append(avg_clustering_entropy)

        # Print progress
        print(f"Fold {fold} Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f} | "
              f"Entropy: {avg_clustering_entropy:.4f} | LR: {current_lr:.2e}")

        # Update best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

        # Early stopping
        if early_stopper.step(val_loss, model):
            print(f"Fold {fold} Early stopping triggered, stopping training")
            break

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss

    print(f"\n{trait_name} Fold {fold} training completed! Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")
    return model, history


def get_fold_predictions(model, data_loader, device):
    """Get predictions for a fold"""
    model.eval()
    predictions = []
    targets = []
    hybrid_ids = []
    env_ids = []

    with torch.no_grad():
        for batch in data_loader:
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)
            trait = batch['trait']

            preds, _ = model(snp, env, hard_clustering=True)

            predictions.extend(preds.cpu().numpy().flatten())
            targets.extend(trait.numpy().flatten())
            hybrid_ids.extend(batch['hybrid_id'])
            env_ids.extend(batch['env_id'])

    return {
        'prediction': predictions,
        'target': targets,
        'hybrid_id': hybrid_ids,
        'env_id': env_ids
    }


def save_token_assignments(model, data_loader, device, save_path):
    """保存Token的SNP分配权重"""
    model.eval()
    all_assignments = []

    with torch.no_grad():
        for batch in data_loader:
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)
            _, _ = model(snp, env, hard_clustering=False)

            clustering_info = model.get_clustering_info()
            if clustering_info is not None and 'assignment' in clustering_info:
                assignment = clustering_info['assignment'].cpu().numpy()
                all_assignments.append(assignment)

    if all_assignments:
        avg_assignment = np.concatenate(all_assignments, axis=0).mean(axis=0)
        np.save(save_path, avg_assignment)
        print(f"Saved token assignments: {save_path}, shape: {avg_assignment.shape}")
        return avg_assignment
    return None


def train_with_5fold_cv(model_class, data_list, num_snps, num_env_vars, device,
                        trait_name, model_name, cv_results_dir, predictions_dir, history_dir):
    """Perform 5-fold cross validation training"""
    kfold = KFold(n_splits=5, shuffle=True, random_state=config['random_state'])

    fold_results = []
    all_fold_predictions = []
    fold_timing_info = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_list), 1):
        print(f"\n{'='*60}")
        print(f"{trait_name} - Fold {fold}/5")
        print(f"{'='*60}")

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        fold_start_time = time.time()

        # Create fresh model for each fold
        model = model_class(num_snps, num_env_vars, num_traits=1)

        # Create data loaders
        train_ds = GeneEnvDataset(train_data, is_train=True)
        val_ds = GeneEnvDataset(val_data, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

        # Train this fold
        model, history = train_fold(
            model, train_loader, val_loader, device, trait_name, fold
        )

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_timing_info.append({
            'fold': fold,
            'start_time': fold_start_time,
            'end_time': fold_end_time,
            'duration_seconds': fold_duration,
            'duration_formatted': f"{int(fold_duration // 60)}m {int(fold_duration % 60)}s"
        })

        # Get predictions
        val_predictions = get_fold_predictions(model, val_loader, device)
        val_predictions['fold'] = fold
        val_predictions['set_type'] = 'validation'
        all_fold_predictions.append(val_predictions)

        train_predictions = get_fold_predictions(model, train_loader, device)
        train_predictions['fold'] = fold
        train_predictions['set_type'] = 'train'
        all_fold_predictions.append(train_predictions)

        # Record fold results
        best_idx = max(0, history['best_epoch'] - 1)
        fold_result = {
            'fold': fold,
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'best_train_loss': history['train_loss'][best_idx],
            'best_val_r2': history['val_r2'][best_idx],
            'best_train_r2': history['train_r2'][best_idx],
            'best_val_pcc': history['val_pcc'][best_idx],
            'best_train_pcc': history['train_pcc'][best_idx],
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'duration_seconds': fold_duration,
            'duration_formatted': f"{int(fold_duration // 60)}m {int(fold_duration % 60)}s"
        }
        fold_results.append(fold_result)

        # Save training history
        hist_df = pd.DataFrame({
            'epoch': list(range(1, len(history['train_loss']) + 1)),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'learning_rate': history['learning_rates'],
            'train_r2': history['train_r2'],
            'val_r2': history['val_r2'],
            'train_pcc': history['train_pcc'],
            'val_pcc': history['val_pcc'],
            'aux_loss': history['aux_loss'],
            'clustering_entropy': history['clustering_entropy']
        })
        hist_csv_path = os.path.join(history_dir, f"{model_name}_fold{fold}_history.csv")
        hist_df.to_csv(hist_csv_path, index=False)

        # Save model checkpoint
        model_path = os.path.join(cv_results_dir, f"{model_name}_fold{fold}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

        # Save token assignments for biology verification
        token_assignments_path = os.path.join(cv_results_dir, f"{model_name}_fold{fold}_token_assignments.npy")
        save_token_assignments(model, train_loader, device, token_assignments_path)

    # Save all predictions
    all_predictions_df = pd.DataFrame(all_fold_predictions)
    predictions_output_path = os.path.join(predictions_dir, f"{model_name}_all_fold_predictions.csv")
    all_predictions_df.to_csv(predictions_output_path, index=False)
    print(f"\nAll fold predictions saved: {predictions_output_path}")

    # Save timing info
    timing_df = pd.DataFrame(fold_timing_info)
    timing_output_path = os.path.join(cv_results_dir, f"{model_name}_fold_timing.csv")
    timing_df.to_csv(timing_output_path, index=False)
    print(f"Fold timing saved: {timing_output_path}")

    # Create summary
    summary = {
        'trait': trait_name,
        'model': model_name,
        'num_folds': 5,
        'fold_results': fold_results,
        'average_val_loss': np.mean([r['best_val_loss'] for r in fold_results]),
        'average_val_r2': np.mean([r['best_val_r2'] for r in fold_results]),
        'average_val_pcc': np.mean([r['best_val_pcc'] for r in fold_results]),
        'average_train_r2': np.mean([r['best_train_r2'] for r in fold_results]),
        'average_train_pcc': np.mean([r['best_train_pcc'] for r in fold_results]),
        'std_val_r2': np.std([r['best_val_r2'] for r in fold_results]),
        'std_val_pcc': np.std([r['best_val_pcc'] for r in fold_results]),
        'total_duration_seconds': sum([r['duration_seconds'] for r in fold_results]),
        'fold_timing': fold_timing_info
    }

    # Save summary
    summary_path = os.path.join(cv_results_dir, f"{model_name}_5fold_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Training summary saved: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Gene-Environment MoE Attention Model (Learnable Clustering + 5-fold CV)')
    parser.add_argument('--traits', type=str, default='1-6',
                        help='Traits to train, comma separated, e.g.: 1,2,3 or 1-6 (default)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default 5)')
    args = parser.parse_args()

    # Parse trait list
    traits_str = args.traits
    if '-' in traits_str:
        start, end = map(int, traits_str.split('-'))
        selected_traits = [f'trait{i}' for i in range(start, end + 1)]
    else:
        selected_traits = [f'trait{int(i)}' for i in traits_str.split(',')]

    print(f"\n{'#'*70}")
    print(f"# Selected traits: {selected_traits}")
    print(f"# Tokenization strategy: Learnable clustering")
    print(f"# Number of folds: {args.n_folds}")
    print(f"{'#'*70}")

    # Check device
    if torch.cuda.is_available():
        device = f"cuda:{config['cuda_device']}"
        print(f"Using GPU: {torch.cuda.get_device_name(config['cuda_device'])}")
    else:
        device = "cpu"
        print("Using CPU")

    # Set directories - adapt to server structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # fsas/
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    cv_results_dir = os.path.join(results_dir, "cv_results")
    predictions_dir = os.path.join(results_dir, "predictions")
    history_dir = os.path.join(results_dir, "history")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cv_results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    # Data paths
    genotype_path = os.path.join(data_dir, "genotype.tsv")
    pheno_path = os.path.join(data_dir, "Phenotypes.csv")
    env_path = os.path.join(data_dir, "Environment_data.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Load dataset
    print("\nPreparing training data...")
    dataset_dict = prepare_dataset(genotype_path, env_path, pheno_path, test_path)

    # Check data availability
    has_data = False
    sample_data = None
    for trait in selected_traits:
        if dataset_dict[trait]['train']:
            sample_data = dataset_dict[trait]['train'][0]
            has_data = True
            break

    if not has_data:
        raise ValueError("Insufficient training data")

    num_snps = sample_data['snp'].shape[0]
    num_env_vars = sample_data['env'].shape[0]
    print(f"SNP feature dim: {num_snps}, Env feature dim: {num_env_vars}")

    # Trait name mapping
    trait_name_map = {
        'trait1': 'Yield',
        'trait2': 'Grain Moisture',
        'trait3': 'Pollen_DAP_days',
        'trait4': 'Silk_DAP_days',
        'trait5': 'Plant_Height_cm',
        'trait6': 'Ear_Height_cm',
        'trait7': 'Twt_kg_m3'
    }

    # Training loop
    all_summaries = {}
    total_start_time = time.time()

    for trait in selected_traits:
        if not dataset_dict[trait]['train']:
            print(f"\nSkipping {trait}: no training data")
            continue

        print(f"\n{'#'*70}")
        print(f"# Starting training - {trait_name_map[trait]}")
        print(f"{'#'*70}")

        # Combine train and val for cross validation
        train_val_data = dataset_dict[trait]['train'] + dataset_dict[trait]['val']
        
        # Validate data size
        if len(train_val_data) < args.n_folds:
            print(f"Warning: {trait_name_map[trait]} data size ({len(train_val_data)}) less than {args.n_folds} folds, skipping")
            continue

        model_name = f"{trait}_full_cluster"

        # Run 5-fold CV
        summary = train_with_5fold_cv(
            model_class=GeneEnvAttentionModelWithMoE,
            data_list=train_val_data,
            num_snps=num_snps,
            num_env_vars=num_env_vars,
            device=device,
            trait_name=trait_name_map[trait],
            model_name=model_name,
            cv_results_dir=cv_results_dir,
            predictions_dir=predictions_dir,
            history_dir=history_dir
        )

        all_summaries[trait] = summary

        # Print fold results summary
        print(f"\n{'='*60}")
        print(f"{trait_name_map[trait]} 5-fold CV results summary:")
        print(f"{'='*60}")
        for fold_result in summary['fold_results']:
            print(f"  Fold {fold_result['fold']}: Val Loss={fold_result['best_val_loss']:.4f}, "
                  f"Val R2={fold_result['best_val_r2']:.4f}, Val PCC={fold_result['best_val_pcc']:.4f}, "
                  f"Time={fold_result['duration_formatted']}")
        print(f"\n  Avg Val Loss: {summary['average_val_loss']:.4f}")
        print(f"  Avg Val R2: {summary['average_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
        print(f"  Avg Val PCC: {summary['average_val_pcc']:.4f} ± {summary['std_val_pcc']:.4f}")
        print(f"  Total training time: {int(summary['total_duration_seconds'] // 60)}m {int(summary['total_duration_seconds'] % 60)}s")

    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\n{'#'*70}")
    print(f"# All training completed!")
    print(f"{'#'*70}")
    print(f"Total time: {int(total_duration // 60)}m {int(total_duration % 60)}s")
    print(f"CV results saved to: {cv_results_dir}")
    print(f"CV results saved to: {cv_results_dir}")

    # Save overall summary
    overall_summary = {
        'all_traits_summary': all_summaries,
        'total_duration_seconds': total_duration,
        'total_duration_formatted': f"{int(total_duration // 60)}m {int(total_duration % 60)}s",
        'num_traits_trained': len(all_summaries),
        'model_type': 'full',
        'tokenization': 'learnable_clustering',
        'n_folds': args.n_folds
    }

    overall_summary_path = os.path.join(cv_results_dir, f"overall_5fold_summary_full_cluster.json")
    with open(overall_summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"\nOverall summary saved: {overall_summary_path}")


if __name__ == "__main__":
    main()