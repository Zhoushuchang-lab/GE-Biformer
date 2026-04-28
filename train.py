# -*- coding: utf-8 -*-
"""
训练脚本入口文件
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# 导入模块化组件
from config import config
from dataset import GeneEnvDataset, prepare_dataset
from model import GeneEnvAttentionModelWithMoE, GeneEnvAttentionModelWithoutMoE, GeneEnvAttentionModelWithoutEffectSeparation, GeneEnvAttentionModelWithoutTokenFusion
from utils import EarlyStopping, save_model, plot_training_history

def train_model(model, train_loader, val_loader, device, trait_name):
    """训练模型"""
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
        'aux_loss': [], 'best_epoch': 0, 'best_val_loss': float('inf')
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        epoch_aux = 0.0
        all_targets = []
        all_preds = []
        
        for batch in train_loader:
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)
            targets = batch['trait'].to(device)
            
            preds, aux_loss = model(snp, env)
            main_loss = criterion(preds, targets.squeeze())
            loss = main_loss + config['aux_loss_coef'] * aux_loss
            
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
        train_r2 = r2_score(all_targets, all_preds)
        train_pcc = np.corrcoef(np.array(all_targets).flatten(), np.array(all_preds).flatten())[0, 1]
        
        # 验证阶段
        model.eval()
        val_loss_total = 0.0
        all_val_t = []
        all_val_p = []
        
        with torch.no_grad():
            for batch in val_loader:
                snp = batch['snp'].to(device)
                env = batch['env'].to(device)
                targets = batch['trait'].to(device)
                preds, _ = model(snp, env)
                loss = criterion(preds, targets.squeeze())
                val_loss_total += loss.item() * targets.size(0)
                all_val_t.extend(targets.detach().cpu().numpy())
                all_val_p.extend(preds.detach().cpu().numpy())
        
        val_loss = val_loss_total / len(val_loader.dataset)
        val_r2 = r2_score(all_val_t, all_val_p)
        val_pcc = np.corrcoef(np.array(all_val_t).flatten(), np.array(all_val_p).flatten())[0, 1]
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['train_pcc'].append(train_pcc)
        history['val_pcc'].append(val_pcc)
        history['aux_loss'].append(avg_aux)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # 维护最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        
        # 早停判断
        if early_stopper.step(val_loss, model):
            print(f"早停触发，停止训练")
            break
    
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    
    print(f"\n{trait_name}训练完成! 最佳epoch: {best_epoch}, 最佳验证损失: {best_val_loss:.6f}")
    return model, history

def record_moe_weights(model, data_loader, device, output_file):
    """记录MoE权重到CSV文件"""
    model.eval()
    moe_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)
            
            # 获取批次中的hybrid_id和env_id
            hybrid_ids = batch.get('hybrid_id', [])
            env_ids = batch.get('env_id', [])
            
            # 获取MoE详细信息
            preds, aux_loss, moe_details = model(snp, env, return_moe_details=True)
            
            # 解析MoE详细信息
            all_probabilities = moe_details['all_probabilities'].cpu().numpy()
            expert_indices = moe_details['expert_indices'].cpu().numpy()
            gating_weights = moe_details['gating_weights'].cpu().numpy()
            
            # 处理每个样本
            for i in range(snp.size(0)):
                sample_idx = batch_idx * data_loader.batch_size + i
                
                # 获取当前样本的hybrid_id和env_id
                hybrid_id = hybrid_ids[i] if hybrid_ids else f"sample_{sample_idx}"
                env_id = env_ids[i] if env_ids else f"env_{sample_idx}"
                
                sample_result = {
                    'sample_index': sample_idx,
                    'hybrid_id': hybrid_id,
                    'env_id': env_id,
                    'prediction': float(preds[i].cpu().numpy()),
                }
                
                # 添加每个专家的概率
                for expert_idx in range(all_probabilities.shape[1]):
                    sample_result[f'expert_{expert_idx}_prob'] = float(all_probabilities[i, expert_idx])
                
                # 添加被选中的专家索引和权重
                for top_k_idx in range(expert_indices.shape[1]):
                    sample_result[f'selected_expert_{top_k_idx}'] = int(expert_indices[i, top_k_idx])
                    sample_result[f'gating_weight_{top_k_idx}'] = float(gating_weights[i, top_k_idx])
                
                moe_results.append(sample_result)
    
    # 保存为CSV
    df = pd.DataFrame(moe_results)
    df.to_csv(output_file, index=False)
    print(f"MoE权重已保存到: {output_file}")
    return df

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练基因型-环境MoE注意力模型')
    parser.add_argument('--traits', type=str, default='3,4,5,6', 
                        help='要训练的性状编号，用逗号分隔，例如：1,2,3 或 3-7')
    parser.add_argument('--model', type=str, default='full',
                        choices=['full', 'no_moe', 'no_effect_sep', 'no_token_fusion'],
                        help='模型类型: full(完整模型), no_moe(无MoE), no_effect_sep(无Effect Separation), no_token_fusion(无Token Fusion)')
    args = parser.parse_args()
    
    # 解析性状参数
    traits_str = args.traits
    if '-' in traits_str:
        start, end = map(int, traits_str.split('-'))
        selected_traits = [f'trait{i}' for i in range(start, end + 1)]
    else:
        selected_traits = [f'trait{int(i)}' for i in traits_str.split(',')]
    
    print(f"\n选择的性状: {selected_traits}")
    print(f"模型类型: {args.model}")
    
    if torch.cuda.is_available():
        device = f"cuda:{config['cuda_device']}"
        print(f"使用GPU: {torch.cuda.get_device_name(config['cuda_device'])}")
    else:
        device = "cpu"
        print("使用CPU训练")
    
    # 设置数据和输出目录（使用相对路径，便于GitHub共享）
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "data")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "gene_env_moe_model")
    moe_analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "moe_analysis")
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(moe_analysis_dir, exist_ok=True)
    
    # 使用整合后的表型数据
    snp_path = os.path.join(data_dir, "012_matrix.tsv")
    pheno_path = os.path.join(data_dir, "Phenotypes.csv")
    env_path = os.path.join(data_dir, "Environment_data.csv")
    
    # 创建moe_analysis目录（如果不存在）
    os.makedirs(moe_analysis_dir, exist_ok=True)
    
    print("\n准备训练数据...")
    dataset_dict = prepare_dataset(snp_path, env_path, pheno_path)
    
    # 检查是否有可用的训练数据
    has_data = False
    sample_data = None
    for trait in selected_traits:
        if dataset_dict[trait]['train']:
            sample_data = dataset_dict[trait]['train'][0]
            has_data = True
            break
    
    if not has_data:
        raise ValueError("没有足够的训练数据")
    
    num_snps = sample_data['snp'].shape[0]
    num_env_vars = sample_data['env'].shape[0]
    print(f"SNP特征维度: {num_snps}, 环境特征维度: {num_env_vars}")
    
    # 模型名称映射
    model_name_map = {
        'full': 'Gene-Environment Interaction Attention Model with MoE',
        'no_moe': 'Gene-Environment Interaction Attention Model without MoE',
        'no_effect_sep': 'Gene-Environment Interaction Attention Model without Effect Separation',
        'no_token_fusion': 'Gene-Environment Interaction Attention Model without Token Fusion'
    }
    
    # 模型类映射
    model_class_map = {
        'full': GeneEnvAttentionModelWithMoE,
        'no_moe': GeneEnvAttentionModelWithoutMoE,
        'no_effect_sep': GeneEnvAttentionModelWithoutEffectSeparation,
        'no_token_fusion': GeneEnvAttentionModelWithoutTokenFusion
    }
    
    training_summary = {}
    
    # 性状名称映射
    trait_name_map = {
        'trait1': 'Yield',
        'trait2': 'Grain Moisture',
        'trait3': 'Pollen_DAP_days',
        'trait4': 'Silk_DAP_days',
        'trait5': 'Plant_Height_cm',
        'trait6': 'Ear_Height_cm',
        'trait7': 'Twt_kg_m3'
    }
    
    # 训练所选性状
    for trait in selected_traits:
        if not dataset_dict[trait]['train']:
            print(f"\n跳过{trait}: 没有训练数据")
            continue
        
        print(f"\n========== 训练{model_name_map[args.model]} - {trait_name_map[trait]} ==========")
        train_ds = GeneEnvDataset(dataset_dict[trait]['train'], is_train=True)
        val_ds = GeneEnvDataset(dataset_dict[trait]['val'], is_train=True)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
        
        # 创建模型
        model_class = model_class_map[args.model]
        model = model_class(num_snps, num_env_vars, num_traits=1)
        model, history = train_model(model, train_loader, val_loader, device, trait_name_map[trait])
        
        model_name = f"{trait}_{args.model}"
        model_path = save_model(model, model_name, model_dir)
        
        # 保存训练历史CSV
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
        })
        hist_csv_path = model_path.rsplit('.', 1)[0] + "_history.csv"
        hist_df.to_csv(hist_csv_path, index=False)
        
        best_idx = max(0, history['best_epoch'] - 1)
        key = f"{trait}_{args.model}"
        training_summary[key] = {
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'train_r2': history['train_r2'][best_idx],
            'val_r2': history['val_r2'][best_idx],
            'model_path': model_path,
        }
        
        summary_path = model_path.rsplit('.', 1)[0] + "_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(training_summary[key], f, ensure_ascii=False, indent=2)
        
        # 记录MoE权重（仅当模型有MoE层时）
        if args.model != 'no_moe':
            print(f"\n记录{trait_name_map[trait]}训练集MoE权重...")
            train_weights_file = os.path.join(moe_analysis_dir, f"{model_name}_train_moe_weights.csv")
            record_moe_weights(model, train_loader, device, train_weights_file)
            
            print(f"记录{trait_name_map[trait]}验证集MoE权重...")
            val_weights_file = os.path.join(moe_analysis_dir, f"{model_name}_val_moe_weights.csv")
            record_moe_weights(model, val_loader, device, val_weights_file)
    
    print("\n========== 训练摘要 ==========")
    for key, summary in training_summary.items():
        print(f"{key}:")
        for metric_key, value in summary.items():
            print(f"  {metric_key}: {value}")
    
    print(f"\n训练完成! 模型已保存到{model_dir}目录")

if __name__ == "__main__":
    main()