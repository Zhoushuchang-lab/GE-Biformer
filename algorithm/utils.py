# -*- coding: utf-8 -*-
"""
工具函数文件
"""

import os
import copy
import json
from typing import Optional
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config import config


class EarlyStopping:
    """早停器"""
    def __init__(self, patience: int = 30, min_delta: float = 1e-4, 
                 mode: str = 'min', verbose: bool = True):
        assert mode in ('min', 'max')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best_score: Optional[float] = None
        self.best_state_dict: Optional[dict] = None
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def step(self, current_score: float, model: Optional[nn.Module] = None) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            if model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"早停: 初始化最佳分数 {self.best_score:.6f}")
            return False

        if self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.num_bad_epochs = 0
            if model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"早停: 指标改进为 {self.best_score:.6f}")
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"早停: 未改进 ({self.num_bad_epochs}/{self.patience})")
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
                if model is not None and self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                    if self.verbose:
                        print("早停: 已恢复至最佳权重")
        return self.should_stop



def save_model(model, trait_name, model_dir):
    """保存模型"""
    current_date = datetime.now().strftime("%m_%d_%H%M")
    model_path = os.path.join(model_dir, f"{trait_name}_moe_model_{current_date}.pt")
    
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'trait_name': trait_name,
        'train_date': current_date,
        'num_snps': model.num_snps,
        'num_env_vars': model.num_env_vars
    }, model_path)
    
    log_file = os.path.join(model_dir, f"{trait_name}_model_{current_date}.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"模型信息:\n")
        f.write(f"特征维度: SNPs={model.num_snps}, 环境变量={model.num_env_vars}\n")
        f.write(f"表型: {trait_name}\n")
        f.write(f"训练日期: {current_date}\n\n")
        f.write(f"配置参数:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"{trait_name}模型已保存: {model_path}")
    return model_path



def plot_training_history(history, title, save_path):
    """绘制训练历史"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(history['train_r2'], label='Train R²')
        plt.plot(history['val_r2'], label='Val R²')
        plt.title('R² Score')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(history['aux_loss'])
        plt.title('MoE Auxiliary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Aux Loss')
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"训练历史图已保存: {save_path}")
    except Exception as e:
        print(f"绘图失败: {e}")



def read_test_csv(file_path):
    """读取测试集，自动识别分隔符"""
    df = pd.read_csv(file_path, sep=None, engine='python', dtype=str)
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    
    # 列名清洗
    aliases = {
        'environment': 'Environment', 'env': 'Environment',
        'hybrid': 'Hybrid', 'hybrid_id': 'Hybrid'
    }
    for c in list(df.columns):
        k = c.strip().lower()
        if k in aliases and aliases[k] not in df.columns:
            df.rename(columns={c: aliases[k]}, inplace=True)
    
    required = {'Environment', 'Hybrid'}
    if not required.issubset(df.columns):
        raise ValueError(f"测试文件缺少必要列 {required}，实际列: {list(df.columns)}")
    
    return df[['Environment', 'Hybrid']].copy()



def find_latest_model(model_dir, trait):
    """查找最新的模型文件"""
    import glob
    pattern = os.path.join(model_dir, f"{trait}_moe_model_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]



def generate_predictions(model, test_loader, device):
    """生成预测结果"""
    model = model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            snp_features = batch['snp'].to(device)
            env_features = batch['env'].to(device)
            hybrid_ids = batch['hybrid_id']
            env_ids = batch['env_id']
            
            preds, _ = model(snp_features, env_features)
            
            for i in range(len(hybrid_ids)):
                predictions.append({
                    'Environment': env_ids[i],
                    'Hybrid': hybrid_ids[i],
                    'value': preds[i].item()
                })
    
    return predictions


def read_tsv_file(file_path, sep='\t', header=None):
    """读取TSV文件
    
    Args:
        file_path: 文件路径
        sep: 分隔符
        header: 是否有表头
    
    Returns:
        数据和列名
    """
    df = pd.read_csv(file_path, sep=sep, header=header)
    # 提取列名（如果有）
    if header is not None:
        columns = df.columns.tolist()
        data = df.values
    else:
        columns = None
        data = df.values
    return data, columns

def save_tsv_file(data, file_path, columns=None, sep='\t'):
    """保存TSV文件
    
    Args:
        data: 数据
        file_path: 文件路径
        columns: 列名
        sep: 分隔符
    """
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, sep=sep, index=False)
    print(f"数据已保存到: {file_path}")

def process_missing_values(data, missing_value=-1, max_missing_rate=0.2, fill_values=[0, 1, 2]):
    """处理缺失值
    
    Args:
        data: 输入数据，numpy数组
        missing_value: 缺失值标记
        max_missing_rate: 最大缺失率阈值，超过此值的样本将被删除
        fill_values: 用于填充缺失值的候选值
    
    Returns:
        处理后的数据，numpy数组
    """
    # 计算每行的缺失率
    missing_mask = data == missing_value
    missing_rate = np.mean(missing_mask, axis=1)
    
    # 过滤掉缺失率过高的样本
    valid_mask = missing_rate <= max_missing_rate
    filtered_data = data[valid_mask].copy()
    
    # 对剩余样本进行随机填充
    filtered_missing_mask = filtered_data == missing_value
    if np.any(filtered_missing_mask):
        # 计算需要填充的位置数量
        num_missing = np.sum(filtered_missing_mask)
        # 生成随机填充值
        fill_values_array = np.random.choice(fill_values, size=num_missing)
        # 填充缺失值
        filtered_data[filtered_missing_mask] = fill_values_array
    
    print(f"原始数据形状: {data.shape}")
    print(f"过滤后数据形状: {filtered_data.shape}")
    print(f"删除的样本数: {data.shape[0] - filtered_data.shape[0]}")
    print(f"填充的缺失值数量: {np.sum(filtered_missing_mask)}")
    
    return filtered_data

def process_missing_values_from_file(input_file, output_file, missing_value=-1, max_missing_rate=0.2, fill_values=[0, 1, 2]):
    """从文件处理缺失值
    
    Args:
        input_file: 输入TSV文件路径
        output_file: 输出TSV文件路径
        missing_value: 缺失值标记
        max_missing_rate: 最大缺失率阈值
        fill_values: 用于填充的候选值
    """
    print(f"读取文件: {input_file}")
    data, columns = read_tsv_file(input_file)
    
    # 处理缺失值
    processed_data = process_missing_values(data, missing_value, max_missing_rate, fill_values)
    
    # 保存处理后的数据
    save_tsv_file(processed_data, output_file, columns)
    print(f"缺失值处理完成，结果保存到: {output_file}")