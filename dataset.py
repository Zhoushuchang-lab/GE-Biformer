# -*- coding: utf-8 -*-
"""
数据处理文件
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import config


class GeneEnvDataset(Dataset):
    """基因型-环境数据集"""
    def __init__(self, data_list, is_train=True):
        self.data = data_list
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 在所有模式下都返回hybrid_id和env_id
        result = {
            'snp': torch.tensor(item['snp'], dtype=torch.float32),
            'env': torch.tensor(item['env'], dtype=torch.float32),
            'hybrid_id': item['hybrid_id'],
            'env_id': item['env_id']
        }
        # 只有训练模式下才返回trait
        if self.is_train:
            result['trait'] = torch.tensor(item['trait'], dtype=torch.float32)
        return result


def preprocess_snp_matrix(snp_matrix):
    """预处理SNP矩阵"""
    mask = snp_matrix == -1
    snp_matrix[mask] = np.random.choice([0, 1, 2], size=np.sum(mask))
    return (snp_matrix - 1.0) / 1.0



def load_012_matrix(file_path):
    """加载SNP数据"""
    print(f"加载SNP数据: {file_path}")
    snp_data = pd.read_csv(file_path, sep='\t', dtype=str, low_memory=False)
    hybrid_ids = snp_data.columns[1:].tolist()
    snp_values = snp_data.iloc[:, 1:].values.astype(np.float32).T
    snp_processed = preprocess_snp_matrix(snp_values)
    scaler = RobustScaler()
    snp_scaled = scaler.fit_transform(snp_processed)
    hybrid_to_snp = {hybrid_id: snp_scaled[i] for i, hybrid_id in enumerate(hybrid_ids)}
    return hybrid_to_snp



def load_environment_data(file_path):
    """加载环境数据"""
    print(f"加载环境数据: {file_path}")
    env_data = pd.read_csv(file_path, sep=',', dtype=str)
    env_ids = env_data.columns[1:]
    env_values = env_data.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
    env_values = env_values.fillna(env_values.mean())
    env_values = env_values.values.T.astype(np.float32)
    scaler = StandardScaler()
    env_values_scaled = scaler.fit_transform(env_values)
    env_features = {env_id: env_values_scaled[i] for i, env_id in enumerate(env_ids)}
    return env_features



def prepare_dataset(snp_path, env_path, pheno_path):
    """准备训练数据集"""
    hybrid_to_snp = load_012_matrix(snp_path)
    env_features = load_environment_data(env_path)
    
    print(f"加载表型数据: {pheno_path}")
    pheno_data = pd.read_csv(pheno_path, sep=',', dtype=str)
    
    # 处理所有性状列
    trait_columns = ['Yield', 'Grain Moisture', 'Pollen_DAP_days', 'Silk_DAP_days', 
                    'Plant_Height_cm', 'Ear_Height_cm', 'Twt_kg_m3']
    
    # 将字符串转换为数值
    for col in trait_columns:
        if col in pheno_data.columns:
            pheno_data[col] = pd.to_numeric(pheno_data[col], errors='coerce')
    
    # 去重处理：每个(Hybrid, Environment)组合只保留第一个观测值
    print(f"原始表型数据行数: {len(pheno_data)}")
    pheno_data = pheno_data.drop_duplicates(subset=['Hybrid', 'Environment'], keep='first')
    print(f"去重后表型数据行数: {len(pheno_data)}")
    
    # 初始化性状数据列表
    trait_data = {
        'trait1': [], 'trait2': [], 'trait3': [], 'trait4': [], 
        'trait5': [], 'trait6': [], 'trait7': []
    }
    
    # 性状列映射
    trait_col_map = {
        'trait1': 'Yield',
        'trait2': 'Grain Moisture',
        'trait3': 'Pollen_DAP_days',
        'trait4': 'Silk_DAP_days',
        'trait5': 'Plant_Height_cm',
        'trait6': 'Ear_Height_cm',
        'trait7': 'Twt_kg_m3'
    }
    
    for _, row in pheno_data.iterrows():
        env_id = row['Environment']
        hybrid_id = row['Hybrid']
        
        if env_id in env_features and hybrid_id in hybrid_to_snp:
            snp_feature = hybrid_to_snp[hybrid_id]
            env_feature = env_features[env_id]
            
            base_feature = {
                'hybrid_id': hybrid_id,
                'env_id': env_id,
                'snp': snp_feature,
                'env': env_feature
            }
            
            # 处理每个性状
            for trait_key, col_name in trait_col_map.items():
                if col_name in pheno_data.columns and not pd.isna(row[col_name]):
                    trait_feature = base_feature.copy()
                    trait_feature['trait'] = float(row[col_name])
                    trait_data[trait_key].append(trait_feature)
    
    # 打印每个性状的有效数据量
    for trait_key, col_name in trait_col_map.items():
        print(f"{trait_key} ({col_name}) 有效数据: {len(trait_data[trait_key])}条")
    
    # 初始化结果字典
    result = {
        'trait1': {'train': [], 'val': []},
        'trait2': {'train': [], 'val': []},
        'trait3': {'train': [], 'val': []},
        'trait4': {'train': [], 'val': []},
        'trait5': {'train': [], 'val': []},
        'trait6': {'train': [], 'val': []},
        'trait7': {'train': [], 'val': []}
    }
    
    # 对每个性状进行train_test_split
    for trait_key in trait_data.keys():
        if len(trait_data[trait_key]) > 0:
            trait_train, trait_val = train_test_split(
                trait_data[trait_key],
                test_size=config['test_size'],
                random_state=config['random_state']
            )
            result[trait_key]['train'] = trait_train
            result[trait_key]['val'] = trait_val
    
    return result