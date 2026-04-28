# -*- coding: utf-8 -*-
"""
配置文件
"""

# ==================== 配置参数 ====================
config = {
    'weights_units': [512, 256, 128],
    'env_units': [128, 64],
    'fusion_units': [512, 256, 128],
    'num_heads': 8,
    'dropout': 0.3,
    'test_size': 0.2,
    'random_state': 42,
    'batch_size': 64,
    'learning_rate': 0.0003,
    'epochs': 100,
    'lr_reduce_factor': 0.5,
    'lr_reduce_patience': 20,
    'min_lr': 1e-6,
    'cuda_device': 0,
    'snp_attention_dim': 64,
    'env_attention_dim': 64,
    'fusion_attention_dim': 256,
    'weight_decay': 1e-5,
    'num_experts': 8,
    'moe_hidden_dim': 128,
    'top_k': 2,
    'aux_loss_coef': 0.01,
    'expert_dropout': 0.2,
    'num_snp_tokens': 8,
    'num_env_tokens': 4,
    'early_patience': 30,
    'early_min_delta': 1e-4,
}