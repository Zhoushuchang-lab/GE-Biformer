# -*- coding: utf-8 -*-
"""
模型定义文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


# ==================== MoE 相关模块 ====================
class Expert(nn.Module):
    """专家网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.layer_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class GatingNetwork(nn.Module):
    """门控网络"""
    def __init__(self, input_dim, num_experts, k=2):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.k = k

    def forward(self, x):
        logits = self.fc(x)
        probabilities = self.softmax(logits)
        
        top_k_weights, top_k_indices = torch.topk(probabilities, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        mask = torch.zeros_like(probabilities).scatter_(-1, top_k_indices, 1.0)
        return top_k_weights, top_k_indices, mask, probabilities


class MoELayer(nn.Module):
    """MoE层"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, k=2, dropout=0.2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(input_dim, num_experts, k)
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, return_gating=False):
        """
        前向传播
        Args:
            x: 输入特征
            return_gating: 是否返回门控详细信息
        """
        batch_size = x.size(0)
        output_dim = self.experts[0].fc2.out_features
        
        # 获取门控网络的输出
        gating_weights, expert_indices, mask, probs = self.gating_network(x)
        
        # 稀疏专家计算
        expert_outputs = torch.zeros(batch_size, output_dim, 
                                   device=x.device, dtype=x.dtype)
        
        flat_expert_indices = expert_indices.view(-1)
        flat_gating_weights = gating_weights.view(-1, 1)
        flat_x = x.repeat_interleave(self.k, dim=0)
        
        for i, expert in enumerate(self.experts):
            idx = (flat_expert_indices == i)
            if idx.any():
                expert_input = flat_x[idx]
                expert_output = expert(expert_input)
                expert_output = expert_output * flat_gating_weights[idx]
                batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(self.k)[idx]
                expert_outputs.index_add_(0, batch_indices, expert_output)
        
        aux_loss = self._calculate_aux_loss(probs, mask)
        expert_outputs = self.output_norm(expert_outputs)
        
        if return_gating:
            return expert_outputs, aux_loss, {
                'gating_weights': gating_weights,      # top_k权重 (batch_size, k)
                'expert_indices': expert_indices,      # top_k专家索引 (batch_size, k)
                'all_probabilities': probs,            # 所有专家完整概率 (batch_size, num_experts)
                'mask': mask,                          # 选择掩码 (batch_size, num_experts)
                'load_distribution': mask.float().mean(dim=0),  # 负载分布
                'importance_distribution': probs.mean(dim=0)    # 重要性分布
            }
        else:
            return expert_outputs, aux_loss
        
    def _calculate_aux_loss(self, probs, mask):
        importance = probs.sum(dim=0)
        importance_loss = (importance.std() / (importance.mean() + 1e-6)) ** 2
        load = mask.float().mean(dim=0)
        load_loss = (load.std() / (load.mean() + 1e-6)) ** 2
        return importance_loss + load_loss


# ==================== 注意力模块 ====================
class GLUDynamicGate(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.SiLU()
        )
        self.value_proj = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ind_effect, co_effect):
        gate = self.sigmoid(self.gate_proj(co_effect))
        value = self.value_proj(ind_effect)
        return gate * value + (1 - gate) * co_effect, gate


class EfficientCooperativeAttention(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim

        self.query = nn.Linear(num_features, hidden_dim)
        self.key = nn.Linear(num_features, hidden_dim)
        self.value = nn.Linear(num_features, hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_features)
        self.norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.norm(x)

        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.proj(attn_output).squeeze(1)
        output = torch.clamp(output, min=-10, max=10)
        return self.dropout(output) + x, attn_weights.detach()


class SNPAttentionModule(nn.Module):
    def __init__(self, num_snps):
        super().__init__()
        self.num_snps = num_snps
        self.independent_attention = self._build_attention_module(num_snps, config['weights_units'])
        self.cooperative_attention = EfficientCooperativeAttention(
            num_features=num_snps,
            hidden_dim=config['snp_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        self.dynamic_gate = GLUDynamicGate(num_snps)

    def _build_attention_module(self, num_features, units):
        layers = []
        prev_size = num_features
        for h_size in units:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.LayerNorm(h_size))
            if len(units) > 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config['dropout']))
                prev_size = h_size
        layers.append(nn.Linear(prev_size, num_features))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        ind_weights = self.independent_attention(x)
        ind_effect = x * ind_weights

        co_effect, co_weights = self.cooperative_attention(x)
        if len(co_effect.shape) == 3:
            co_effect = co_effect.squeeze(1)

        weighted, gate_weights = self.dynamic_gate(ind_effect, co_effect)
        weighted += x

        return {
            'processed_features': weighted,
            'attention_weights': co_weights,
            'gate_weights': gate_weights,
            'independent_weights': ind_weights
        }


class EnvironmentAttentionModule(nn.Module):
    def __init__(self, num_env_vars):
        super().__init__()
        self.num_env_vars = num_env_vars
        self.independent_attention = self._build_attention_module(num_env_vars, config['env_units'])
        self.cooperative_attention = EfficientCooperativeAttention(
            num_features=num_env_vars,
            hidden_dim=config['env_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        self.dynamic_gate = GLUDynamicGate(num_env_vars)

    def _build_attention_module(self, num_features, units):
        layers = []
        prev_size = num_features
        for h_size in units:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.LayerNorm(h_size))
            if len(units) > 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config['dropout']))
                prev_size = h_size
        layers.append(nn.Linear(prev_size, num_features))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        ind_weights = self.independent_attention(x)
        ind_effect = x * ind_weights

        co_effect, co_weights = self.cooperative_attention(x)
        if len(co_effect.shape) == 3:
            co_effect = co_effect.squeeze(1)

        weighted, gate_weights = self.dynamic_gate(ind_effect, co_effect)
        weighted += x

        return {
            'processed_features': weighted,
            'attention_weights': co_weights,
            'gate_weights': gate_weights,
            'independent_weights': ind_weights
        }


# ==================== Token化跨模态融合模块 ====================
class TokenWiseCrossModalFusion(nn.Module):
    """Token化跨模态注意力融合"""
    def __init__(self, snp_dim: int, env_dim: int, fusion_dim: int,
                 num_heads: int = 8, dropout: float = 0.3,
                 num_snp_tokens: int = 8, num_env_tokens: int = 4):
        super().__init__()

        assert snp_dim >= num_snp_tokens and env_dim >= num_env_tokens
        self.num_snp_tokens = num_snp_tokens
        self.num_env_tokens = num_env_tokens
        self.fusion_dim = fusion_dim

        self.snp_segment_proj = nn.Linear(snp_dim // num_snp_tokens, fusion_dim)
        self.env_segment_proj = nn.Linear(env_dim // num_env_tokens, fusion_dim)

        self.snp_to_env = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
        self.env_to_snp = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def _segment_tokens(self, x: torch.Tensor, num_tokens: int, proj: nn.Linear) -> torch.Tensor:
        B, D = x.shape
        seg = D // num_tokens
        x = x[:, :seg * num_tokens]
        x = x.view(B, num_tokens, seg)
        tokens = proj(x)
        return tokens

    def forward(self, snp_features: torch.Tensor, env_features: torch.Tensor) -> torch.Tensor:
        snp_tokens = self._segment_tokens(snp_features, self.num_snp_tokens, self.snp_segment_proj)
        env_tokens = self._segment_tokens(env_features, self.num_env_tokens, self.env_segment_proj)

        snp_attn, _ = self.snp_to_env(query=snp_tokens, key=env_tokens, value=env_tokens)
        env_attn, _ = self.env_to_snp(query=env_tokens, key=snp_tokens, value=snp_tokens)

        snp_pooled = snp_attn.mean(dim=1)
        env_pooled = env_attn.mean(dim=1)

        fused = torch.cat([snp_pooled, env_pooled], dim=1)
        return self.out_proj(fused)


# ==================== 主模型 ====================
class GeneEnvAttentionModelWithMoE(nn.Module):
    """基因型-环境 MoE 注意力模型"""
    def __init__(self, num_snps, num_env_vars, num_traits=1):
        super().__init__()
        self.num_snps = num_snps
        self.num_env_vars = num_env_vars
        self.num_traits = num_traits
        
        self.snp_processor = SNPAttentionModule(num_snps)
        self.env_processor = EnvironmentAttentionModule(num_env_vars)
        
        fusion_dim = config['fusion_attention_dim']
        self.cross_modal_fusion = TokenWiseCrossModalFusion(
            snp_dim=num_snps,
            env_dim=num_env_vars,
            fusion_dim=fusion_dim,
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_snp_tokens=config['num_snp_tokens'],
            num_env_tokens=config['num_env_tokens'],
        )
        
        self.moe_layer = MoELayer(
            input_dim=fusion_dim,
            hidden_dim=config['moe_hidden_dim'],
            output_dim=fusion_dim,
            num_experts=config['num_experts'],
            k=config['top_k'],
            dropout=config['expert_dropout']
        )
        
        layers = []
        input_dim = fusion_dim
        for i in range(len(config['fusion_units'])):
            layers.append(nn.Linear(input_dim, config['fusion_units'][i]))
            layers.append(nn.LayerNorm(config['fusion_units'][i]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))
            input_dim = config['fusion_units'][i]
        
        self.feature_network = nn.Sequential(*layers)
        self.predictor = nn.Linear(input_dim, num_traits)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, snp_features, env_features, return_moe_details=False):
        """
        前向传播
        Args:
            snp_features: SNP特征
            env_features: 环境特征
            return_moe_details: 是否返回MoE详细信息
        """
        snp_outputs = self.snp_processor(snp_features)
        snp_processed = snp_outputs['processed_features']
        
        env_outputs = self.env_processor(env_features)
        env_processed = env_outputs['processed_features']
        
        fused_features = self.cross_modal_fusion(snp_processed, env_processed)
        
        if return_moe_details:
            moe_features, aux_loss, moe_details = self.moe_layer(
                fused_features, return_gating=True
            )
        else:
            moe_features, aux_loss = self.moe_layer(fused_features)
            moe_details = None
        
        if len(self.feature_network) > 0:
            integrated_features = self.feature_network(moe_features)
        else:
            integrated_features = moe_features
        
        pred = self.predictor(integrated_features)
        if self.num_traits == 1:
            pred = pred.squeeze(-1)
        
        if return_moe_details:
            return pred, aux_loss, moe_details
        else:
            return pred, aux_loss


class GeneEnvAttentionModelWithoutEffectSeparation(nn.Module):
    """基因型-环境注意力模型（无Effect Separation）"""
    def __init__(self, num_snps, num_env_vars, num_traits=1):
        super().__init__()
        self.num_snps = num_snps
        self.num_env_vars = num_env_vars
        self.num_traits = num_traits
        
        self.snp_linear = nn.Sequential(
            nn.Linear(num_snps, config['weights_units'][-1]),
            nn.LayerNorm(config['weights_units'][-1]),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['weights_units'][-1], num_snps),
            nn.Tanh()
        )
        
        self.env_linear = nn.Sequential(
            nn.Linear(num_env_vars, config['env_units'][-1]),
            nn.LayerNorm(config['env_units'][-1]),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['env_units'][-1], num_env_vars),
            nn.Tanh()
        )
        
        # 添加投影层，确保embed_dim能被num_heads整除
        self.snp_proj = nn.Linear(num_snps, config['snp_attention_dim'])
        self.env_proj = nn.Linear(num_env_vars, config['env_attention_dim'])
        
        self.snp_attention = nn.MultiheadAttention(
            embed_dim=config['snp_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.snp_norm = nn.LayerNorm(config['snp_attention_dim'])
        
        self.env_attention = nn.MultiheadAttention(
            embed_dim=config['env_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.env_norm = nn.LayerNorm(config['env_attention_dim'])
        
        fusion_dim = config['fusion_attention_dim']
        self.snp_fc = nn.Linear(config['snp_attention_dim'], fusion_dim)
        self.env_fc = nn.Linear(config['env_attention_dim'], fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim * 2)
        self.fusion_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        
        self.moe_layer = MoELayer(
            input_dim=fusion_dim,
            hidden_dim=config['moe_hidden_dim'],
            output_dim=fusion_dim,
            num_experts=config['num_experts'],
            k=config['top_k'],
            dropout=config['expert_dropout']
        )
        
        layers = []
        input_dim = fusion_dim
        for i in range(len(config['fusion_units'])):
            layers.append(nn.Linear(input_dim, config['fusion_units'][i]))
            layers.append(nn.LayerNorm(config['fusion_units'][i]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))
            input_dim = config['fusion_units'][i]
        
        self.feature_network = nn.Sequential(*layers)
        self.predictor = nn.Linear(input_dim, num_traits)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, snp_features, env_features, return_moe_details=False):
        snp_weighted = self.snp_linear(snp_features)
        snp_effect = snp_features * snp_weighted
        
        env_weighted = self.env_linear(env_features)
        env_effect = env_features * env_weighted
        
        # 投影到适合注意力机制的维度
        snp_projected = self.snp_proj(snp_effect)
        env_projected = self.env_proj(env_effect)
        
        # 使用投影后的特征进行注意力计算
        snp_attn_out, _ = self.snp_attention(
            snp_projected.unsqueeze(1), snp_projected.unsqueeze(1), snp_projected.unsqueeze(1)
        )
        snp_fused = self.snp_norm(snp_projected + snp_attn_out.squeeze(1))
        
        env_attn_out, _ = self.env_attention(
            env_projected.unsqueeze(1), env_projected.unsqueeze(1), env_projected.unsqueeze(1)
        )
        env_fused = self.env_norm(env_projected + env_attn_out.squeeze(1))
        
        snp_proj = self.snp_fc(snp_fused)
        env_proj = self.env_fc(env_fused)
        
        fused = torch.cat([snp_proj, env_proj], dim=1)
        fused = self.fusion_norm(fused)
        fused_features = self.fusion_fc(fused)
        fused_features = F.gelu(fused_features)
        
        if return_moe_details:
            moe_features, aux_loss, moe_details = self.moe_layer(
                fused_features, return_gating=True
            )
        else:
            moe_features, aux_loss = self.moe_layer(fused_features)
            moe_details = None
        
        if len(self.feature_network) > 0:
            integrated_features = self.feature_network(moe_features)
        else:
            integrated_features = moe_features
        
        pred = self.predictor(integrated_features)
        if self.num_traits == 1:
            pred = pred.squeeze(-1)
        
        if return_moe_details:
            return pred, aux_loss, moe_details
        else:
            return pred, aux_loss


class GeneEnvAttentionModelWithoutTokenFusion(nn.Module):
    """基因型-环境注意力模型（无Token Fusion）"""
    def __init__(self, num_snps, num_env_vars, num_traits=1):
        super().__init__()
        self.num_snps = num_snps
        self.num_env_vars = num_env_vars
        self.num_traits = num_traits
        
        self.snp_processor = SNPAttentionModule(num_snps)
        self.env_processor = EnvironmentAttentionModule(num_env_vars)
        
        fusion_dim = config['fusion_attention_dim']
        self.snp_proj = nn.Linear(num_snps, fusion_dim)
        self.env_proj = nn.Linear(num_env_vars, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim * 2)
        self.fusion_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        
        self.moe_layer = MoELayer(
            input_dim=fusion_dim,
            hidden_dim=config['moe_hidden_dim'],
            output_dim=fusion_dim,
            num_experts=config['num_experts'],
            k=config['top_k'],
            dropout=config['expert_dropout']
        )
        
        layers = []
        input_dim = fusion_dim
        for i in range(len(config['fusion_units'])):
            layers.append(nn.Linear(input_dim, config['fusion_units'][i]))
            layers.append(nn.LayerNorm(config['fusion_units'][i]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))
            input_dim = config['fusion_units'][i]
        
        self.feature_network = nn.Sequential(*layers)
        self.predictor = nn.Linear(input_dim, num_traits)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, snp_features, env_features, return_moe_details=False):
        snp_outputs = self.snp_processor(snp_features)
        snp_processed = snp_outputs['processed_features']
        
        env_outputs = self.env_processor(env_features)
        env_processed = env_outputs['processed_features']
        
        snp_proj = self.snp_proj(snp_processed)
        env_proj = self.env_proj(env_processed)
        
        fused = torch.cat([snp_proj, env_proj], dim=1)
        fused = self.fusion_norm(fused)
        fused_features = self.fusion_fc(fused)
        fused_features = F.gelu(fused_features)
        
        if return_moe_details:
            moe_features, aux_loss, moe_details = self.moe_layer(
                fused_features, return_gating=True
            )
        else:
            moe_features, aux_loss = self.moe_layer(fused_features)
            moe_details = None
        
        if len(self.feature_network) > 0:
            integrated_features = self.feature_network(moe_features)
        else:
            integrated_features = moe_features
        
        pred = self.predictor(integrated_features)
        if self.num_traits == 1:
            pred = pred.squeeze(-1)
        
        if return_moe_details:
            return pred, aux_loss, moe_details
        else:
            return pred, aux_loss


class GeneEnvAttentionModelWithoutMoE(nn.Module):
    """基因型-环境注意力模型（无MoE）"""
    def __init__(self, num_snps, num_env_vars, num_traits=1):
        super().__init__()
        self.num_snps = num_snps
        self.num_env_vars = num_env_vars
        self.num_traits = num_traits
        
        self.snp_processor = SNPAttentionModule(num_snps)
        self.env_processor = EnvironmentAttentionModule(num_env_vars)
        
        fusion_dim = config['fusion_attention_dim']
        self.cross_modal_fusion = TokenWiseCrossModalFusion(
            snp_dim=num_snps,
            env_dim=num_env_vars,
            fusion_dim=fusion_dim,
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_snp_tokens=config['num_snp_tokens'],
            num_env_tokens=config['num_env_tokens'],
        )
        
        layers = []
        input_dim = fusion_dim
        for i in range(len(config['fusion_units'])):
            layers.append(nn.Linear(input_dim, config['fusion_units'][i]))
            layers.append(nn.LayerNorm(config['fusion_units'][i]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))
            input_dim = config['fusion_units'][i]
        
        self.feature_network = nn.Sequential(*layers)
        self.predictor = nn.Linear(input_dim, num_traits)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, snp_features, env_features, return_moe_details=False):
        snp_outputs = self.snp_processor(snp_features)
        snp_processed = snp_outputs['processed_features']
        
        env_outputs = self.env_processor(env_features)
        env_processed = env_outputs['processed_features']
        
        fused_features = self.cross_modal_fusion(snp_processed, env_processed)
        
        aux_loss = torch.tensor(0.0, device=snp_features.device)
        moe_details = None
        
        if len(self.feature_network) > 0:
            integrated_features = self.feature_network(fused_features)
        else:
            integrated_features = fused_features
        
        pred = self.predictor(integrated_features)
        if self.num_traits == 1:
            pred = pred.squeeze(-1)
        
        return pred, aux_loss