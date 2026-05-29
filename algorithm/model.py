# -*- coding: utf-8 -*-
"""
模型定义文件 - 仅支持可学习聚类的 Tokenization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config


# ==================== 可学习聚类 Tokenizer ====================

class LearnableClusterTokenizer(nn.Module):
    """
    可学习的聚类 Tokenizer
    """
    def __init__(self, input_dim: int, num_tokens: int, fusion_dim: int,
                 temperature: float = 0.5):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.temperature = temperature

        self.token_prototypes = nn.Parameter(
            torch.randn(num_tokens, input_dim) * 0.1
        )

        self.token_projector = nn.Sequential(
            nn.Linear(1, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        self.token_interaction = nn.MultiheadAttention(
            fusion_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.interaction_norm = nn.LayerNorm(fusion_dim)

        self.last_assignment = None
        self.last_hard_assignment = None

    def forward(self, features: torch.Tensor, hard: bool = False) -> torch.Tensor:
        B, D = features.shape
        assert D == self.input_dim, f"Expected input dim {self.input_dim}, got {D}"

        similarity = torch.einsum('bd,td->bdt', features, self.token_prototypes)

        raw_assignment = F.softmax(similarity / self.temperature, dim=-1)
        
        if hard:
            assignment = F.softmax(similarity / self.temperature, dim=-1)
            hard_assignment = torch.zeros_like(assignment)
            hard_assignment.scatter_(-1, assignment.argmax(dim=-1, keepdim=True), 1.0)
            assignment = hard_assignment
            self.last_hard_assignment = assignment
        elif self.training:
            assignment = F.gumbel_softmax(
                similarity, tau=self.temperature, hard=False, dim=-1
            )
        else:
            assignment = raw_assignment

        self.last_raw_assignment = raw_assignment.detach()
        self.last_assignment = assignment.detach()

        token_raw = torch.einsum('bd,bdt->bt', features, assignment).unsqueeze(-1)

        count_per_token = assignment.sum(dim=1, keepdim=True)
        count_per_token = count_per_token.transpose(1, 2)
        token_raw = token_raw / (count_per_token + 1e-8)

        tokens = self.token_projector(token_raw)

        tokens, attn_weights = self.token_interaction(tokens, tokens, tokens)
        tokens = self.interaction_norm(tokens)

        return tokens

    def get_clustering_entropy(self) -> torch.Tensor:
        if not hasattr(self, 'last_raw_assignment') or self.last_raw_assignment is None:
            return torch.tensor(0.0, device=self.token_prototypes.device)
        
        assignment = self.last_raw_assignment
        entropy_per_snp = -(assignment * torch.log(assignment + 1e-8)).sum(dim=-1)
        mean_entropy = entropy_per_snp.mean()
        
        max_entropy = np.log(assignment.shape[-1])
        normalized_entropy = mean_entropy / max_entropy
        
        return normalized_entropy

    def get_token_diversity(self) -> torch.Tensor:
        if not hasattr(self, 'last_raw_assignment') or self.last_raw_assignment is None:
            return torch.tensor(0.0, device=self.token_prototypes.device)
        
        assignment = self.last_raw_assignment
        token_usage = assignment.mean(dim=1).mean(dim=0)
        
        target = torch.ones_like(token_usage) / len(token_usage)
        
        diversity_loss = F.kl_div(
            (token_usage + 1e-8).log(), 
            target, 
            reduction='batchmean'
        )
        
        return diversity_loss


# ==================== Token 化跨模态融合模块 ====================

class TokenWiseCrossModalFusion(nn.Module):
    """
    Token化跨模态注意力融合（仅可学习聚类）
    """
    def __init__(self, snp_dim: int, env_dim: int, fusion_dim: int,
                 num_heads: int = 8, dropout: float = 0.3,
                 num_snp_tokens: int = 8, num_env_tokens: int = 4):
        super().__init__()

        self.num_snp_tokens = num_snp_tokens
        self.num_env_tokens = num_env_tokens
        self.fusion_dim = fusion_dim

        self.snp_tokenizer = LearnableClusterTokenizer(
            input_dim=snp_dim,
            num_tokens=num_snp_tokens,
            fusion_dim=fusion_dim,
            temperature=config['clustering_temperature']
        )

        self.env_tokenizer = LearnableClusterTokenizer(
            input_dim=env_dim,
            num_tokens=num_env_tokens,
            fusion_dim=fusion_dim,
            temperature=config['clustering_temperature']
        )

        self.snp_to_env = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.env_to_snp = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, snp_features: torch.Tensor, env_features: torch.Tensor,
                hard_clustering: bool = False) -> torch.Tensor:
        snp_tokens = self.snp_tokenizer(snp_features, hard=hard_clustering)
        env_tokens = self.env_tokenizer(env_features, hard=hard_clustering)

        snp_attn_out, _ = self.snp_to_env(
            query=snp_tokens, key=env_tokens, value=env_tokens
        )
        env_attn_out, _ = self.env_to_snp(
            query=env_tokens, key=snp_tokens, value=snp_tokens
        )

        snp_pooled = snp_attn_out.mean(dim=1)
        env_pooled = env_attn_out.mean(dim=1)

        fused = torch.cat([snp_pooled, env_pooled], dim=1)
        return self.out_proj(fused)

    def get_clustering_info(self):
        if hasattr(self.snp_tokenizer, 'last_assignment'):
            return {
                'assignment': self.snp_tokenizer.last_assignment,
                'hard_assignment': self.snp_tokenizer.last_hard_assignment,
                'prototypes': self.snp_tokenizer.token_prototypes,
                'entropy': self.snp_tokenizer.get_clustering_entropy(),
                'diversity': self.snp_tokenizer.get_token_diversity()
            }
        return None


# ==================== 注意力模块 ====================

class EfficientCooperativeAttention(nn.Module):
    """高效协同注意力"""
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


class GLUDynamicGate(nn.Module):
    """GLU动态门控"""
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


class SNPAttentionModule(nn.Module):
    """SNP注意力模块"""
    def __init__(self, num_snps):
        super().__init__()
        self.num_snps = num_snps

        layers = []
        prev_size = num_snps
        for h_size in config['weights_units']:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.LayerNorm(h_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_snps))
        layers.append(nn.Tanh())
        self.independent_attention = nn.Sequential(*layers)

        self.cooperative_attention = EfficientCooperativeAttention(
            num_features=num_snps,
            hidden_dim=config['snp_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        self.dynamic_gate = GLUDynamicGate(num_snps)

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
    """环境注意力模块"""
    def __init__(self, num_env_vars):
        super().__init__()
        self.num_env_vars = num_env_vars

        layers = []
        prev_size = num_env_vars
        for h_size in config['env_units']:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.LayerNorm(h_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, num_env_vars))
        layers.append(nn.Tanh())
        self.independent_attention = nn.Sequential(*layers)

        self.cooperative_attention = EfficientCooperativeAttention(
            num_features=num_env_vars,
            hidden_dim=config['env_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        self.dynamic_gate = GLUDynamicGate(num_env_vars)

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


# ==================== MoE 层 ====================

class Expert(nn.Module):
    """专家网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(input_dim, num_experts, k)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x, return_gating=False):
        batch_size = x.size(0)
        output_dim = self.experts[0].fc2.out_features

        gating_weights, expert_indices, mask, probs = self.gating_network(x)

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
                'gating_weights': gating_weights,
                'expert_indices': expert_indices,
                'all_probabilities': probs,
                'mask': mask,
                'load_distribution': mask.float().mean(dim=0),
                'importance_distribution': probs.mean(dim=0)
            }
        else:
            return expert_outputs, aux_loss

    def _calculate_aux_loss(self, probs, mask):
        importance = probs.sum(dim=0)
        importance_loss = (importance.std() / (importance.mean() + 1e-6)) ** 2
        load = mask.float().mean(dim=0)
        load_loss = (load.std() / (load.mean() + 1e-6)) ** 2
        return importance_loss + load_loss


# ==================== 主模型 ====================

class GeneEnvAttentionModelWithMoE(nn.Module):
    """
    基因型-环境 MoE 注意力模型（仅可学习聚类）
    """
    def __init__(self, num_snps, num_env_vars, num_traits=1):
        super().__init__()
        self.num_snps = num_snps
        self.num_env_vars = num_env_vars
        self.num_traits = num_traits

        self.snp_processor = SNPAttentionModule(num_snps)
        self.env_processor = EnvironmentAttentionModule(num_env_vars)

        self.cross_modal_fusion = TokenWiseCrossModalFusion(
            snp_dim=num_snps,
            env_dim=num_env_vars,
            fusion_dim=config['fusion_attention_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_snp_tokens=config['num_snp_tokens'],
            num_env_tokens=config['num_env_tokens']
        )

        self.moe_layer = MoELayer(
            input_dim=config['fusion_attention_dim'],
            hidden_dim=config['moe_hidden_dim'],
            output_dim=config['fusion_attention_dim'],
            num_experts=config['num_experts'],
            k=config['top_k'],
            dropout=config['expert_dropout']
        )

        layers = []
        input_dim = config['fusion_attention_dim']
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

    def forward(self, snp_features, env_features, return_moe_details=False, hard_clustering=False):
        snp_outputs = self.snp_processor(snp_features)
        snp_processed = snp_outputs['processed_features']

        env_outputs = self.env_processor(env_features)
        env_processed = env_outputs['processed_features']

        fused_features = self.cross_modal_fusion(
            snp_processed, env_processed, hard_clustering=hard_clustering
        )

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

    def get_clustering_info(self):
        return self.cross_modal_fusion.get_clustering_info()
