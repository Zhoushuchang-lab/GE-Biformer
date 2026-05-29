# -*- coding: utf-8 -*-
"""
Token生物学功能验证脚本（使用真实染色体位置）
验证Token是否将功能相关的SNP组织到一起

使用方法:
    python analysis_token_biology.py --trait trait1 --fold 1 --top_k 100 --n_permutations 1000
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithm'))
from config import config


def load_snp_positions(snp_file_path):
    """
    从SNP ID解析真实的染色体位置
    
    SNP ID格式: S1_1007742 (染色体_物理位置)
    数据已按染色体+物理位置排序
    
    注意：genotype.tsv 第一列是SNP ID，从第二行开始（第一行是表头）
    """
    # 读取第一列（SNP ID），跳过表头
    df = pd.read_csv(snp_file_path, sep='\t', usecols=[0], dtype=str)
    snp_ids = df.iloc[:, 0].tolist()  # 第一列，所有行
    
    print(f"Loaded {len(snp_ids)} SNP IDs")
    print(f"First 5 SNPs: {snp_ids[:5]}")
    
    positions = []
    parse_fail_count = 0
    
    for snp_id in snp_ids:
        if pd.isna(snp_id):
            positions.append((1, len(positions)))
            parse_fail_count += 1
            continue
            
        if str(snp_id).startswith('S') and '_' in str(snp_id):
            try:
                parts = str(snp_id).split('_')
                chr_num = int(parts[0][1:])  # S1 -> 1, S10 -> 10
                pos = int(parts[1])
                positions.append((chr_num, pos))
            except (ValueError, IndexError):
                positions.append((1, len(positions)))
                parse_fail_count += 1
        else:
            positions.append((1, len(positions)))
            parse_fail_count += 1
    
    if parse_fail_count > 0:
        print(f"Warning: {parse_fail_count} SNPs failed to parse, using index as position")
    
    # 统计染色体分布
    chrs = [p[0] for p in positions]
    print(f"Chromosome range: {min(chrs)} - {max(chrs)}")
    
    return positions, snp_ids


def load_model(model_path, num_snps, num_env_vars, device):
    """加载训练好的模型"""
    from model import GeneEnvAttentionModelWithMoE

    model = GeneEnvAttentionModelWithMoE(
        num_snps=num_snps,
        num_env_vars=num_env_vars,
        num_traits=1
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def extract_token_snp_weights(model, data_loader, device):
    """
    从训练好的模型中提取每个Token对每个SNP的分配权重

    返回:
        assignment_matrix: [num_snps, num_tokens] 平均分配权重
    """
    all_assignments = []

    with torch.no_grad():
        for batch in data_loader:
            snp = batch['snp'].to(device)
            env = batch['env'].to(device)

            _, _ = model(snp, env, hard_clustering=False)

            clustering_info = model.get_clustering_info()
            if clustering_info is not None and 'assignment' in clustering_info:
                assignment = clustering_info['assignment']
                all_assignments.append(assignment.cpu().numpy())

    if all_assignments:
        avg_assignment = np.concatenate(all_assignments, axis=0).mean(axis=0)
        return avg_assignment
    else:
        return None


def calculate_clustering_score(positions, snp_indices):
    """
    计算一组SNP在染色体上的聚集程度（使用真实物理位置）

    指标: 平均相邻距离的倒数（越聚集，得分越高）

    参数:
        positions: 所有SNP的位置列表 [(chr, pos), ...]
        snp_indices: 要分析的SNP索引列表

    返回:
        score: 聚集得分（越高表示越聚集）
        chr_distribution: 染色体分布
    """
    selected_positions = [positions[i] for i in snp_indices]

    chr_groups = {}
    for idx, (chr_num, pos) in enumerate(selected_positions):
        if chr_num not in chr_groups:
            chr_groups[chr_num] = []
        chr_groups[chr_num].append((idx, pos))

    total_score = 0
    for chr_num, positions_list in chr_groups.items():
        if len(positions_list) < 2:
            continue

        positions_list.sort(key=lambda x: x[1])
        positions_only = [p for _, p in positions_list]

        distances = np.diff(positions_only)

        if len(distances) > 0:
            avg_distance = np.mean(distances)
            if avg_distance > 0:
                # 使用真实物理距离（bp），归一化到Mb使分数在合理范围
                score = 1.0 / (avg_distance / 1e6 + 1)
                total_score += score * len(positions_list)

    if len(selected_positions) > 0:
        total_score = total_score / len(selected_positions)

    return total_score, chr_groups


def permutation_test(model_assignments, snp_positions, num_tokens,
                     top_k=100, n_permutations=1000):
    """
    排列检验：验证Token内SNP的聚集是否显著高于随机

    返回:
        p_values: 每个Token的p值
        observed_scores: 观察到的聚集得分
        random_scores_distribution: 随机得分的分布
    """
    num_snps = len(snp_positions)
    observed_scores = []
    random_scores_distribution = []
    p_values = []

    for token_id in range(num_tokens):
        weights = model_assignments[:, token_id]
        top_indices = np.argsort(weights)[-top_k:]

        obs_score, _ = calculate_clustering_score(snp_positions, top_indices)
        observed_scores.append(obs_score)

        random_scores = []
        for _ in range(n_permutations):
            random_indices = np.random.choice(num_snps, size=top_k, replace=False)
            rand_score, _ = calculate_clustering_score(snp_positions, random_indices)
            random_scores.append(rand_score)

        random_scores_distribution.append(random_scores)

        p_value = np.mean(np.array(random_scores) >= obs_score)
        p_values.append(p_value)

        print(f"Token {token_id+1}: Observed Score = {obs_score:.4f}, p = {p_value:.4f}")

    return p_values, observed_scores, random_scores_distribution


def plot_genome_token_heatmap(snp_positions, model_assignments, num_tokens, 
                               save_path=None):
    """
    绘制整个基因组的Token分配热图
    X轴：按染色体位置排列的SNP
    Y轴：8个Token
    颜色：分配权重
    """
    # 准备数据：按染色体位置排序
    snp_indices = list(range(len(snp_positions)))
    snp_indices.sort(key=lambda i: (snp_positions[i][0], snp_positions[i][1]))
    
    # 重新排列assignment矩阵
    sorted_assignment = model_assignments[snp_indices]
    
    # 获取每条染色体的边界
    chr_boundaries = []
    current_chr = snp_positions[snp_indices[0]][0]
    for i, idx in enumerate(snp_indices):
        chr_num = snp_positions[idx][0]
        if chr_num != current_chr:
            chr_boundaries.append(i)
            current_chr = chr_num
    chr_boundaries.append(len(snp_indices))
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # 转置：Y轴=Token，X轴=SNP
    im = ax.imshow(sorted_assignment.T, aspect='auto', cmap='Blues', 
                   interpolation='none', vmin=0, vmax=1)
    
    # 标记染色体边界
    for boundary in chr_boundaries[:-1]:
        ax.axvline(x=boundary, color='red', linestyle='-', linewidth=1, alpha=0.7)
    
    # 添加染色体标签
    chr_starts = [0] + chr_boundaries[:-1]
    chr_centers = [(chr_starts[i] + chr_boundaries[i]) / 2 for i in range(len(chr_boundaries))]
    ax.set_xticks(chr_centers)
    ax.set_xticklabels([f'Chr{i+1}' for i in range(len(chr_boundaries))])
    
    ax.set_xlabel('SNP (ordered by chromosome position)', fontsize=12)
    ax.set_ylabel('Token ID', fontsize=12)
    ax.set_title('SNP-to-Token Assignment Weights Across Genome', fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Assignment Weight', fontsize=10)
    
    # 设置Y轴刻度
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels([f'Token {i+1}' for i in range(num_tokens)])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved genome heatmap: {save_path}")
    plt.close()
    
    return chr_boundaries


def plot_token_chromosome_distribution(snp_positions, model_assignments, num_tokens,
                                        top_k=100, save_path=None):
    """
    绘制每个Token高权重SNP在染色体上的分布
    """
    fig, axes = plt.subplots(num_tokens, 1, figsize=(16, 2.5*num_tokens))
    if num_tokens == 1:
        axes = [axes]
    
    # 获取所有SNP的染色体和位置
    all_chrs = [p[0] for p in snp_positions]
    all_positions = [p[1] for p in snp_positions]
    
    for token_id in range(num_tokens):
        ax = axes[token_id]
        
        weights = model_assignments[:, token_id]
        top_indices = np.argsort(weights)[-top_k:]
        top_weights = weights[top_indices]
        
        top_chrs = [snp_positions[i][0] for i in top_indices]
        top_positions = [snp_positions[i][1] for i in top_indices]
        
        # 绘制所有SNP（浅灰色背景）
        ax.scatter(all_positions, all_chrs, alpha=0.05, s=1, c='lightgray')
        
        # 绘制高权重SNP
        scatter = ax.scatter(top_positions, top_chrs, c=top_weights, s=20,
                              cmap='Reds', alpha=0.8, vmin=0, vmax=1)
        
        ax.set_ylabel(f'Token {token_id+1}\n(Chromosome)')
        ax.set_xlabel('Physical Position (bp)')
        ax.set_title(f'Token {token_id+1}: Top {len(top_indices)} SNPs by Assignment Weight')
        ax.set_yticks(range(1, 11))
        ax.set_ylim(0.5, 10.5)
        ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes, label='Assignment Weight', shrink=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved token distribution: {save_path}")
    plt.close()


def plot_token_aggregation(snp_positions, model_assignments, num_tokens,
                            top_k=100, save_path=None):
    """
    绘制每个Token的SNP基因组分布图（使用真实位置）
    """
    fig, axes = plt.subplots(num_tokens, 1, figsize=(16, 2.5*num_tokens))
    if num_tokens == 1:
        axes = [axes]

    all_positions_mb = [p[1] / 1e6 for p in snp_positions]
    all_chrs = [p[0] for p in snp_positions]

    for token_id in range(num_tokens):
        ax = axes[token_id]

        weights = model_assignments[:, token_id]
        top_indices = np.argsort(weights)[-top_k:]
        top_weights = weights[top_indices]
        
        top_chrs = [snp_positions[i][0] for i in top_indices]
        top_positions_mb = [snp_positions[i][1] / 1e6 for i in top_indices]

        # 绘制所有SNP
        ax.scatter(all_positions_mb, all_chrs, alpha=0.1, s=1, c='gray', label='All SNPs')
        
        # 绘制高权重SNP
        scatter = ax.scatter(top_positions_mb, top_chrs, c=top_weights, s=15,
                              cmap='Reds', alpha=0.8, vmin=0, vmax=1, 
                              label=f'Top {top_k} SNPs')

        ax.set_ylabel(f'Token {token_id+1}\n(Chromosome)')
        ax.set_xlabel('Physical Position (Mb)')
        ax.set_title(f'Token {token_id+1}: {len(top_indices)} high-weight SNPs')
        ax.set_yticks(range(1, 11))
        ax.set_ylim(0.5, 10.5)
        ax.legend(loc='upper right', fontsize=8)

    plt.colorbar(scatter, ax=axes, label='Assignment Weight', shrink=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_permutation_results(p_values, observed_scores, random_distribution,
                             num_tokens, save_path=None):
    """
    绘制排列检验结果
    """
    n_cols = min(4, num_tokens)
    n_rows = (num_tokens + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten()

    for token_id in range(num_tokens):
        ax = axes[token_id]

        random_scores = random_distribution[token_id]
        ax.hist(random_scores, bins=30, alpha=0.7, color='gray',
                label='Random distribution')

        obs = observed_scores[token_id]
        p = p_values[token_id]
        ax.axvline(obs, color='red', linewidth=2, label=f'Observed (p={p:.3f})')

        ax.set_xlabel('Clustering Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Token {token_id+1}')
        ax.legend(fontsize=8)

    for i in range(num_tokens, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Permutation Test: SNP Clustering Significance', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_pvalue_distribution(p_values, save_path=None):
    """绘制p值分布图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    tokens = range(1, len(p_values) + 1)
    colors = ['red' if p < 0.05 else 'blue' for p in p_values]

    ax.bar(tokens, p_values, color=colors, alpha=0.7)
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=2, label='p=0.05')
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, label='p=0.01')

    ax.set_xlabel('Token ID')
    ax.set_ylabel('p-value')
    ax.set_title('P-value Distribution for Token SNP Clustering')
    ax.legend()

    significant_005 = sum(1 for p in p_values if p < 0.05)
    significant_001 = sum(1 for p in p_values if p < 0.01)
    ax.text(0.02, 0.98, f'Significant (p<0.05): {significant_005}/{len(p_values)}\n'
                          f'Highly Significant (p<0.01): {significant_001}/{len(p_values)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_token_assignment_barplot(model_assignments, num_tokens, top_n=20, save_path=None):
    """
    绘制每个Token权重最高的Top-N SNP
    """
    fig, axes = plt.subplots(num_tokens, 1, figsize=(12, 2*num_tokens))
    if num_tokens == 1:
        axes = [axes]

    for token_id in range(num_tokens):
        ax = axes[token_id]
        weights = model_assignments[:, token_id]
        top_indices = np.argsort(weights)[-top_n:]
        top_weights = weights[top_indices]
        
        ax.barh(range(top_n), top_weights, color='steelblue')
        ax.set_xlabel('Assignment Weight')
        ax.set_ylabel('SNP Rank')
        ax.set_title(f'Token {token_id+1}: Top {top_n} SNPs')
        ax.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Token Biology Function Verification')
    parser.add_argument('--trait', type=str, default='trait1',
                        help='Trait to analyze, e.g., trait1')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold to analyze (1-5)')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top SNPs per token')
    parser.add_argument('--n_permutations', type=int, default=1000,
                        help='Number of permutations for test')
    parser.add_argument('--data_dir', type=str,
                        default='../data',
                        help='Data directory')
    parser.add_argument('--results_dir', type=str,
                        default='../results',
                        help='Results directory')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Token Biology Function Verification (with Real Chromosome Positions)")
    print(f"{'='*60}")
    print(f"Trait: {args.trait}")
    print(f"Fold: {args.fold}")
    print(f"Top-K SNPs per Token: {args.top_k}")
    print(f"Permutations: {args.n_permutations}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir
    results_dir = args.results_dir

    genotype_path = os.path.join(data_dir, "genotype.tsv")
    model_path = os.path.join(results_dir, "cv_results",
                               f"{args.trait}_full_cluster_fold{args.fold}.pt")

    print(f"\nLoading SNP positions from: {genotype_path}")
    snp_positions, snp_ids = load_snp_positions(genotype_path)
    print(f"Loaded {len(snp_positions)} SNPs")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    from dataset import GeneEnvDataset, prepare_dataset
    from torch.utils.data import DataLoader

    env_path = os.path.join(data_dir, "Environment_data.csv")
    pheno_path = os.path.join(data_dir, "Phenotypes.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print("\nPreparing dataset...")
    dataset_dict = prepare_dataset(genotype_path, env_path, pheno_path, test_path)

    train_val_data = dataset_dict[args.trait]['train'] + dataset_dict[args.trait]['val']
    if not train_val_data:
        print(f"ERROR: No training data found for {args.trait}")
        return

    num_snps = train_val_data[0]['snp'].shape[0]
    num_env_vars = train_val_data[0]['env'].shape[0]
    print(f"Data dimensions: {num_snps} SNPs, {num_env_vars} environment variables")

    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return

    model = load_model(model_path, num_snps, num_env_vars, device)

    train_ds = GeneEnvDataset(train_val_data, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    print("\nExtracting token assignment weights...")
    model_assignments = extract_token_snp_weights(model, train_loader, device)

    if model_assignments is None:
        print("ERROR: Failed to extract token assignments from model")
        return

    print(f"Assignment matrix shape: {model_assignments.shape}")

    num_tokens = config['num_snp_tokens']
    print(f"\nRunning permutation tests for {num_tokens} SNP tokens...")
    p_values, observed_scores, random_distribution = permutation_test(
        model_assignments, snp_positions, num_tokens,
        top_k=args.top_k, n_permutations=args.n_permutations
    )

    output_dir = os.path.join(results_dir, "token_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # 1. 基因组Token分配热图（最重要，直接回答审稿人）
    genome_heatmap_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_genome_token_heatmap.png")
    plot_genome_token_heatmap(snp_positions, model_assignments, num_tokens, 
                               save_path=genome_heatmap_path)

    # 2. 每个Token在染色体上的分布
    token_dist_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_token_chromosome_distribution.png")
    plot_token_chromosome_distribution(snp_positions, model_assignments, num_tokens,
                                        top_k=args.top_k, save_path=token_dist_path)

    # 3. 传统聚集图
    agg_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_token_aggregation.png")
    plot_token_aggregation(snp_positions, model_assignments, num_tokens,
                          top_k=args.top_k, save_path=agg_path)

    # 4. 排列检验结果
    perm_test_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_permutation_test.png")
    plot_permutation_results(p_values, observed_scores, random_distribution, num_tokens,
                            save_path=perm_test_path)

    # 5. p值分布
    pvalue_dist_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_pvalue_distribution.png")
    plot_pvalue_distribution(p_values, save_path=pvalue_dist_path)

    # 6. Top SNP条形图
    barplot_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_top_snps.png")
    plot_token_assignment_barplot(model_assignments, num_tokens, top_n=20, save_path=barplot_path)

    # 保存结果JSON
    results = {
        'trait': args.trait,
        'fold': args.fold,
        'top_k': args.top_k,
        'n_permutations': args.n_permutations,
        'num_tokens': num_tokens,
        'p_values': p_values,
        'observed_scores': observed_scores,
        'significant_tokens_p005': sum(1 for p in p_values if p < 0.05),
        'significant_tokens_p001': sum(1 for p in p_values if p < 0.01),
        'mean_observed_score': float(np.mean(observed_scores)),
        'mean_random_score': float(np.mean([np.mean(rd) for rd in random_distribution]))
    }

    results_path = os.path.join(output_dir, f"{args.trait}_fold{args.fold}_token_biology_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_path}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Significant tokens (p < 0.05): {results['significant_tokens_p005']}/{num_tokens}")
    print(f"Highly significant tokens (p < 0.01): {results['significant_tokens_p001']}/{num_tokens}")
    print(f"Mean observed score: {results['mean_observed_score']:.4f}")
    print(f"Mean random score: {results['mean_random_score']:.4f}")
    print("="*60)
    print("\nOutput files:")
    print(f"  - Genome token heatmap: {genome_heatmap_path}")
    print(f"  - Token distribution: {token_dist_path}")
    print(f"  - Permutation test: {perm_test_path}")
    print(f"  - P-value distribution: {pvalue_dist_path}")
    print("="*60)


if __name__ == "__main__":
    main()