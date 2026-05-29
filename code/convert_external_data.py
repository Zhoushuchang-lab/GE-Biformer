"""
将外部数据集 (data/other_data) 转换为项目统一格式
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'data', 'other_data')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'other_data', 'converted')

os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 60)
print('外部数据格式转换')
print('=' * 60)

# ========== 1. 基因型数据转换 ==========
print('\n[1/3] 转换基因型数据...')

g_add = pd.read_csv(os.path.join(SRC_DIR, 'g_data_add.csv'))
g_dom = pd.read_csv(os.path.join(SRC_DIR, 'g_data_dom.csv'))
n_geno, n_marker = g_add.shape
print(f'  g_data_add: {g_add.shape}')
print(f'  g_data_dom: {g_dom.shape}')

g_combined = g_add.values.astype(np.float32) + g_dom.values.astype(np.float32)
genotype_ids = [f'Geno_{i+1}' for i in range(n_geno)]
marker_ids = list(g_add.columns)

rows = []
rows.append('<Marker>\t' + '\t'.join(genotype_ids))
for i, marker_id in enumerate(marker_ids):
    row = marker_id + '\t' + '\t'.join(str(v) for v in g_combined[:, i])
    rows.append(row)

genotype_output = os.path.join(OUT_DIR, 'genotype.tsv')
with open(genotype_output, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rows))

print(f'  已保存: {genotype_output} ({n_marker} markers x {n_geno} genotypes)')

# ========== 2. 环境数据转换 ==========
print('\n[2/3] 转换环境数据...')

ev_data = pd.read_csv(os.path.join(SRC_DIR, 'ev_data.csv'))
n_env, n_var = ev_data.shape
print(f'  ev_data: {ev_data.shape}')

env_ids = [f'Env_{i+1}' for i in range(n_env)]
var_ids = list(ev_data.columns)

env_header = 'ENV_Variable,' + ','.join(env_ids)
lines = [env_header]
for i, var_id in enumerate(var_ids):
    line = var_id + ',' + ','.join(str(v) for v in ev_data.iloc[:, i].values)
    lines.append(line)

env_output = os.path.join(OUT_DIR, 'Environment_data.csv')
with open(env_output, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'  已保存: {env_output} ({n_var} vars x {n_env} envs)')

# ========== 3. 表型数据转换 ==========
print('\n[3/3] 转换表型数据...')

pheno = pd.read_csv(os.path.join(SRC_DIR, 'pheno_wtn.csv'))
print(f'  pheno_wtn: {pheno.shape}')

pheno_out = pheno[['Env_n', 'Geno_n', 'BLUEs_wtn']].copy()
pheno_out.columns = ['Environment', 'Hybrid', 'BLUEs_wtn']

pheno_output = os.path.join(OUT_DIR, 'Phenotypes.csv')
pheno_out.to_csv(pheno_output, index=False)
print(f'  已保存: {pheno_output} ({len(pheno_out)} rows)')

print('\n' + '=' * 60)
print('转换完成！')
print('=' * 60)
