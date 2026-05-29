import pandas as pd
import numpy as np

pheno = pd.read_csv('d:/doc/GEBiformer/data/Phenotypes.csv')
print(f'Phenotypes: {pheno.shape}')
print(f'Columns: {pheno.columns.tolist()}')
print(f'Unique Environments: {pheno["Environment"].nunique()}')
print(f'Unique Hybrids: {pheno["Hybrid"].nunique()}')

env_counts = pheno['Environment'].value_counts()
print(f'\nEnvironment sample counts (top 10):\n{env_counts.head(10)}')
print(f'Environment sample counts (bottom 10):\n{env_counts.tail(10)}')

hybrid_counts = pheno['Hybrid'].value_counts()
print(f'\nHybrid sample counts (top 10):\n{hybrid_counts.head(10)}')

# Count hybrids per environment
env_hybrid_count = pheno.groupby('Environment')['Hybrid'].nunique()
print(f'\nHybrids per Environment (top 10):\n{env_hybrid_count.head(10)}')
print(f'Hybrids per Environment (bottom 10):\n{env_hybrid_count.tail(10)}')

# Count environments per hybrid
hybrid_env_count = pheno.groupby('Hybrid')['Environment'].nunique()
print(f'\nEnvironments per Hybrid (top 10):\n{hybrid_env_count.head(10)}')
print(f'Environments per Hybrid (stats): mean={hybrid_env_count.mean():.1f}, min={hybrid_env_count.min()}, max={hybrid_env_count.max()}')

# Traits available
trait_cols = ['Yield', 'Grain Moisture', 'Pollen_DAP_days', 'Silk_DAP_days', 
              'Plant_Height_cm', 'Ear_Height_cm', 'Twt_kg_m3']
for t in trait_cols:
    if t in pheno.columns:
        non_null = pheno[t].notna().sum()
        print(f'Trait {t}: {non_null} non-null values')
