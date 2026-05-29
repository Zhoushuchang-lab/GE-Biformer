import pandas as pd

# pheno_wtn stats
pheno = pd.read_csv('d:/doc/GEBiformer/data/other_data/pheno_wtn.csv')
print('=== pheno_wtn.csv ===')
print(f'Shape: {pheno.shape}')
print(f'Columns: {pheno.columns.tolist()}')
print(f'Unique Environments: {pheno["Env_n"].nunique()}')
print(f'Unique Genotypes: {pheno["Geno_n"].nunique()}')
print(f'BLUEs_wtn range: {pheno["BLUEs_wtn"].min():.2f} - {pheno["BLUEs_wtn"].max():.2f}')

# g_data_add stats
add = pd.read_csv('d:/doc/GEBiformer/data/other_data/g_data_add.csv')
print(f'\n=== g_data_add.csv ===')
print(f'Shape: {add.shape}')

# g_data_dom stats
dom = pd.read_csv('d:/doc/GEBiformer/data/other_data/g_data_dom.csv')
print(f'\n=== g_data_dom.csv ===')
print(f'Shape: {dom.shape}')

# ev_data stats
ev = pd.read_csv('d:/doc/GEBiformer/data/other_data/ev_data.csv')
print(f'\n=== ev_data.csv ===')
print(f'Shape: {ev.shape}')
print(f'Rows (environments): {ev.shape[0]}')
print(f'Columns (env vars): {ev.shape[1]}')

print(f'\n=== Summary ===')
print(f'g_data_add: {add.shape[0]} genotypes x {add.shape[1]} markers')
print(f'g_data_dom: {dom.shape[0]} genotypes x {dom.shape[1]} markers')
print(f'ev_data: {ev.shape[0]} environments with {ev.shape[1]} env vars')
print(f'pheno_wtn: {pheno.shape[0]} gene-env combos')
