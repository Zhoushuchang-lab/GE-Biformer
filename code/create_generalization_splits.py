import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def dedup_pairs(df):
    return df.drop_duplicates(subset=['Hybrid', 'Environment'], keep='first').reset_index(drop=True)


def draw_non_overlapping_test_groups(items, n_groups, test_pct, rng):
    """Draw n_groups non-overlapping subsets, each ~test_pct of items."""
    n_total = len(items)
    n_per_group = max(1, int(n_total * test_pct))
    total_needed = n_per_group * n_groups
    if total_needed > n_total:
        n_per_group = max(1, n_total // n_groups)
        total_needed = n_per_group * n_groups
    perm = rng.permutation(items)
    groups = []
    for i in range(n_groups):
        start = i * n_per_group
        end = start + n_per_group
        groups.append(set(perm[start:end]))
    actual_pct = n_per_group / n_total * 100
    print(f'  Each test group: {n_per_group}/{n_total} ({actual_pct:.1f}%) items')
    return groups


def create_unseen_env_splits(df, n_splits=5, test_pct=0.10, random_state=42):
    rng = np.random.default_rng(random_state)
    envs = df['Environment'].unique()
    test_pct_display = test_pct * 100
    print(f'\n=== Unseen Environment Splits (target: ~{test_pct_display:.0f}% test, {n_splits} folds) ===')
    env_groups = draw_non_overlapping_test_groups(envs, n_splits, test_pct, rng)

    base_dir = Path('data')
    for i, test_envs in enumerate(env_groups):
        test_df = df[df['Environment'].isin(test_envs)].reset_index(drop=True)
        train_df = df[~df['Environment'].isin(test_envs)].reset_index(drop=True)

        out_dir = base_dir / f'unseen_environment_data{i+1}'
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / 'train.csv', index=False)
        test_df.to_csv(out_dir / 'test.csv', index=False)

        _print_split_stats(f'Unseen Env Split {i+1}', train_df, test_df,
                           check_env_overlap=True, check_geno_overlap=False)


def create_unseen_geno_splits(df, n_splits=5, test_pct=0.10, random_state=42):
    rng = np.random.default_rng(random_state)
    hybrids = df['Hybrid'].unique()
    test_pct_display = test_pct * 100
    print(f'\n=== Unseen Genotype Splits (target: ~{test_pct_display:.0f}% test, {n_splits} folds) ===')
    hybrid_groups = draw_non_overlapping_test_groups(hybrids, n_splits, test_pct, rng)

    base_dir = Path('data')
    for i, test_hybrids in enumerate(hybrid_groups):
        test_df = df[df['Hybrid'].isin(test_hybrids)].reset_index(drop=True)
        train_df = df[~df['Hybrid'].isin(test_hybrids)].reset_index(drop=True)

        out_dir = base_dir / f'unseen_genotype_data{i+1}'
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / 'train.csv', index=False)
        test_df.to_csv(out_dir / 'test.csv', index=False)

        _print_split_stats(f'Unseen Geno Split {i+1}', train_df, test_df,
                           check_env_overlap=False, check_geno_overlap=True)


def create_unseen_both_splits(df, n_splits=5, final_test_ratio=0.10, random_state=42):
    rng = np.random.default_rng(random_state)

    a = final_test_ratio
    frac = (a - np.sqrt(a * (1 - a))) / (2 * a - 1) if a < 0.5 else 0.5
    frac = max(min(frac, 0.5), 0.05)

    envs = df['Environment'].unique()
    hybrids = df['Hybrid'].unique()
    n_env_sample = max(1, int(len(envs) * frac))
    n_hyb_sample = max(1, int(len(hybrids) * frac))

    print(f'\n=== Unseen Both Splits (target: ~{final_test_ratio*100:.0f}% test, {n_splits} folds) ===')
    print(f'  Using per-fold: {n_env_sample}/{len(envs)} environments ({frac*100:.1f}%), '
          f'{n_hyb_sample}/{len(hybrids)} hybrids ({frac*100:.1f}%)')

    env_groups = draw_non_overlapping_test_groups(envs, n_splits, frac, rng)
    hybrid_groups = draw_non_overlapping_test_groups(hybrids, n_splits, frac, rng)

    base_dir = Path('data')

    for i in range(n_splits):
        test_envs = env_groups[i]
        test_hybrids = hybrid_groups[i]

        test_mask = df['Environment'].isin(test_envs) & df['Hybrid'].isin(test_hybrids)
        test_df = df[test_mask].reset_index(drop=True)

        exclude_mask = df['Environment'].isin(test_envs) | df['Hybrid'].isin(test_hybrids)
        train_df = df[~exclude_mask].reset_index(drop=True)

        out_dir = base_dir / f'unseen_both_data{i+1}'
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / 'train.csv', index=False)
        test_df.to_csv(out_dir / 'test.csv', index=False)

        print(f'\nUnseen Both Fold {i+1}: test envs={len(test_envs)}/{df["Environment"].nunique()} '
              f'({len(test_envs)/df["Environment"].nunique()*100:.1f}%), '
              f'test hybrids={len(test_hybrids)}/{df["Hybrid"].nunique()} '
              f'({len(test_hybrids)/df["Hybrid"].nunique()*100:.1f}%)')
        _print_split_stats(f'Unseen Both Split {i+1}', train_df, test_df,
                           check_env_overlap=True, check_geno_overlap=True)


def _print_split_stats(name, train_df, test_df, check_env_overlap=False, check_geno_overlap=False):
    total = len(train_df) + len(test_df)
    test_pct = len(test_df) / total * 100
    print(f'\n--- {name} ---')
    print(f'  Total samples: {total}')
    print(f'  Train samples: {len(train_df)}')
    print(f'  Test samples:  {len(test_df)}')
    print(f'  Test percentage: {test_pct:.2f}%')

    if check_env_overlap:
        train_envs = set(train_df['Environment'].unique())
        test_envs = set(test_df['Environment'].unique())
        env_overlap = train_envs & test_envs
        print(f'  Unique environments in train: {len(train_envs)}, test: {len(test_envs)}')
        print(f'  Environment overlap: {len(env_overlap)} (should be 0)')

    if check_geno_overlap:
        train_hyb = set(train_df['Hybrid'].unique())
        test_hyb = set(test_df['Hybrid'].unique())
        hyb_overlap = train_hyb & test_hyb
        print(f'  Unique hybrids in train: {len(train_hyb)}, test: {len(test_hyb)}')
        print(f'  Hybrid overlap: {len(hyb_overlap)} (should be 0)')


def main():
    parser = argparse.ArgumentParser(description='Generate generalization experiment splits.')
    parser.add_argument('--experiment', required=True,
                        choices=['unseen_env', 'unseen_geno', 'unseen_both', 'all'],
                        help='Which experiment to run (or "all" for all three)')
    args = parser.parse_args()

    csv_path = Path('data') / 'Phenotypes.csv'
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent.parent / 'data' / 'Phenotypes.csv'

    print(f'Reading {csv_path} ...')
    df = pd.read_csv(csv_path)
    original_count = len(df)
    df = dedup_pairs(df)
    print(f'Loaded {original_count} rows, {len(df)} unique (Hybrid, Environment) pairs after dedup')
    print(f'Unique environments: {df["Environment"].nunique()}')
    print(f'Unique hybrids:     {df["Hybrid"].nunique()}')

    if args.experiment in ('unseen_env', 'all'):
        create_unseen_env_splits(df)

    if args.experiment in ('unseen_geno', 'all'):
        create_unseen_geno_splits(df)

    if args.experiment in ('unseen_both', 'all'):
        create_unseen_both_splits(df)

    print('\nAll done.')


if __name__ == '__main__':
    main()
