#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/GLOBEM/Shift_Analysis/Label_shift/Stress.ipynb for dataset GLOBEM."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "GLOBEM"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(str(DATA_ROOT))
INTERMEDIATE_DIR = Path(os.getenv('INTERMEDIATE_DIR', PROJECT_ROOT / 'Intermediate'))

sns.set_theme(style='whitegrid')
print(f'INTERMEDIATE_DIR: {INTERMEDIATE_DIR}')


# dataset_paths = sorted(INTERMEDIATE_DIR.glob('*.csv'))
# if not dataset_paths:
#     raise FileNotFoundError(f'No CSV datasets found in {INTERMEDIATE_DIR}')


dataset_paths = [
    Path(str(DATA_ROOT / "Intermediate/dep--INS-W_merged_cat_processed.csv"))
]


dataset_filter = os.getenv('DATASET_NAME')
if dataset_filter:
    matches = [p for p in dataset_paths if p.stem == dataset_filter or p.name == dataset_filter]
    if not matches:
        raise FileNotFoundError(f'DATASET_NAME={dataset_filter} not found in {INTERMEDIATE_DIR}')
    dataset_paths = matches

print(f'Found {len(dataset_paths)} dataset(s): {[p.name for p in dataset_paths]}')


LABEL_CANDIDATES = ['depression_binary'] # 'stress_binary'
ID_CANDIDATES = ['pid']

def detect_id_col(df):
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col
    return None

def detect_label_cols(df):
    return [c for c in LABEL_CANDIDATES if c in df.columns]

def compute_label_tables(df, id_col, label_col):
    data = df.dropna(subset=[label_col])
    label_counts = data.groupby([id_col, label_col]).size().reset_index(name='count')
    label_pivot = label_counts.pivot(index=id_col, columns=label_col, values='count').fillna(0)
    label_pivot.columns = label_pivot.columns.astype(str)
    label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100
    return label_counts, label_pivot, label_pivot_percent.reset_index()

def plot_percentages(label_pivot_percent, id_col, label_col, dataset_label):
    melted = label_pivot_percent.melt(id_vars=id_col, var_name='Label', value_name='Percentage')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x=id_col, y='Percentage', hue='Label', palette='Set2')
    plt.title(f"{dataset_label}: Distribution of {label_col} (%)")
    plt.ylabel('Percentage')
    plt.xlabel(id_col)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_counts(label_pivot, id_col, label_col, dataset_label):
    ax = label_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))
    ax.set_title(f"{dataset_label}: Counts of {label_col}")
    ax.set_xlabel(id_col)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def chi_square_effect(df, id_col, label_col):
    contingency = pd.crosstab(df[id_col], df[label_col])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    n = contingency.values.sum()
    w = np.sqrt(chi2 / n) if n > 0 else np.nan
    return {'chi2': chi2, 'p_value': p_val, 'dof': dof, 'cohens_w': w, 'n': n}


all_results = []

for dataset_path in dataset_paths:
    dataset_label = dataset_path.stem
    print(f"""
{'='*80}
Dataset: {dataset_path.name}
{'='*80}
""")
    df = pd.read_csv(dataset_path)
    id_col = detect_id_col(df)
    if not id_col:
        print('- Skipping (no participant id column found).')

    label_cols = detect_label_cols(df)
    if not label_cols:
        print('- Skipping (no label column found).')

    for label_col in label_cols:
        df_label = df.dropna(subset=[label_col])
        if df_label.empty:
            print(f"- {label_col}: no data after dropping NaN, skipping.")

        label_counts, label_pivot, label_pivot_percent = compute_label_tables(df_label, id_col, label_col)
        plot_percentages(label_pivot_percent, id_col, label_col, dataset_label)
        plot_counts(label_pivot, id_col, label_col, dataset_label)

        stats = chi_square_effect(df_label, id_col, label_col)
        stats.update({
            'dataset': dataset_label,
            'label_col': label_col,
            'n_participants': label_pivot.shape[0]
        })
        all_results.append(stats)
        print(f"- {label_col}: chi2={stats['chi2']:.3f}, p={stats['p_value']:.4f}, w={stats['cohens_w']:.4f}, n={stats['n']}")


summary_df = pd.DataFrame(all_results)


