#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/CollegeExperience/Shift_Analysis/Concept_Shift/ESM_multivariate.ipynb for dataset CollegeExperience."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "CollegeExperience"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from copy import deepcopy
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(str(DATA_ROOT))
INTERMEDIATE_DIR = Path(os.getenv('INTERMEDIATE_DIR', PROJECT_ROOT / 'data_preprocessing/college_experience_dataset/processed_data/android/android_ios_2week_aggregations_fill.csv'))

sns.set_theme(style='whitegrid')
print(f'INTERMEDIATE_DIR: {INTERMEDIATE_DIR}')


# ---- Run configuration ----
NORMALIZATION_MODE = 'feature'  # 'user' or 'feature'
OUTPUT_DIR = str(RESULTS_ROOT / "RQ1/ConceptShift")

# Dataset list (single dataset)
dataset_paths = [
    Path(str(DATA_ROOT / "data_preprocessing/college_experience_dataset/processed_data/android/android_ios_2week_aggregations_fill.csv"))
]

print(f'Found {len(dataset_paths)} dataset(s): {[p.name for p in dataset_paths]}')

# Configuration
LABEL_CANDIDATES = ['stress_binary_personal']
ID_CANDIDATES = ['pcode']
# Prefixes per profile; adjust as needed for other domains
PROFILES = {
    'phone_usage': ['unlock_'],
    'mobility': ['loc_'],
    'physical_status': ['act_', 'step_'],
    'sleep': ['sleep'],
    'social_behavior': ['call_', 'sms_', 'audio_']
}
def detect_id_col(df):
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col
    return None
def detect_label_col(df):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    return None
def get_columns_by_prefix(data, prefixes):
    cols = []
    for prefix in prefixes:
        cols.extend([c for c in data.columns if c.startswith(prefix)])
    # drop duplicates while preserving order
    seen = set()
    uniq = []
    for c in cols:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq
def kmeans_binarize_multivariate(data, cols):
    if len(cols) == 0:
        return np.array(['l'] * len(data))
    X = data[cols].values
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    center_sums = centers.sum(axis=1)
    low_cluster = np.argmin(center_sums)
    return np.where(labels == low_cluster, 'l', 'h')
def discretize_df_multivariate(data, cols, id_col, profile_name):
    df_copy = deepcopy(data)
    try:
        binarized = kmeans_binarize_multivariate(df_copy, cols)
    except Exception:
        binarized = np.array(['l'] * len(df_copy))
    new_col = f"{profile_name}_profile"
    df_copy[new_col] = binarized
    df_copy = df_copy[[id_col, new_col, 'label']].copy()
    df_copy[id_col] = df_copy[id_col].astype(str)
    return df_copy

# Normalization mode: 'user' or 'feature'
# If NORMALIZATION_MODE is set above, use it; otherwise fall back to env
NORMALIZATION_MODE = str(globals().get('NORMALIZATION_MODE', '')).strip().lower() or os.getenv('NORMALIZATION_MODE', 'feature').strip().lower()
if NORMALIZATION_MODE not in {'user', 'feature'}:
    NORMALIZATION_MODE = 'feature'

def normalize_feature_wise(df, exclude_cols):
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    for col in numeric_cols:
        vals = pd.to_numeric(df_norm[col], errors='coerce')
        minv = vals.min(skipna=True)
        maxv = vals.max(skipna=True)
        if pd.isna(minv) or pd.isna(maxv):
            df_norm[col] = vals
        elif maxv == minv:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (vals - minv) / (maxv - minv)
    return df_norm

def normalize_user_wise(df, group_col, exclude_cols):
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    for _, group_data in df_norm.groupby(group_col):
        if len(group_data) > 1:
            means = group_data[numeric_cols].mean(axis=0)
            stds = group_data[numeric_cols].std(axis=0).replace(0, np.nan)
            scaled = (group_data[numeric_cols] - means) / stds
            df_norm.loc[group_data.index, numeric_cols] = scaled.fillna(0.0)
    return df_norm

chi_results = []

for dataset_path in dataset_paths:
    dataset_label = dataset_path.stem
    print(f"""
{'='*80}
Dataset: {dataset_path.name}
{'='*80}
""")
    df = pd.read_csv(dataset_path)
    id_col = detect_id_col(df)
    label_col = detect_label_col(df)

    if not id_col or not label_col:
        print(f"- Skipping (id_col={id_col}, label_col={label_col}).")

    df = df.dropna(subset=[label_col]).copy()
    df['label'] = df[label_col].astype(str)
    # Apply normalization
    exclude_cols = [id_col, label_col, 'label']
    if NORMALIZATION_MODE == 'user':
        print('  - Normalization: user-wise')
        df = normalize_user_wise(df, id_col, exclude_cols)
    else:
        print('  - Normalization: feature-wise')
        df = normalize_feature_wise(df, exclude_cols)
    selected_users = df[id_col].dropna().unique()
    print(f"- Using id_col={id_col}, label_col={label_col}, participants={len(selected_users)}")

    for profile_name, prefixes in PROFILES.items():
        list_df = []
        for user in selected_users:
            df_user = df[df[id_col] == user]
            actual_columns = get_columns_by_prefix(df_user, prefixes)
            if len(actual_columns) == 0:
            df_user_disc = discretize_df_multivariate(df_user, actual_columns, id_col, profile_name)
            list_df.append(df_user_disc)

        if not list_df:
            print(f"  - {profile_name}: no matching columns, skipped.")

        df_profile = pd.concat(list_df, ignore_index=True)
        profile_col = f"{profile_name}_profile"

        label_counts = df_profile.groupby([id_col, profile_col, 'label']).size().reset_index(name='count')
        label_pivot = label_counts.pivot_table(index=[id_col, profile_col], columns='label', values='count', fill_value=0)
        label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100
        label_pivot_percent = label_pivot_percent.reset_index()

        # Plot faceted percentages
        label_melted = label_pivot_percent.melt(id_vars=[id_col, profile_col], var_name='Label', value_name='Percentage')
        g = sns.catplot(
            data=label_melted,
            x=id_col,
            y='Percentage',
            hue='Label',
            col=profile_col,
            kind='bar',
            height=5,
            aspect=1,
            palette='Set2'
        )
        g.set_axis_labels(f"Participant ({id_col})", 'Percentage (%)')
        g.set(ylim=(0, 100))
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle(f"{dataset_label}: Label distribution by {profile_name} profile")
        plt.show()

        # Chi-square per profile category
        categories = sorted(label_melted[profile_col].unique())
        for cat in categories:
            df_cat_counts = df_profile[df_profile[profile_col] == cat]
            contingency = pd.crosstab(df_cat_counts[id_col], df_cat_counts['label'])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p, dof, expected = chi2_contingency(contingency)
                total_n = contingency.values.sum()
                cohens_w = np.sqrt(chi2 / total_n) if total_n > 0 else 0
                sig = 'SIGNIFICANT' if p < 0.05 else 'Not Significant'
            else:
                chi2 = p = dof = cohens_w = None
                sig = 'Not enough data'
            chi_results.append({
                'dataset': dataset_label,
                'profile': profile_name,
                'category': cat,
                'chi2': None if chi2 is None else round(chi2, 3),
                'p_value': p,
                'dof': dof,
                'cohens_w': None if cohens_w is None else round(cohens_w, 3),
                'significance': sig,
                'participants': contingency.shape[0]
            })


chi_results_df = pd.DataFrame(chi_results)


# Save Chi-Square summary table
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f'chi_square_summary_{NORMALIZATION_MODE}.csv')
chi_results_df.to_csv(output_path, index=False)
print(f'Saved Chi-Square summary to: {output_path}')

