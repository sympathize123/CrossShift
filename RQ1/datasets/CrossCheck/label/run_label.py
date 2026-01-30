#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/CrossCheck/Shift_Analysis/Label_Shift/Stress.ipynb for dataset CrossCheck."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "CrossCheck"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))



import os, sys
from Funcs.Utility import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset /var/nfs_share/Overfitting/CrossCheck/Intermediate/ema_f.csv
df = pd.read_csv(os.path.join( str(DATA_ROOT / "ema_f&l_full.csv")))

print(df.head())

# Filter the data for the selected users
df_selected = df

# --- Normalization before label shift ---
from sklearn.preprocessing import StandardScaler

ID_COLS = ['pcode', 'date']
LABEL_COL = 'stress_binary_personal'
feature_cols = [c for c in df_selected.columns if c not in ID_COLS + [LABEL_COL]]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_selected[c])]

def normalize_per_user(df, features, user_col='pcode'):
    parts = []
    for _, sub in df.groupby(user_col):
        X = sub[features].astype(float).fillna(0)
        scaled = StandardScaler().fit_transform(X)
        sub_scaled = sub.copy()
        sub_scaled[features] = scaled
        parts.append(sub_scaled)
    return pd.concat(parts, ignore_index=True)

def normalize_per_feature(df, features):
    X = df[features].astype(float).fillna(0)
    scaled = StandardScaler().fit_transform(X)
    df_scaled = df.copy()
    df_scaled[features] = scaled
    return df_scaled

# Normalize with explicit options for comparison
NORMALIZATION_MODE = 'feature'  # options: 'user', 'feature'

df_user_norm = normalize_per_user(df_selected, feature_cols)
df_feature_norm = normalize_per_feature(df_selected, feature_cols)

if NORMALIZATION_MODE == 'user':
    df_selected = df_user_norm
elif NORMALIZATION_MODE == 'feature':
    df_selected = df_feature_norm
else:
    raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}")

# Use normalized data for label shift calculations (labels unchanged)

# Calculate the count of each label per user
label_counts = df_selected.groupby(['pcode', 'stress_binary_personal']).size().reset_index(name='count')

# Pivot the data to have labels as columns
label_pivot = label_counts.pivot(index='pcode', columns='stress_binary_personal', values='count').fillna(0)

# Rename columns for clarity
label_pivot.columns = ['Stress_0', 'Stress_1']

# Calculate percentages for better comparison
label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100

# Reset index for plotting
label_pivot_percent = label_pivot_percent.reset_index()

# Melt the dataframe for seaborn
label_melted = label_pivot_percent.melt(id_vars='pcode', value_vars=['Stress_0', 'Stress_1'],
                                       var_name='Stress_Level', value_name='Percentage')

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=label_melted, x='pcode', y='Percentage', hue='Stress_Level', palette='Set2')

# Add titles and labels
plt.title('Distribution of Stress Binary Personal Labels Across Users', fontsize=16)
plt.xlabel('User (pcode)', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.legend(title='Stress Level', labels=['0', '1'])
plt.ylim(0, 100)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate the count of each label per user
label_counts = df_selected.groupby(['pcode', 'stress_binary_personal']).size().unstack(fill_value=0)

# Rename columns for clarity
label_counts.columns = ['Stress_0', 'Stress_1']

# Plotting
label_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])

# Add titles and labels
plt.title('Stacked Bar Chart of Stress Binary Personal Labels Across Users', fontsize=16)
plt.xlabel('User (pcode)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Stress Level')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# # Calculate the count of each overall stress label per user
# label_counts_overall = df_selected.groupby(['uid', 'stress_binary_overall']).size().reset_index(name='count')

# # Pivot the data to have labels as columns
# label_pivot_overall = label_counts_overall.pivot(index='uid', columns='stress_binary_overall', values='count').fillna(0)

# # Rename columns for clarity
# label_pivot_overall.columns = ['Stress_0_Overall', 'Stress_1_Overall']

# # Calculate percentages for better comparison
# label_pivot_overall_percent = label_pivot_overall.div(label_pivot_overall.sum(axis=1), axis=0) * 100

# # Reset index for plotting
# label_pivot_overall_percent = label_pivot_overall_percent.reset_index()

# # Melt the dataframe for seaborn
# label_melted_overall = label_pivot_overall_percent.melt(
#     id_vars='uid', 
#     value_vars=['Stress_0_Overall', 'Stress_1_Overall'],
#     var_name='Stress_Level_Overall', 
#     value_name='Percentage'
# )

# # Plotting
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     data=label_melted_overall, 
#     x='uid', 
#     y='Percentage', 
#     hue='Stress_Level_Overall', 
#     palette='Set3'
# )

# # Add titles and labels
# plt.title('Distribution of Stress Binary Overall Labels Across Users', fontsize=16)
# plt.xlabel('User (uid)', fontsize=14)
# plt.ylabel('Percentage (%)', fontsize=14)
# plt.legend(title='Stress Level', labels=['0', '1'])
# plt.ylim(0, 100)

# # Show the plot
# plt.tight_layout()
# plt.show()

from scipy.stats import chi2_contingency

# Create contingency table for personal binary stress labels
contingency_personal_stress = pd.crosstab(
    df_selected['pcode'], 
    df_selected['stress_binary_personal']
)

# Perform Chi-Square Test
chi2_personal_stress, p_personal_stress, dof_personal_stress, ex_personal_stress = chi2_contingency(contingency_personal_stress)

print("Chi-Square Test for Personal Binary Stress Labels")
print("Chi2 Statistic:", chi2_personal_stress)
print("p-value:", p_personal_stress)
print("Degrees of Freedom:", dof_personal_stress)

# Calculate Cohen's w for both stress label types
import numpy as np

def cohens_w(chi2_statistic, n):
    """
    Calculate Cohen's w effect size for chi-square tests
    
    Parameters:
    chi2_statistic: Chi-square statistic
    n: Total sample size
    
    Returns:
    w: Cohen's w effect size
    """
    w = np.sqrt(chi2_statistic / n)
    return w

# Get total sample size for personal stress
n_personal = contingency_personal_stress.values.sum()

# Calculate Cohen's w for personal stress
w_personal = cohens_w(chi2_personal_stress, n_personal)

print("Cohen's w for Personal Binary Stress Labels:")
print(f"Effect size (w): {w_personal:.4f}")
print(f"Sample size (n): {n_personal}")


# Interpret effect sizes
def interpret_cohens_w(w):
    """Interpret Cohen's w effect size"""
    if w < 0.1:
        return "Small"
    elif w < 0.3:
        return "Medium"
    elif w < 0.5:
        return "Large"
    else:
        return "Very Large"

print(f"\nEffect Size Interpretation:")
print(f"Personal stress: {interpret_cohens_w(w_personal)} effect")


# from scipy.stats import chi2_contingency

# # Create contingency table for overall binary stress labels
# contingency_personal_stress = pd.crosstab(
#     df_selected['uid'], 
#     df_selected['stress_binary_overall']
# )

# # Perform Chi-Square Test
# chi2_personal_stress, p_personal_stress, dof_personal_stress, ex_personal_stress = chi2_contingency(contingency_personal_stress)

# print("Chi-Square Test for Overall Binary Stress Labels")
# print("Chi2 Statistic:", chi2_personal_stress)
# print("p-value:", p_personal_stress)
# print("Degrees of Freedom:", dof_personal_stress)

