#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/K-EmoPhone/Shift_Analysis/Label_shift/Stress.ipynb for dataset K-EmoPhone."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "K-EmoPhone"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

X, y, groups, t, datetimes = pickle.load(open(str(DATA_ROOT / "Intermediate/features_stress_fixed-current.pkl"), mode='rb'))

df = pd.DataFrame({'User': groups, 'datetime': datetimes, 'Label': y})

# df_merged = pd.merge(df, X, left_index=True, right_index=True)
df_merged = pd.merge(df, X, left_index=True, right_index=True)

# Sort the DataFrame by datetime
df_merged = df_merged.sort_values(by=['User', 'datetime'])

# Update groups and datetimes
groups_specific = df_merged['User'].to_numpy()
datetimes = df_merged['datetime'].to_numpy()
y_specific = df_merged['Label'].to_numpy()
X = df_merged.drop(columns=['User', 'datetime', 'Label'])


# Calculate the count of each label per user
label_counts = df_merged.groupby(['User', 'Label']).size().reset_index(name='count')


# Pivot the data to have labels as columns
label_pivot = label_counts.pivot(index='User', columns='Label', values='count').fillna(0)


# Rename columns for clarity
label_pivot.columns = ['Stress_0', 'Stress_1']

# Calculate percentages for better comparison
label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100

# Reset index for plotting
label_pivot_percent = label_pivot_percent.reset_index()

# Melt the dataframe for seaborn
label_melted = label_pivot_percent.melt(id_vars='User', value_vars=['Stress_0', 'Stress_1'],
                                       var_name='Stress_Level', value_name='Percentage')

from scipy.stats import chi2_contingency

# Create contingency table for personal binary stress labels
contingency_personal_stress = pd.crosstab(
    df_merged['User'], 
    df_merged['Label']
)

# Perform Chi-Square Test
chi2_personal_stress, p_personal_stress, dof_personal_stress, ex_personal_stress = chi2_contingency(contingency_personal_stress)

print("Chi-Square Test for Personal Binary Stress Labels")
print("Chi2 Statistic:", chi2_personal_stress)
print("p-value:", p_personal_stress)
print("Degrees of Freedom:", dof_personal_stress)


import numpy as np

# Calculate Cohen's w effect size
n_total = contingency_personal_stress.sum().sum()
cohens_w = np.sqrt(chi2_personal_stress / n_total)

print(f"Cohen's w: {cohens_w:.4f}")
print(f"Sample size (n): {n_total}")
