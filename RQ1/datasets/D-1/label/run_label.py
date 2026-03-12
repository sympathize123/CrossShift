#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/D-1/Shift_Analysis/Label_Shift/Stress.ipynb for dataset D-1."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "D-1"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))



import os, sys
from Funcs.Utility import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(os.path.join( str(DATA_ROOT / "Intermediate/hourly_data_interpretable/stress_hourly_f&l.csv")))

print(df.head())

# Filter the data for the selected users
df_selected = df

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

# Calculate the count of each overall stress label per user
label_counts_overall = df_selected.groupby(['pcode', 'stress_binary_personal']).size().reset_index(name='count')

# Pivot the data to have labels as columns
label_pivot_overall = label_counts_overall.pivot(index='pcode', columns='stress_binary_personal', values='count').fillna(0)

# Rename columns for clarity
label_pivot_overall.columns = ['Stress_0_Overall', 'Stress_1_Overall']

# Calculate percentages for better comparison
label_pivot_overall_percent = label_pivot_overall.div(label_pivot_overall.sum(axis=1), axis=0) * 100

# Reset index for plotting
label_pivot_overall_percent = label_pivot_overall_percent.reset_index()

# Melt the dataframe for seaborn
label_melted_overall = label_pivot_overall_percent.melt(
    id_vars='pcode', 
    value_vars=['Stress_0_Overall', 'Stress_1_Overall'],
    var_name='Stress_Level_Overall', 
    value_name='Percentage'
)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(
    data=label_melted_overall, 
    x='pcode', 
    y='Percentage', 
    hue='Stress_Level_Overall', 
    palette='Set3'
)

# Add titles and labels
plt.title('Distribution of Stress Binary Overall Labels Across Users', fontsize=16)
plt.xlabel('User (pcode)', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.legend(title='Stress Level', labels=['0', '1'])
plt.ylim(0, 100)

# Show the plot
plt.tight_layout()
plt.show()

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

