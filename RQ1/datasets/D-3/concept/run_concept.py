#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/D-3/Shift_Analysis/Concept_Shift/ESM_multivariate.ipynb for dataset D-3."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "D-3"
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

# For PERMANOVA and PERMDISP
from scipy.spatial.distance import euclidean, cdist
from skbio.stats.distance import DistanceMatrix, permanova
from scipy.stats import levene
from sklearn.cluster import KMeans

# ---- Run configuration ----
NORMALIZATION_MODE = 'user'  # 'user' or 'feature'
OUTPUT_DIR = str(RESULTS_ROOT / "RQ1/ConceptShift")

import os, sys
from Funcs.Utility import *

df = pd.read_csv(os.path.join(PATH_INTERMEDIATE_HOURLY_INTERPRETABLE, "stress_hourly_f&l.csv"))



df.rename(columns={'stress_binary_personal': 'label'}, inplace=True)

# Normalization mode: 'user' or 'feature'
# If NORMALIZATION_MODE is set above, use it; otherwise fall back to env
NORMALIZATION_MODE = str(globals().get('NORMALIZATION_MODE', '')).strip().lower() or os.getenv('NORMALIZATION_MODE', 'feature').strip().lower()
if NORMALIZATION_MODE not in {'user', 'feature'}:
    NORMALIZATION_MODE = 'feature'

def normalize_feature_wise(df):
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    exclude_cols = ['label', 'pcode']
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

def normalize_user_wise(df):
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    exclude_cols = ['label', 'pcode']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    for user_id, group_data in df_norm.groupby('pcode'):
        if len(group_data) > 1:
            means = group_data[numeric_cols].mean(axis=0)
            stds = group_data[numeric_cols].std(axis=0).replace(0, np.nan)
            scaled = (group_data[numeric_cols] - means) / stds
            df_norm.loc[group_data.index, numeric_cols] = scaled.fillna(0.0)
    return df_norm

if NORMALIZATION_MODE == 'user':
    print('Applying user-wise normalization...')
    df = normalize_user_wise(df)
else:
    print('Applying feature-wise normalization...')
    df = normalize_feature_wise(df)


df['label'] = df['label'].astype(str)

warnings.filterwarnings("ignore")

# Print basic info about the dataset
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns.tolist())

df_selected = df

def kmeans_binarize_multivariate(data, cols):
    """
    Binarizes a multivariate dataset (subset of columns) into two clusters using KMeans.
    The cluster with the lower sum of centers is labeled "l" and the other "h".
    Parameters:
      data: pandas DataFrame containing the columns.
      cols: list of column names to use for clustering.
    Returns:
      A numpy array of binary labels ("l" or "h") for each row.
    """
    # Extract the data for the given columns
    X = data[cols].values
    # Run KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Calculate a summary statistic (sum of feature values) for each center
    center_sums = centers.sum(axis=1)
    # Identify the cluster with the lower sum
    low_cluster = np.argmin(center_sums)
    # Create binary labels: "l" for the lower cluster, "h" for the higher cluster
    binary_labels = np.where(labels == low_cluster, "l", "h")
    return binary_labels

def discretize_df_multivariate(data, cols, pid, profile_name):
    """
    Applies multivariate KMeans binarization for the specified columns in the dataframe.
    The result is added as a new column whose name reflects the profile.
    """
    df_copy = deepcopy(data)
    try:
        # Apply multivariate binarization
        binarized = kmeans_binarize_multivariate(df_copy, cols)
        # Name the new column using the profile name, e.g., "Physical_profile"
        new_col = f"{profile_name}_profile"
        df_copy[new_col] = binarized
    except Exception as e:
        # If binarization fails, default to "l"
        new_col = f"{profile_name}_profile"
        df_copy[new_col] = "l"
    # Insert the user id as a new column and reset index
    df_copy.insert(0, 'pid', pid)
    df_copy.reset_index(drop=True, inplace=True)
    df_copy.set_index('pid', inplace=True)
    print(f"Multivariate Binarization (KMeans) applied for user {pid} on columns: {cols} -> new column '{new_col}'")
    return df_copy

# profiles = {
#     "Physical": [
#         "FitbitHeartrate_count",
#         "FitbitStepcount_count",
#         "Fitbitcalorie_count",
#         "Fitbitdistance_count",
#         "ACT#state_changes"
#     ],
#     "Mobility": [
#         'LOC_NUM_PLCS_VIST', 'LOC_TIME_NONE', 'LOC_TIME_WORK',
#         'LOC_TIME_EATING', 'LOC_TIME_SOCIAL', 'LOC_TIME_OTHERS', 'LOC_TIME_GYM', 'LOC_TIME_HOME'
#     ],
#     "Phone usage": [
#         'APP_DUR_INFO_sum',
#         'APP_DUR_ENTER_sum',
#         'APP_DUR_HEALTH_sum',
#         'APP_DUR_WORK_sum',
#         'keyevent_TIME_sum',
#         'APP_DUR_SOCIAL_sum',
#         'SCR_DUR_sum'
#     ],
#     "Sleep": [
#         'sleep_duration_hr', 'sleep_onset_hour', 'sleep_midpoint_hour'
#     ],
#     "Social": [
#         "MSG#CNT", "CALL#CNT", "CALL#DUR"
#     ]
# }
profiles = {
    "Physical": [
        "Fitbit",
        "ACT#"
    ],
    "Mobility": [
        'LOC_'
    ],
    "Phone usage": [
        'APP_DUR_',
        'keyevent_',
        'SCR_'
    ],
    "Sleep": [
        'sleep_'
    ],
    "Social": [
        "MSG#", "CALL#"
    ]
}

# chi_results = []

# sns.set(style="whitegrid")

# # -----------------------------
# # Iterate over each profile and perform analysis
# # -----------------------------
# for profile_name, feature_list in profiles.items():
#     print(f"\nProcessing profile: {profile_name}")
    
#     # Create an empty list to collect discretized data for all users for the current profile
#     list_df = []
    
#     # Discretize the profile for each selected user (user-specific multivariate binarization)
#     for user in selected_users:
#         df_user = df_selected[df_selected['pcode'] == user].copy()
#         df_user_disc = discretize_df_multivariate(df_user, feature_list, user, profile_name)
#         # Reset the index to keep 'pid'
#         df_user_disc = df_user_disc.reset_index()  # keep 'pid' as a column
#         list_df.append(df_user_disc)
    
#     # Concatenate the discretized data for all selected users
#     df_profile = pd.concat(list_df, ignore_index=True)
    
#     # The new column name for the profile
#     profile_col = f"{profile_name}_profile"
    
#     # Group by user (pid), the profile category, and the target label
#     label_counts = df_profile.groupby(['pid', profile_col, 'label']).size().reset_index(name='count')
    
#     # Pivot so that each row corresponds to a user and a profile bin, with columns for each label value.
#     label_pivot = label_counts.pivot_table(index=['pid', profile_col], columns='label', values='count', fill_value=0)
    
#     # Calculate percentages (row-wise)
#     label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100
#     label_pivot_percent = label_pivot_percent.reset_index()
    
#     # Melt the dataframe for plotting
#     label_melted = label_pivot_percent.melt(id_vars=['pid', profile_col], var_name='Label', value_name='Percentage')
    
#     # ---------------------------
#     # a. Faceted Bar Plot
#     # ---------------------------
#     plt.figure()
#     g = sns.catplot(
#         data=label_melted,
#         x="pid", 
#         y="Percentage", 
#         hue="Label", 
#         col=profile_col,
#         kind="bar", 
#         height=6, 
#         aspect=1,
#         palette="Set2"
#     )
#     g.set_axis_labels("User (pid)", "Percentage (%)")
#     g.set(ylim=(0, 100))
#     plt.subplots_adjust(top=0.85)
#     g.fig.suptitle(f'Concept Shift: Distribution of Label by {profile_name} Profile Categories Across Users', fontsize=16)
#     plt.show()
    
#     # ---------------------------
#     # b. Stacked Bar Charts per Profile Category
#     # ---------------------------
#     categories = sorted(label_melted[profile_col].unique())
#     for cat in categories:
#         df_cat = label_melted[label_melted[profile_col] == cat]
#         # Pivot so that rows are users and columns are label percentages
#         df_cat_pivot = df_cat.pivot(index="pid", columns="Label", values="Percentage").fillna(0)
#         ax = df_cat_pivot.plot(kind="bar", stacked=True, figsize=(10, 6),
#                                title=f'Stacked Bar Chart for {profile_name} Profile = {cat}', color=sns.color_palette("Set2"))
#         plt.xlabel("User (pid)")
#         plt.ylabel("Percentage (%)")
#         plt.ylim(0, 100)
#         plt.legend(title="Label")
#         plt.tight_layout()
#         plt.show()
    
#     # ---------------------------
#     # c. Chi-Square Tests: Record results in chi_results list
#     # ---------------------------
#     print(f"Chi-Square Test results for profile: {profile_name}")
#     for cat in categories:
#         df_cat_counts = df_profile[df_profile[profile_col] == cat]
#         contingency = pd.crosstab(df_cat_counts['pcode'], df_cat_counts['label'])
#         if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
#             chi2, p, dof, expected = chi2_contingency(contingency)
#             significance = "SIGNIFICANT" if p < 0.05 else "Not Significant"
            
#             # Calculate Cohen's w (effect size)
#             total_n = contingency.values.sum()
#             cohens_w = np.sqrt(chi2 / total_n) if total_n > 0 else 0
            
#             print(f"  Category {cat}: chi2 = {chi2:.2f}, p-value = {p:.2e}, dof = {dof}, Cohen's w = {cohens_w:.3f} --> {significance}")
#             chi_results.append({
#                 "Profile": profile_name,
#                 "Category": cat,
#                 "Chi2": round(chi2, 2),
#                 "p-value": f"{p:.2e}",
#                 "dof": dof,
#                 "Cohen's w": round(cohens_w, 3),
#                 "Significance": significance
#             })
#         else:
#             print(f"  Category {cat}: Not enough data to run chi-square test.")
#             chi_results.append({
#                 "Profile": profile_name,
#                 "Category": cat,
#                 "Chi2": None,
#                 "p-value": None,
#                 "dof": None,
#                 "Cohen's w": None,
#                 "Significance": "Not enough data"
#             })

# Add this function at the beginning of the cell
def get_columns_by_prefix(data, prefixes):
    """
    Get columns that start with any of the given prefixes
    """
    matching_columns = []
    for prefix in prefixes:
        matching_columns.extend([col for col in data.columns if col.startswith(prefix)])
    return matching_columns

chi_results = []

# Define selected_users as the unique 'pcode' values in df_selected
selected_users = df_selected['pcode'].unique()


sns.set(style="whitegrid")

# -----------------------------
# Iterate over each profile and perform analysis
# -----------------------------
for profile_name, feature_list in profiles.items():
    print(f"\nProcessing profile: {profile_name}")
    
    # Create an empty list to collect discretized data for all users for the current profile
    list_df = []
    
    # Discretize the profile for each selected user (user-specific multivariate binarization)
    for user in selected_users:
        df_user = df_selected[df_selected['pcode'] == user].copy()
        
        # Get actual columns that match the prefixes
        actual_columns = get_columns_by_prefix(df_user, feature_list)
        print(f"  User {user}: Found {len(actual_columns)} columns matching prefixes {feature_list}: {actual_columns}")
        
        df_user_disc = discretize_df_multivariate(df_user, actual_columns, user, profile_name)
        # Reset the index to keep 'pid'
        df_user_disc = df_user_disc.reset_index()  # keep 'pid' as a column
        list_df.append(df_user_disc)
    
    # Concatenate the discretized data for all selected users
    df_profile = pd.concat(list_df, ignore_index=True)
    
    # The new column name for the profile
    profile_col = f"{profile_name}_profile"
    
    # Group by user (pid), the profile category, and the target label
    label_counts = df_profile.groupby(['pid', profile_col, 'label']).size().reset_index(name='count')
    
    # Pivot so that each row corresponds to a user and a profile bin, with columns for each label value.
    label_pivot = label_counts.pivot_table(index=['pid', profile_col], columns='label', values='count', fill_value=0)
    
    # Calculate percentages (row-wise)
    label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100
    label_pivot_percent = label_pivot_percent.reset_index()
    
    # Melt the dataframe for plotting
    label_melted = label_pivot_percent.melt(id_vars=['pid', profile_col], var_name='Label', value_name='Percentage')
    
    # ---------------------------
    # a. Faceted Bar Plot
    # ---------------------------
    plt.figure()
    g = sns.catplot(
        data=label_melted,
        x="pid", 
        y="Percentage", 
        hue="Label", 
        col=profile_col,
        kind="bar", 
        height=6, 
        aspect=1,
        palette="Set2"
    )
    g.set_axis_labels("User (pid)", "Percentage (%)")
    g.set(ylim=(0, 100))
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(f'Concept Shift: Distribution of Label by {profile_name} Profile Categories Across Users', fontsize=16)
    plt.show()
    
    # ---------------------------
    # b. Stacked Bar Charts per Profile Category
    # ---------------------------
    categories = sorted(label_melted[profile_col].unique())
    for cat in categories:
        df_cat = label_melted[label_melted[profile_col] == cat]
        # Pivot so that rows are users and columns are label percentages
        df_cat_pivot = df_cat.pivot(index="pid", columns="Label", values="Percentage").fillna(0)
        ax = df_cat_pivot.plot(kind="bar", stacked=True, figsize=(10, 6),
                               title=f'Stacked Bar Chart for {profile_name} Profile = {cat}', color=sns.color_palette("Set2"))
        plt.xlabel("User (pid)")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.legend(title="Label")
        plt.tight_layout()
        plt.show()
    
    # ---------------------------
    # c. Chi-Square Tests: Record results in chi_results list
    # ---------------------------
    print(f"Chi-Square Test results for profile: {profile_name}")
    for cat in categories:
        df_cat_counts = df_profile[df_profile[profile_col] == cat]
        contingency = pd.crosstab(df_cat_counts['pcode'], df_cat_counts['label'])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(contingency)
            significance = "SIGNIFICANT" if p < 0.05 else "Not Significant"
            
            # Calculate Cohen's w (effect size)
            total_n = contingency.values.sum()
            cohens_w = np.sqrt(chi2 / total_n) if total_n > 0 else 0
            
            print(f"  Category {cat}: chi2 = {chi2:.2f}, p-value = {p:.2e}, dof = {dof}, Cohen's w = {cohens_w:.3f} --> {significance}")
            chi_results.append({
                "Profile": profile_name,
                "Category": cat,
                "Chi2": round(chi2, 2),
                "p-value": f"{p:.2e}",
                "dof": dof,
                "Cohen's w": round(cohens_w, 3),
                "Significance": significance
            })
        else:
            print(f"  Category {cat}: Not enough data to run chi-square test.")
            chi_results.append({
                "Profile": profile_name,
                "Category": cat,
                "Chi2": None,
                "p-value": None,
                "dof": None,
                "Cohen's w": None,
                "Significance": "Not enough data"
            })

chi_results_df = pd.DataFrame(chi_results)

def highlight_significance(val):
    if val == "SIGNIFICANT":
        return "background-color: lightgreen"
    elif val == "Not Significant":
        return "background-color: lightcoral"
    else:
        return ""

styled_table = chi_results_df.style.applymap(highlight_significance, subset=["Significance"]) \
                                   .set_properties(**{'text-align': 'center'}) \
                                   .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

print("Chi-Square Test Summary:")
display(styled_table)

# Save Chi-Square summary table
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f'chi_square_summary_{NORMALIZATION_MODE}.csv')
chi_results_df.to_csv(output_path, index=False)
print(f'Saved Chi-Square summary to: {output_path}')

