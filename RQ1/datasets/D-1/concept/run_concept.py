#!/usr/bin/env python3
"""Converted from /var/nfs_share/Overfitting/D-1/Shift_Analysis/Concept_Shift /Stress.multivariate.ipynb for dataset D-1."""

from pathlib import Path
import os
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "D-1"
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

import re
import pickle
from collections import defaultdict

# === Step 1: Load the .pkl dataset ===
pkl_path = str(DATA_ROOT / "Intermediate/features_stress_fixed-current.pkl")  # Change this to your actual filename
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# If the pickle file contains a tuple, get the first item (usually a DataFrame)
df = data[0] if isinstance(data, tuple) else data

# === Step 2: Define category matching patterns ===
categories = {
    "Phone usage behavior": [r'APP', r'SCR', r'key-event'],
    "Social behavior": [r'CAE', r'MSG'],
    "Physical status": [r'ACT', r'FST', r'FDI', r'FCL', r'ACE', r'STP'],
    "Mobility": [r'LOC'],
    "Sleep": [r'Sleep']
}

# === Step 3: Classify columns ===
categorized = defaultdict(list)
uncategorized = []

for col in df.columns:
    matched = False
    for cat, patterns in categories.items():
        if any(re.search(pattern, col, re.IGNORECASE) for pattern in patterns):
            categorized[cat].append(col)
            matched = True
    if not matched:
        uncategorized.append(col)

# === Step 4: Create summary outputs ===
categorized_df = pd.DataFrame(
    [(col, cat) for cat, cols in categorized.items() for col in cols],
    columns=["Feature", "Category"]
)

uncategorized_df = pd.DataFrame(uncategorized, columns=["Feature"])

# === Step 5: Save or print results ===
# categorized_df.to_csv("categorized_features.csv", index=False)
# uncategorized_df.to_csv("uncategorized_features.csv", index=False)

print("Categorized features saved to 'categorized_features.csv'")
print("Uncategorized features saved to 'uncategorized_features.csv'")

categorized_df['Category'].value_counts().sort_values(ascending=False)

profiles = {
    category: categorized_df.loc[categorized_df['Category'] == category, 'Feature'].tolist()
    for category in categorized_df['Category'].unique()
}


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

# Normalization mode: 'user' or 'feature'
# If NORMALIZATION_MODE is set above, use it; otherwise fall back to env
NORMALIZATION_MODE = str(globals().get('NORMALIZATION_MODE', '')).strip().lower() or os.getenv('NORMALIZATION_MODE', 'feature').strip().lower()

if NORMALIZATION_MODE not in {'user', 'feature'}:
    NORMALIZATION_MODE = 'feature'

def normalize_feature_wise(df):
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    exclude_cols = ['Label']
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
    exclude_cols = ['Label']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    for user_id, group_data in df_norm.groupby('User'):
        if len(group_data) > 1:
            means = group_data[numeric_cols].mean(axis=0)
            stds = group_data[numeric_cols].std(axis=0).replace(0, np.nan)
            scaled = (group_data[numeric_cols] - means) / stds
            df_norm.loc[group_data.index, numeric_cols] = scaled.fillna(0.0)
    return df_norm

if NORMALIZATION_MODE == 'user':
    print('Applying user-wise normalization...')
    df_merged = normalize_user_wise(df_merged)
else:
    print('Applying feature-wise normalization...')
    df_merged = normalize_feature_wise(df_merged)

unique_users = df_merged['User'].unique()


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

chi_results = []
all_profile_data = {}
# -----------------------------
# Iterate over each profile and perform analysis
# -----------------------------
for profile_name, feature_list in profiles.items():
    print(f"\nProcessing profile: {profile_name}")
    
    # Create an empty list to collect discretized data for all users for the current profile
    list_df = []
    
    # Discretize the profile for each selected user (user-specific multivariate binarization)
    for user in unique_users:
        df_user = df_merged[df_merged['User'] == user].copy()
        df_user_disc = discretize_df_multivariate(df_user, feature_list, user, profile_name)
        # Reset the index to keep 'pid'
        df_user_disc = df_user_disc.reset_index()  # keep 'pid' as a column
        list_df.append(df_user_disc)
    
    # Concatenate the discretized data for all selected users
    df_profile = pd.concat(list_df, ignore_index=True)
    all_profile_data[profile_name] = df_profile.copy()
    
    # The new column name for the profile
    profile_col = f"{profile_name}_profile"
    
    # Group by user (pid), the profile category, and the target label
    label_counts = df_profile.groupby(['pid', profile_col, 'Label']).size().reset_index(name='count')
    
    # Pivot so that each row corresponds to a user and a profile bin, with columns for each label value.
    label_pivot = label_counts.pivot_table(index=['pid', profile_col], columns='Label', values='count', fill_value=0)

    # Calculate percentages (row-wise)
    label_pivot_percent = label_pivot.div(label_pivot.sum(axis=1), axis=0) * 100
    label_pivot_percent = label_pivot_percent.reset_index()
    
    # Melt the dataframe for plotting
    label_melted = label_pivot_percent.melt(id_vars=['pid', profile_col], var_name='Label', value_name='Percentage')
    # ---------------------------
    # c. Chi-Square Tests: Record results in chi_results list
    # ---------------------------
    categories = sorted(label_melted[profile_col].unique())
    print(f"Chi-Square Test results for profile: {profile_name}")
    for cat in categories:
        df_cat_counts = df_profile[df_profile[profile_col] == cat]
        contingency = pd.crosstab(df_cat_counts['User'], df_cat_counts['Label'])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(contingency)
            significance = "SIGNIFICANT" if p < 0.05 else "Not Significant"
            print(f"  Category {cat}: chi2 = {chi2:.2f}, p-value = {p:.2e}, dof = {dof} --> {significance}")
            chi_results.append({
                "Profile": profile_name,
                "Category": cat,
                "Chi2": round(chi2, 2),
                "p-value": f"{p:.2e}",
                "dof": dof,
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
                "Significance": "Not enough data"
            })
def get_sample_size(profile_name, category):
    if profile_name in all_profile_data:
        df_profile = all_profile_data[profile_name]
        profile_col = f"{profile_name}_profile"
        if profile_col in df_profile.columns:
            subset = df_profile[df_profile[profile_col] == category]
            return len(subset)
    return 0
chi_results_df = pd.DataFrame(chi_results)
chi_results_df['Sample_Size'] = chi_results_df.apply(
    lambda row: get_sample_size(row['Profile'], row['Category']), axis=1
)

chi_results_df = pd.DataFrame(chi_results)

# Add Sample Size column
chi_results_df['Sample_Size'] = chi_results_df.apply(
   lambda row: get_sample_size(row['Profile'], row['Category']), axis=1
)

# Add Cohen's w column
chi_results_df['Cohens_w'] = chi_results_df.apply(
   lambda row: (chi_results_df.loc[chi_results_df.index == row.name, 'Chi2'].values[0] / 
                chi_results_df.loc[chi_results_df.index == row.name, 'Sample_Size'].values[0])**0.5 
               if chi_results_df.loc[chi_results_df.index == row.name, 'Sample_Size'].values[0] > 0 
               else 0, axis=1
)

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

