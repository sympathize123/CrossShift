#!/usr/bin/env python3
"""
Create merged PRAUC figure for any dataset.
Usage: python create_merged_prauc_figure_dataset.py <results_csv> <data_path> <dataset_name> <output_dir>
"""
import sys
import os
import pandas as pd
import numpy as np
import cloudpickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def main():
    if len(sys.argv) < 5:
        print("Usage: python create_merged_prauc_figure_dataset.py <results_csv> <data_path> <dataset_name> <output_dir>")
        sys.exit(1)
    
    results_csv = sys.argv[1]
    data_path = sys.argv[2]
    dataset_name = sys.argv[3]
    output_dir = sys.argv[4]
    
    os.chdir(output_dir)
    
    # Read results
    results_df = pd.read_csv(results_csv)
    
    # Get original user order
    with open(data_path, 'rb') as f:
        data = cloudpickle.load(f)
    
    # Handle different data formats (tuple or dict)
    if isinstance(data, dict):
        # Extract X, y, groups from dictionary
        X = None
        y = None
        groups = None
        
        # Try different possible keys for X
        for key in ['X', 'features', 'X_train', 'data']:
            if key in data:
                X = data[key]
                break
        
        # Try different possible keys for y
        for key in ['y', 'labels', 'y_train', 'target']:
            if key in data:
                y = data[key]
                break
        
        # Try different possible keys for groups
        for key in ['groups', 'pcode', 'user_id', 'users']:
            if key in data:
                groups = data[key]
                break
        
        # If not found, try to find by type
        if X is None:
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    X = value
                    break
        if y is None:
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series)) and len(value.shape) == 1:
                    y = value
                    break
        if groups is None:
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series, list)) and key.lower() in ['groups', 'pcode', 'user', 'users', 'user_id']:
                    groups = np.array(value) if not isinstance(value, np.ndarray) else value
                    break
        
        if X is None or y is None or groups is None:
            raise ValueError(f"Could not extract X, y, groups from dict. Keys: {list(data.keys())}")
        
        t = data.get('t', None)
        datetimes = data.get('datetimes', None) or data.get('datetime', None)
    elif isinstance(data, tuple):
        if len(data) >= 5:
            X, y, groups, t, datetimes = data[:5]
        elif len(data) >= 3:
            X, y, groups = data[:3]
            t, datetimes = None, None
        else:
            raise ValueError(f"Expected tuple with at least 3 elements, got {len(data)}")
    else:
        raise ValueError(f"Expected tuple or dict, got {type(data)}")
    
    # Get unique users in original order
    unique_groups = []
    seen = set()
    for g in groups:
        if g not in seen:
            unique_groups.append(g)
            seen.add(g)
    
    valid_users_set = set(results_df['User'].tolist())
    original_user_order = [g for g in unique_groups if g in valid_users_set]
    
    # Reorder results
    user_to_data = {row['User']: row for _, row in results_df.iterrows()}
    results_df_sorted = pd.DataFrame([user_to_data[user] for user in original_user_order])
    
    # Separate OOD and non-OOD
    ood_df = results_df[results_df['IsOOD'] == True].copy()
    non_ood_df = results_df[results_df['IsOOD'] == False].copy()
    
    # Calculate means for PRAUC
    mean_prauc_all = results_df['PRAUC'].mean()
    std_prauc_all = results_df['PRAUC'].std()
    mean_prauc_ood = ood_df['PRAUC'].mean() if len(ood_df) > 0 else np.nan
    std_prauc_ood = ood_df['PRAUC'].std() if len(ood_df) > 0 else np.nan
    mean_prauc_non_ood = non_ood_df['PRAUC'].mean()
    std_prauc_non_ood = non_ood_df['PRAUC'].std()
    
    n_all = len(results_df)
    n_ood = len(ood_df)
    n_non_ood = len(non_ood_df)
    
    print(f"{dataset_name} - All users: {n_all}, OOD: {n_ood}, Non-OOD: {n_non_ood}")
    print(f"Mean PRAUC - All: {mean_prauc_all:.4f}, OOD: {mean_prauc_ood:.4f}, Non-OOD: {mean_prauc_non_ood:.4f}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    colors = ['red' if is_ood else 'blue' for is_ood in results_df_sorted['IsOOD']]
    ax.scatter(range(len(results_df_sorted)), results_df_sorted['PRAUC'], 
               c=colors, alpha=0.6, s=100, zorder=3)
    
    # Add mean lines
    if n_non_ood > 0 and not np.isnan(mean_prauc_non_ood):
        ax.axhline(y=mean_prauc_non_ood, color='green', linestyle='--', linewidth=2,
                   label=f'Mean (Non-OOD): {mean_prauc_non_ood:.4f}', zorder=2)
    
    if not np.isnan(mean_prauc_all):
        ax.axhline(y=mean_prauc_all, color='orange', linestyle='--', linewidth=2,
                   label=f'Mean (All): {mean_prauc_all:.4f}', zorder=1)
    
    if n_ood > 0 and not np.isnan(mean_prauc_ood):
        ax.axhline(y=mean_prauc_ood, color='darkred', linestyle='--', linewidth=2.5, 
                   label=f'Mean (OOD): {mean_prauc_ood:.4f}', zorder=3)
    
    # Formatting
    ax.set_xlabel('User Index', fontsize=12)
    ax.set_ylabel('PRAUC', fontsize=12)
    ax.set_title(f'PRAUC: All Users vs Non-OOD Only (LOSO) - {dataset_name}', fontsize=14)
    # Grid removed for cleaner appearance
    
    # Set y-axis range
    prauc_min = results_df_sorted['PRAUC'].min()
    prauc_max = results_df_sorted['PRAUC'].max()
    y_range = prauc_max - prauc_min
    y_padding = y_range * 0.1 if y_range > 0 else 0.1
    ax.set_ylim([max(0, prauc_min - y_padding), min(1, prauc_max + y_padding)])
    
    # Set x-axis labels
    ax.set_xticks(range(len(results_df_sorted)))
    ax.set_xticklabels([u.replace('P', '') for u in results_df_sorted['User']], 
                       rotation=45, ha='right', fontsize=8)
    
    # Legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Non-OOD users'),
        Patch(facecolor='red', alpha=0.6, label='OOD users')
    ]
    if n_non_ood > 0 and not np.isnan(mean_prauc_non_ood):
        legend_elements.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2,
                                      label=f'Mean (Non-OOD): {mean_prauc_non_ood:.4f}'))
    if not np.isnan(mean_prauc_all):
        legend_elements.append(Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                                      label=f'Mean (All): {mean_prauc_all:.4f}'))
    if n_ood > 0 and not np.isnan(mean_prauc_ood):
        legend_elements.append(Line2D([0], [0], color='darkred', linestyle='--', linewidth=2.5,
                                      label=f'Mean (OOD): {mean_prauc_ood:.4f}'))
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    output_file = f'ood_detection_loso_prauc_merged_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved: {output_file}")

if __name__ == '__main__':
    main()
