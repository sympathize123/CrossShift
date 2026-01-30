#!/usr/bin/env python3
"""
unified.py - Conditional Shift Analysis
--------------------------------------
Unified analysis script for conditional shift detection.
Analyzes data separately for each label (0 and 1) across all data types.
"""
import os, sys
from Funcs.Utility import *
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova, permdisp
from skbio.stats.ordination import pcoa
from sklearn.preprocessing import StandardScaler
import warnings
import json
import pickle
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "CrossCheck"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))

warnings.filterwarnings("ignore")

# =============================================================
# Conditional Shift Analysis - Unified
# -------------------------------------------------------------
# This script performs comprehensive conditional shift analysis
# across all data types for each label separately.
# =============================================================

# ----------------------------
# Configuration
# ----------------------------
# Set to True for normalized analysis, False for non-normalized
USE_NORMALIZATION = True

# Data types to analyze
DATA_TYPES = {
    'phone_usage': ['SCR_DUR', 'SCR_CNT'],
    'mobility': ['LOC_DST', 'LOC_CNT', 'LOC_DST_PER_PLACE'],
    'physical_status': ['ACT_WALKING', 
        'ACT_RUNNING', 
        'ACT_STILL', 
        'ACT_IN_VEHICLE', 
        'ACT_ON_BIKE', 
        'ACT_TILTING', 
        'ACT_FOOT', 
        'ACT_UNKNOWN'],
    'sleep': ['SLP_DUR', 'SLP_START', 'SLP_END'],
    'social_behavior': ['CAL_IN_CNT', 
        'CAL_IN_DUR', 
        'CAL_OUT_CNT', 
        'CAL_OUT_DUR', 
        'CAL_MISS_CNT', 
        'MSG_IN_CNT', 
        'MSG_OUT_CNT', 
        'CON_DUR', 
        'CON_CNT', 
        'CON_VCN']
}
# ----------------------------
# Helper Functions
# ----------------------------
def robust_groups_from_df(df: pd.DataFrame) -> np.ndarray:
    """Get group labels vector 'pcode' robustly (column or index level)."""
    if 'pcode' in df.columns:
        return df['pcode'].to_numpy()
    if isinstance(df.index, pd.MultiIndex) and 'pcode' in df.index.names:
        return df.index.get_level_values('pcode').to_numpy()
    raise KeyError("Could not find 'pcode' as a column or index level.")

def extract_features_by_type(df: pd.DataFrame, feature_patterns: list) -> list:
    """Extract features based on patterns."""
    available_features = df.columns.tolist()
    extracted_features = []
    
    available_features_upper = [f.upper() for f in available_features]
    
    for pattern in feature_patterns:
        pattern_upper = pattern.upper()
        for i, feature in enumerate(available_features_upper):
            if pattern_upper in feature:
                original_feature = available_features[i]
                if original_feature not in extracted_features:
                    extracted_features.append(original_feature)
    
    return extracted_features

def extract_F(res) -> float:
    """Extract F from skbio result."""
    if hasattr(res, "to_series"):
        s = res.to_series()
        for k in s.index:
            if str(k).lower() in ("test statistic", "statistic", "f"):
                return float(s[k])
    for key in ("test statistic", "statistic", "F", "f-statistic"):
        try:
            return float(res[key])
        except Exception:
            pass
    for attr in ("test_statistic", "statistic", "F"):
        if hasattr(res, attr):
            return float(getattr(res, attr))
    raise KeyError("Could not extract F from result.")

def effect_sizes_oneway(F: float, N: int, k: int):
    """Return (pseudo_R2, eta2, df1, df2) for one-way designs."""
    df1 = k - 1
    df2 = N - k
    if df1 <= 0 or df2 <= 0:
        return np.nan, np.nan, df1, df2
    val = (F * df1) / (F * df1 + df2)
    return val, val, df1, df2

def derive_r2_from_f(F: float, N: int, k: int):
    """Derive R² from F value."""
    df1 = k - 1
    df2 = N - k
    if df1 <= 0 or df2 <= 0:
        return np.nan, df1, df2
    r2 = (F * df1) / (F * df1 + df2)
    return r2, df1, df2

def filter_social_behavior_zeros(df: pd.DataFrame, social_features: list) -> pd.DataFrame:
    """Filter out samples where all social behavior features are zero."""
    if not social_features:
        return df
    
    # Fill NaN values with 0 for social features
    social_data = df[social_features].fillna(0)
    
    # Drop rows where ALL social features are zero
    all_zero_mask = (social_data == 0).all(axis=1)
    df_filtered = df[~all_zero_mask].copy()
    
    dropped_count = all_zero_mask.sum()
    print(f"Social behavior zero filtering: dropped {dropped_count} samples with all social features = 0")
    print(f"Original samples: {len(df)}, Filtered samples: {len(df_filtered)}")
    print(f"Retention rate: {(len(df_filtered) / len(df)) * 100:.2f}%")
    
    return df_filtered

def normalize_data_within_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data within each pcode group using StandardScaler."""
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude label columns from normalization
    exclude_cols = ['stress_binary_personal', 'stress_binary_overall', 'pcode', 'day', 'date']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for group_name, group_data in df.groupby('pcode'):
        if len(group_data) > 1:
            scaler = StandardScaler()
            group_normalized = scaler.fit_transform(group_data[numeric_cols])
            df_normalized.loc[group_data.index, numeric_cols] = group_normalized
    
    return df_normalized

def normalize_data_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize data per feature (column-wise), preserving pcode + label columns."""
    df_normalized = df.copy()

    # numeric columns only
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns

    # exclude label columns
    exclude_cols = ['stress_binary_personal', 'label', 'pcode', 'day', 'date']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    # min-max normalize per column
    for col in numeric_cols:
        vals = pd.to_numeric(df_normalized[col], errors='coerce')
        minv = vals.min(skipna=True)
        maxv = vals.max(skipna=True)

        if pd.isna(minv) or pd.isna(maxv):
            # all-NaN column -> leave as-is
            df_normalized[col] = vals
        elif maxv == minv:
            # constant column -> set to 0.0
            df_normalized[col] = 0.0
        else:
            df_normalized[col] = (vals - minv) / (maxv - minv)

    return df_normalized

def ensure_dir(path):
    """Create directory and all parent directories if they don't exist."""
    os.makedirs(path, exist_ok=True)

def get_output_paths():
    """Create organized output directory structure and return paths."""
    PATH_RESULTS= str(RESULTS_ROOT)
    base_dir = PATH_RESULTS
    ensure_dir(base_dir)
    rq1_dir = os.path.join(base_dir, "RQ1")
    conditional_dir = os.path.join(rq1_dir, "ConditionalShift")
    ensure_dir(conditional_dir)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'base': base_dir,
        'rq1': rq1_dir,
        'conditional': conditional_dir,
        'timestamp': timestamp
    }

def run_analysis_for_label(df_filtered, data_type, feature_patterns, output_paths, analysis_type, label_status):
    """Run complete analysis for a specific label status."""
    print(f"\n{'='*60}")
    print(f"ANALYZING DATA TYPE: {data_type.upper()} - LABEL {label_status}")
    print(f"{'='*60}")
    
    # Extract features
    feature_keys_available = extract_features_by_type(df_filtered, feature_patterns)
    
    if not feature_keys_available:
        print(f"No {data_type} features found for label {label_status}. Skipping...")
        return None
    
    print(f"Detected {len(feature_keys_available)} {data_type} features for label {label_status}:")
    for i, feature in enumerate(feature_keys_available, 1):
        print(f"  {i}. {feature}")
    
    # Apply social behavior zero filtering if this is social behavior data
    if data_type == 'social_behavior':
        print(f"\nApplying social behavior zero filtering for label {label_status}...")
        df_filtered = filter_social_behavior_zeros(df_filtered, feature_keys_available)
        
        # Re-extract features after filtering
        feature_keys_available = extract_features_by_type(df_filtered, feature_patterns)
        if not feature_keys_available:
            print(f"No social behavior features remaining after zero filtering for label {label_status}. Skipping...")
            return None
    
    # Extract feature matrix and align groups
    X = df_filtered[feature_keys_available]
    groups = robust_groups_from_df(df_filtered)
    
    # Drop rows with any NaNs in X (and align groups)
    mask = np.all(~np.isnan(X.to_numpy(dtype=float, copy=False)), axis=1)
    X = X.loc[mask]
    groups = groups[mask]

    # MAX_PER_GROUP = 50  # try 20, 50, 100

    # gser = pd.Series(groups, index=X.index)  # group labels aligned to X.index

    # idx = (
    #     gser.groupby(gser, group_keys=False)
    #         .apply(lambda s: s.sample(n=min(len(s), MAX_PER_GROUP), random_state=42))
    #         .index
    # )

    # X = X.loc[idx]
    # groups = gser.loc[idx].to_numpy()
    
    # Basic sanity checks
    N = X.shape[0]
    unique_groups, counts = np.unique(groups, return_counts=True)
    k = unique_groups.size
    if k < 2:
        print(f"Need at least 2 groups for PERMANOVA/PERMDISP; got {k} for label {label_status}. Skipping...")
        return None
    
    # Filter out groups with insufficient data (less than 2 samples)
    min_group_size = np.min(counts)
    if min_group_size < 2:
        print(f"Filtering out groups with insufficient data (min group size: {min_group_size})")
        # Find groups with sufficient data
        sufficient_groups = unique_groups[counts >= 2]
        print(f"Keeping groups with sufficient data: {sufficient_groups.tolist()}")
        
        # Filter data to keep only groups with sufficient samples
        mask = np.isin(groups, sufficient_groups)
        X = X.loc[mask]
        groups = groups[mask]
        
        # Recalculate after filtering
        N = X.shape[0]
        unique_groups, counts = np.unique(groups, return_counts=True)
        k = unique_groups.size
        
        if k < 2:
            print(f"After filtering, need at least 2 groups for PERMANOVA/PERMDISP; got {k} for label {label_status}. Skipping...")
            return None
    
    print(f"Groups for label {label_status}: {groups}")
    print(f"Unique groups: {unique_groups}, counts: {counts}")
    print(f"Data summary for label {label_status}: N={N}, k={k}, groups={unique_groups.tolist()}, group sizes={counts.tolist()}")
    
    
    # Distance matrix computation
    print("Computing distance matrix...")
    ids = [f"s{i}" for i in range(N)]
    Xv = X.to_numpy(dtype=np.float64, copy=False)
    D = squareform(pdist(Xv, metric='euclidean'))
    distance_matrix = DistanceMatrix(D, ids=ids)
    
    # PERMANOVA
    print("Running PERMANOVA...")
    perms_permanova = int(os.getenv('PERMS_PERMANOVA', '100'))
    permanova_result = permanova(distance_matrix, groups, permutations=perms_permanova, seed=42)
    
    F_permanova = extract_F(permanova_result)
    pseudo_R2, _, df1_permanova, df2_permanova = effect_sizes_oneway(F_permanova, N, k)
    r2_permanova, _, _ = derive_r2_from_f(F_permanova, N, k)
    
    # PERMDISP with error handling
    print("Running PERMDISP...")
    perms_permdisp = int(os.getenv('PERMS_PERMDISP', '100'))
    
    try:
        permdisp_result = permdisp(distance_matrix, groups, permutations=perms_permdisp, seed=42)
        F_permdisp = extract_F(permdisp_result)
        _, eta2, df1_permdisp, df2_permdisp = effect_sizes_oneway(F_permdisp, N, k)
        r2_permdisp, _, _ = derive_r2_from_f(F_permdisp, N, k)
        permdisp_p_value = float(permdisp_result['p-value'])
    except (ZeroDivisionError, ValueError, RuntimeError) as e:
        print(f"Warning: PERMDISP failed with error: {e}. Setting results to NaN.")
        F_permdisp = np.nan
        r2_permdisp = np.nan
        permdisp_p_value = np.nan
    
    # PCoA computation
    print("Computing PCoA...")
    pcoa_res = pcoa(distance_matrix)
    coords_full = pcoa_res.samples
    coords_2d = coords_full.iloc[:, :2].copy()
    coords_2d.columns = ["PC1", "PC2"]
    
    coords_full['group'] = groups
    coords_2d['group'] = groups
    
    # Compute centroids
    uniq_groups = coords_2d['group'].unique()
    centroids_data = {}
    for g in uniq_groups:
        group_data = coords_2d[coords_2d['group'] == g]
        centroids_data[g] = {
            'PC1_mean': float(group_data['PC1'].mean()),
            'PC2_mean': float(group_data['PC2'].mean()),
            'PC1_std': float(group_data['PC1'].std()),
            'PC2_std': float(group_data['PC2'].std()),
            'n_samples': len(group_data)
        }
    
    # Compute distances to centroids
    coord_cols = [c for c in coords_full.columns if c != 'group']
    arr = coords_full[coord_cols].to_numpy(dtype=np.float64, copy=False)
    g = coords_full['group'].to_numpy()
    
    codes, levels = pd.factorize(g)
    centroids = np.vstack([arr[codes == i].mean(axis=0) for i in range(len(levels))])
    dists = np.linalg.norm(arr - centroids[codes], axis=1)
    
    # Create streamlined analysis data export
    analysis_data = {
        'metadata': {
            'data_type': data_type,
            'analysis_type': analysis_type,
            'label_status': label_status,
            'timestamp': output_paths['timestamp'],
            'N': N,
            'k': k,
            'groups': unique_groups.tolist(),
            'group_counts': counts.tolist(),
            'features_used': feature_keys_available,
            'n_features': len(feature_keys_available)
        },
        'statistical_results': {
            'permanova': {
                'r2': float(r2_permanova) if not np.isnan(r2_permanova) else None,
                'p_value': float(permanova_result['p-value']),
                'F_statistic': float(F_permanova) if not np.isnan(F_permanova) else None,
                'permutations': perms_permanova
            },
            'permdisp': {
                'r2': float(r2_permdisp) if not np.isnan(r2_permdisp) else None,
                'p_value': float(permdisp_p_value) if not np.isnan(permdisp_p_value) else None,
                'F_statistic': float(F_permdisp) if not np.isnan(F_permdisp) else None,
                'permutations': perms_permdisp
            }
        },
        'centroids': centroids_data,
        'dispersion_data': {
            'distances_to_centroids': dists.tolist(),
            'groups': groups.tolist()
        }
    }
    
    # Save comprehensive data export
    export_filename = f"analysis_data_{data_type}_{analysis_type}_label_{label_status}_{output_paths['timestamp']}.pkl"
    export_path = os.path.join(output_paths['conditional'], export_filename)
    
    with open(export_path, 'wb') as f:
        pickle.dump(analysis_data, f)
    
    # Save JSON version for easier inspection
    json_filename = f"analysis_data_{data_type}_{analysis_type}_label_{label_status}_{output_paths['timestamp']}.json"
    json_path = os.path.join(output_paths['conditional'], json_filename)
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = analysis_data.copy()
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"[EXPORT] Analysis data saved for label {label_status}:")
    print(f"  - Pickle: {export_filename}")
    print(f"  - JSON: {json_filename}")
    print(f"  - Path: {output_paths['conditional']}")
    
    return analysis_data

def run_analysis_for_data_type(df_hourly_interpretable, data_type, feature_patterns, output_paths, analysis_type):
    """Run analysis for a specific data type across both labels."""
    print(f"\n{'='*80}")
    print(f"ANALYZING DATA TYPE: {data_type.upper()}")
    print(f"{'='*80}")
    
    # Get unique label values
    unique_labels = df_hourly_interpretable['stress_binary_personal'].unique()
    print(f"Found label values: {unique_labels}")
    
    # Run analysis for each label status
    label_results = {}
    for label_status in [0.0, 1.0]:  # Use float values to match the data
        if label_status in unique_labels:
            print(f"\n{'='*60}")
            print(f"PROCESSING LABEL STATUS: {label_status}")
            print(f"{'='*60}")
            
            # Filter data for current label
            df_filtered = df_hourly_interpretable[df_hourly_interpretable['stress_binary_personal'] == label_status].copy()
            print(f"Filtered data for label {label_status}: {len(df_filtered)} samples")
            
            # Run analysis for this label
            analysis_data = run_analysis_for_label(df_filtered, data_type, feature_patterns, output_paths, analysis_type, label_status)
            if analysis_data:
                label_results[label_status] = analysis_data
        else:
            print(f"Label {label_status} not found in dataset. Skipping...")
    
    return label_results

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Setup output directories
    output_paths = get_output_paths()
    analysis_type = "normalized" if USE_NORMALIZATION else "non_normalized"
    print(f"Conditional Shift Analysis - Unified")
    print(f"Analysis type: {analysis_type.upper()}")
    print(f"Output directory: {output_paths['conditional']}")
    print(f"Timestamp: {output_paths['timestamp']}")
    
    # Load data
    print("Loading and preprocessing data...")
    #Overfitting/CrossCheck/ema_f&l.csv
    df_hourly_interpretable = pd.read_csv(str(DATA_ROOT / "ema_f&l_full.csv"))
    
    # Drop rows with NaN in stress_binary_personal
    print(f"Original data shape: {df_hourly_interpretable.shape}")
    df_hourly_interpretable = df_hourly_interpretable.dropna(subset=['stress_binary_personal'])
    print(f"After dropping NaN in stress_binary_personal: {df_hourly_interpretable.shape}")
    df_hourly_interpretable = df_hourly_interpretable.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], errors='ignore')
    
    if 'pcode' not in df_hourly_interpretable.columns:
        raise KeyError("Input CSV must contain a 'pcode' column.")
    
    # Check for label column
    if 'stress_binary_personal' not in df_hourly_interpretable.columns:
        raise KeyError("Input CSV must contain a 'label' column for conditional shift analysis.")
    
    # Apply normalization if requested
    if USE_NORMALIZATION:
        print("Applying StandardScaler normalization within groups...")
        df_hourly_interpretable = normalize_data_within_groups(df_hourly_interpretable)
    else:
        print("Using non-normalized data...")
    
    # Run analysis for each data type
    all_analysis_data = {}
    for data_type, feature_patterns in DATA_TYPES.items():
        label_results = run_analysis_for_data_type(df_hourly_interpretable, data_type, feature_patterns, output_paths, analysis_type)
        if label_results:
            all_analysis_data[data_type] = label_results
    
    # Save combined summary
    summary_filename = f"analysis_summary_{analysis_type}_{output_paths['timestamp']}.json"
    summary_path = os.path.join(output_paths['conditional'], summary_filename)
    
    summary_data = {
        'metadata': {
            'analysis_type': analysis_type,
            'timestamp': output_paths['timestamp'],
            'data_types_analyzed': list(all_analysis_data.keys()),
            'total_data_types': len(all_analysis_data),
            'shift_type': 'conditional'
        },
        'data_type_summaries': {}
    }
    
    for data_type, label_results in all_analysis_data.items():
        summary_data['data_type_summaries'][data_type] = {}
        for label_status, analysis_data in label_results.items():
            summary_data['data_type_summaries'][data_type][f'label_{int(label_status)}'] = {
                'N': analysis_data['metadata']['N'],
                'k': analysis_data['metadata']['k'],
                'n_features': analysis_data['metadata']['n_features'],
                'permanova_r2': analysis_data['statistical_results']['permanova']['r2'],
                'permdisp_r2': analysis_data['statistical_results']['permdisp']['r2']
            }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save comprehensive comparison data
    comparison_filename = f"analysis_comparison_{analysis_type}_{output_paths['timestamp']}.json"
    comparison_path = os.path.join(output_paths['conditional'], comparison_filename)
    
    comparison_data = {
        'metadata': {
            'analysis_type': analysis_type,
            'timestamp': output_paths['timestamp'],
            'shift_type': 'conditional',
            'labels_analyzed': [0.0, 1.0]
        },
        'label_comparisons': {}
    }
    
    for data_type, label_results in all_analysis_data.items():
        comparison_data['label_comparisons'][data_type] = {}
        if 0.0 in label_results and 1.0 in label_results:
            # Compare labels 0.0 and 1.0
            label_0_data = label_results[0.0]
            label_1_data = label_results[1.0]
            
            comparison_data['label_comparisons'][data_type] = {
                'sample_size_comparison': {
                    'label_0_N': label_0_data['metadata']['N'],
                    'label_1_N': label_1_data['metadata']['N'],
                    'ratio': label_0_data['metadata']['N'] / label_1_data['metadata']['N'] if label_1_data['metadata']['N'] > 0 else float('inf')
                },
                 'statistical_comparison': {
                    'permanova_r2_diff': (label_0_data['statistical_results']['permanova']['r2'] - label_1_data['statistical_results']['permanova']['r2']) if (label_0_data['statistical_results']['permanova']['r2'] is not None and label_1_data['statistical_results']['permanova']['r2'] is not None) else None,
                    'permdisp_r2_diff': (label_0_data['statistical_results']['permdisp']['r2'] - label_1_data['statistical_results']['permdisp']['r2']) if (label_0_data['statistical_results']['permdisp']['r2'] is not None and label_1_data['statistical_results']['permdisp']['r2'] is not None) else None,
                    'permanova_f_ratio': label_0_data['statistical_results']['permanova']['F_statistic'] / label_1_data['statistical_results']['permanova']['F_statistic'] if (label_0_data['statistical_results']['permanova']['F_statistic'] is not None and label_1_data['statistical_results']['permanova']['F_statistic'] is not None and label_1_data['statistical_results']['permanova']['F_statistic'] > 0) else None,
                    'permdisp_f_ratio': label_0_data['statistical_results']['permdisp']['F_statistic'] / label_1_data['statistical_results']['permdisp']['F_statistic'] if (label_0_data['statistical_results']['permdisp']['F_statistic'] is not None and label_1_data['statistical_results']['permdisp']['F_statistic'] is not None and label_1_data['statistical_results']['permdisp']['F_statistic'] > 0) else None
                },
                'centroid_comparison': {
                    'label_0_centroids': label_0_data['centroids'],
                    'label_1_centroids': label_1_data['centroids'],
                    'centroid_distance': {
                        g: np.linalg.norm(np.array(label_0_data['centroids'][g]['PC1_mean'], dtype=float) - np.array(label_1_data['centroids'][g]['PC1_mean'], dtype=float)) for g in label_0_data['centroids']
                    }
                },      
                'feature_comparison': {
                    'label_0_features': label_0_data['metadata']['features_used'],
                    'label_1_features': label_1_data['metadata']['features_used'],
                    'common_features': list(set(label_0_data['metadata']['features_used']) & set(label_1_data['metadata']['features_used'])),
                    'unique_to_label_0': list(set(label_0_data['metadata']['features_used']) - set(label_1_data['metadata']['features_used'])),
                    'unique_to_label_1': list(set(label_1_data['metadata']['features_used']) - set(label_0_data['metadata']['features_used']))
                }
            }
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"CONDITIONAL SHIFT ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Analysis type: {analysis_type.upper()}")
    print(f"Shift type: CONDITIONAL")
    print(f"Data types processed: {len(all_analysis_data)}")
    print(f"Output directory: {output_paths['conditional']}")
    print(f"Summary file: {summary_filename}")
    print(f"Comparison file: {comparison_filename}")
    print(f"Individual data files saved for each data type and label")
    print(f"{'='*80}")
