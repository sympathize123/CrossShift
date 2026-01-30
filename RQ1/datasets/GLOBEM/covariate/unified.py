#!/usr/bin/env python3
"""
unified.py - Covariate Shift Analysis
------------------------------------
Unified analysis script for covariate shift detection.
Analyzes all data together across all data types.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova, permdisp
from skbio.stats.ordination import pcoa
from sklearn.preprocessing import StandardScaler
import warnings
import json
import pickle
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "GLOBEM"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))

warnings.filterwarnings("ignore")

# =============================================================
# Covariate Shift Analysis - Unified
# -------------------------------------------------------------
# This script performs comprehensive covariate shift analysis
# across all data types for all data together.
# =============================================================

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE_DIR = Path(os.getenv("INTERMEDIATE_DIR", PROJECT_ROOT / "Intermediate"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "Results"))

# ----------------------------
# Configuration
# ----------------------------
# Normalization mode: "user" (user-wise), "feature" (feature-wise), or "none"

def env_flag(name: str, default: bool) -> bool:
    """Read boolean flag from environment variables."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def env_choice(name: str, default: str, allowed: set) -> str:
    """Read a string choice from environment variables."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    return val if val in allowed else default

# Backward-compatible: if NORMALIZATION_MODE is unset, fall back to USE_NORMALIZATION.
USE_NORMALIZATION = env_flag("USE_NORMALIZATION", True)
NORMALIZATION_MODE = env_choice("NORMALIZATION_MODE", "feature", {"feature", "user", "none"})
if os.getenv("NORMALIZATION_MODE") is None:
    NORMALIZATION_MODE = "feature" if USE_NORMALIZATION else "none"

# Data types to analyze
DATA_TYPES = {
    'phone_usage': ['phone_usage', 'keyevent', 'APP', 'SCR'],
    'mobility': ['mobility', 'LOC'],
    'physical_status': ['physical', 'ACT', 'FCL', 'FAC', 'FDI', 'FST', 'FitbitHeartrate', 'FitbitStepcount', 'Fitbitcalorie', 'Fitbitdistance'],
    'sleep': ['sleep'],
    'social_behavior': ['social', 'CALL', 'MSG']
}

# ----------------------------
# Helper Functions
# ----------------------------
def robust_groups_from_df(df: pd.DataFrame) -> np.ndarray:
    """Get group labels vector robustly (supports 'pcode' or 'pid')."""
    for col in ('pcode', 'pid'):
        if col in df.columns:
            return df[col].to_numpy()
    if isinstance(df.index, pd.MultiIndex):
        for level in ('pcode', 'pid'):
            if level in df.index.names:
                return df.index.get_level_values(level).to_numpy()
    raise KeyError("Could not find participant identifier ('pcode' or 'pid').")

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

def normalize_data_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize data per feature (column-wise), preserving pcode + label columns."""
    df_normalized = df.copy()

    # numeric columns only
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns

    # exclude label columns
    exclude_cols = ['stress_binary_personal', 'next_hour_binary_personal', 'pcode', 'pid', 'date', 'device_type', 'stress_binary', 'pss4_EMA']
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

def normalize_data_user_wise(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize data within each user group using StandardScaler."""
    df_normalized = df.copy()
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
    exclude_cols = ['stress_binary_personal', 'next_hour_binary_personal', 'pcode', 'pid', 'date', 'device_type', 'stress_binary', 'pss4_EMA']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if not numeric_cols:
        return df_normalized

    col_idx = [df_normalized.columns.get_loc(c) for c in numeric_cols]
    groups = robust_groups_from_df(df_normalized)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        scaler = StandardScaler()
        vals = df_normalized.iloc[idx, col_idx].to_numpy(dtype=np.float64, copy=False)
        df_normalized.iloc[idx, col_idx] = scaler.fit_transform(vals)

    return df_normalized


def ensure_dir(path):
    """Create directory and all parent directories if they don't exist."""
    os.makedirs(path, exist_ok=True)

def list_intermediate_datasets():
    """Return sorted list of CSV datasets in the Intermediate directory."""
    return sorted(p for p in INTERMEDIATE_DIR.glob("*.csv") if p.is_file())

def get_output_paths(dataset_label: str):
    """Create organized output directory structure and return paths."""
    base_dir = str(RESULTS_DIR)
    ensure_dir(base_dir)
    rq1_dir = os.path.join(base_dir, "RQ1")
    covariate_dir = os.path.join(rq1_dir, "CovariateShift", dataset_label)
    ensure_dir(covariate_dir)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'base': base_dir,
        'rq1': rq1_dir,
        'covariate': covariate_dir,
        'timestamp': timestamp
    }

def run_analysis_for_data_type(df_hourly_interpretable, data_type, feature_patterns, output_paths, analysis_type, dataset_label):
    """Run complete analysis for a specific data type."""
    print(f"\n{'='*80}")
    print(f"ANALYZING DATA TYPE: {data_type.upper()}")
    print(f"{'='*80}")
    
    # Extract features
    feature_keys_available = extract_features_by_type(df_hourly_interpretable, feature_patterns)
    
    if not feature_keys_available:
        print(f"No {data_type} features found. Skipping...")
        return None
    
    print(f"Detected {len(feature_keys_available)} {data_type} features:")
    for i, feature in enumerate(feature_keys_available, 1):
        print(f"  {i}. {feature}")
    
    # Apply social behavior zero filtering if this is social behavior data
    if data_type == 'social_behavior':
        print(f"\nApplying social behavior zero filtering...")
        df_hourly_interpretable = filter_social_behavior_zeros(df_hourly_interpretable, feature_keys_available)
        
        # Re-extract features after filtering
        feature_keys_available = extract_features_by_type(df_hourly_interpretable, feature_patterns)
        if not feature_keys_available:
            print(f"No social behavior features remaining after zero filtering. Skipping...")
            return None
    
    # Extract feature matrix and align groups
    X = df_hourly_interpretable[feature_keys_available]
    groups = robust_groups_from_df(df_hourly_interpretable)
    
    # Drop rows with any NaNs in X (and align groups)
    mask = np.all(~np.isnan(X.to_numpy(dtype=float, copy=False)), axis=1)
    X = X.loc[mask]
    groups = groups[mask]

    # Drop groups with <2 rows or zero within-group dispersion
    bad_groups = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        subgroup = X.iloc[idx].to_numpy(dtype=np.float64, copy=False)
        if len(subgroup) < 2:
            bad_groups.append(g)
            continue
        if np.allclose(subgroup - subgroup.mean(axis=0), 0):
            bad_groups.append(g)
    if bad_groups:
        print(f"Dropping groups with insufficient/zero dispersion: {bad_groups}")
        keep_mask = ~np.isin(groups, bad_groups)
        X = X.loc[keep_mask]
        groups = groups[keep_mask]
    if len(groups) == 0:
        print("No data remaining after dropping problematic groups. Skipping...")
        return None
    
    # Basic sanity checks
    N = X.shape[0]
    unique_groups, counts = np.unique(groups, return_counts=True)
    k = unique_groups.size
    if k < 2:
        print(f"Need at least 2 groups for PERMANOVA/PERMDISP; got {k}. Skipping...")
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
            print(f"After filtering, need at least 2 groups for PERMANOVA/PERMDISP; got {k}. Skipping...")
            return None
    
    print(f"Data summary: N={N}, k={k}, groups={unique_groups.tolist()}, group sizes={counts.tolist()}")
    
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
            'dataset': dataset_label,
            'data_type': data_type,
            'analysis_type': analysis_type,
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
    export_filename = f"{dataset_label}_analysis_data_{data_type}_{analysis_type}_{output_paths['timestamp']}.pkl"
    export_path = os.path.join(output_paths['covariate'], export_filename)
    
    with open(export_path, 'wb') as f:
        pickle.dump(analysis_data, f)
    
    # Save JSON version for easier inspection
    json_filename = f"{dataset_label}_analysis_data_{data_type}_{analysis_type}_{output_paths['timestamp']}.json"
    json_path = os.path.join(output_paths['covariate'], json_filename)
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = analysis_data.copy()
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"[EXPORT] Analysis data saved for dataset '{dataset_label}':")
    print(f"  - Pickle: {export_filename}")
    print(f"  - JSON: {json_filename}")
    print(f"  - Path: {output_paths['covariate']}")
    
    return analysis_data

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    analysis_type = {
        "user": "user_normalized",
        "feature": "feature_normalized",
        "none": "non_normalized",
    }.get(NORMALIZATION_MODE, "feature_normalized")
    dataset_paths = [
        Path(str(DATA_ROOT / "Intermediate/dep--INS-W_merged_cat_processed.csv"))
    ]

    print("Covariate Shift Analysis - Unified")
    print(f"Analysis type: {analysis_type.upper()}")
    print(f"Intermediate directory: {INTERMEDIATE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Datasets discovered: {len(dataset_paths)}")

    overall_summaries = {}

    for dataset_path in dataset_paths:
        dataset_label = dataset_path.stem

        print(f"\n{'#'*80}")
        print(f"DATASET: {dataset_label}")
        print(f"{'#'*80}")

        # Setup output directories for this dataset
        output_paths = get_output_paths(dataset_label)
        print(f"Output directory: {output_paths['covariate']}")
        print(f"Timestamp: {output_paths['timestamp']}")
        print(f"Loading and preprocessing data from: {dataset_path}")

        df_hourly_interpretable = pd.read_csv(dataset_path)
        df_hourly_interpretable = df_hourly_interpretable.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], errors='ignore')
        
        if not any(col in df_hourly_interpretable.columns for col in ('pcode', 'pid')):
            raise KeyError(f"Input CSV '{dataset_path.name}' must contain a 'pcode' or 'pid' column.")

        # Apply normalization based on mode
        if NORMALIZATION_MODE == "user":
            print("Applying user-wise StandardScaler normalization...")
            df_hourly_interpretable = normalize_data_user_wise(df_hourly_interpretable)
        elif NORMALIZATION_MODE == "feature":
            print("Applying per-feature min-max normalization...")
            df_hourly_interpretable = normalize_data_per_feature(df_hourly_interpretable)
        else:
            print("Using non-normalized data...")

        # Run analysis for each data type
        all_analysis_data = {}
        for data_type, feature_patterns in DATA_TYPES.items():
            analysis_data = run_analysis_for_data_type(df_hourly_interpretable, data_type, feature_patterns, output_paths, analysis_type, dataset_label)
            if analysis_data:
                all_analysis_data[data_type] = analysis_data

        # Save combined summary for this dataset
        summary_filename = f"{dataset_label}_analysis_summary_{analysis_type}_{output_paths['timestamp']}.json"
        summary_path = os.path.join(output_paths['covariate'], summary_filename)

        summary_data = {
            'metadata': {
                'dataset': dataset_label,
                'analysis_type': analysis_type,
                'timestamp': output_paths['timestamp'],
                'data_types_analyzed': list(all_analysis_data.keys()),
                'total_data_types': len(all_analysis_data),
                'shift_type': 'covariate'
            },
            'data_type_summaries': {}
        }

        for data_type, analysis_data in all_analysis_data.items():
            summary_data['data_type_summaries'][data_type] = {
                'N': analysis_data['metadata']['N'],
                'k': analysis_data['metadata']['k'],
                'n_features': analysis_data['metadata']['n_features'],
                'permanova_r2': analysis_data['statistical_results']['permanova']['r2'],
                'permanova_p_value': analysis_data['statistical_results']['permanova']['p_value'],
                'permdisp_r2': analysis_data['statistical_results']['permdisp']['r2'],
                'permdisp_p_value': analysis_data['statistical_results']['permdisp']['p_value']
            }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        overall_summaries[dataset_label] = summary_data

        print(f"\n{'='*80}")
        print(f"COVARIATE SHIFT ANALYSIS COMPLETE FOR {dataset_label}")
        print(f"{'='*80}")
        print(f"Analysis type: {analysis_type.upper()}")
        print(f"Shift type: COVARIATE")
        print(f"Data types processed: {len(all_analysis_data)}")
        print(f"Output directory: {output_paths['covariate']}")
        print(f"Summary file: {summary_filename}")
        print(f"Individual data files saved for each data type")
        print(f"{'='*80}")

    print(f"\nProcessed {len(overall_summaries)} dataset(s): {list(overall_summaries.keys())}")
