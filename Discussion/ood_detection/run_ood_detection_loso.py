# -*- coding: utf-8 -*-
import os
import sys
import warnings
# Force unbuffered output for real-time logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from tqdm import tqdm
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.spatial.distance import jensenshannon

# Add Posthoc_Analysis to path (one level up from ood_detection) for utility import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Posthoc_Analysis'))
from utility import load

warnings.filterwarnings('ignore')

# Configuration
seed = 42
N_JOBS = -1
TOP_K = 40  # Number of features to select per fold
N_CLUSTERS = 5  # Number of clusters for concept shift
SHIFT_THRESHOLD_PERCENTILE = 75  # Percentile for threshold calibration
MAX_TEST_SAMPLES_FOR_GATE = None  # Set to e.g., 200 for early prediction scenarios

def concept_shift_from_trained_model(
    model, X_train, X_test,
    n_clusters=5, random_state=42, eps=1e-8, max_test_samples=None
):
    """
    Concept-shift proxy using a trained model.
    Assumes X_train/X_test already contain ONLY the selected features.
    - Fit KMeans on TRAIN X only
    - Use model.predict_proba on train/test
    - Compare cluster-conditional mean predicted probs via JS distance
    
    Args:
        model: Already trained XGBClassifier (trained on selected features)
        X_train: Training features (DataFrame with selected features only)
        X_test: Test features (DataFrame with selected features only)
        n_clusters: Number of clusters for KMeans
        random_state: Random seed
        eps: Small value for clipping
        max_test_samples: If set, only use first N test samples for shift computation
    
    Returns: JS distance in [0,1] (sqrt of JS divergence).
    """
    try:
        # Limit test samples for deployable realism (if specified)
        if max_test_samples is not None and len(X_test) > max_test_samples:
            X_test = X_test.iloc[:max_test_samples]

        if len(X_train) < 10 or len(X_test) < 10:
            return 1.0

        X_train_np = X_train.values
        X_test_np = X_test.values

        # KMeans on TRAIN only (no transductive leakage)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        c_train = kmeans.fit_predict(X_train_np)
        c_test = kmeans.predict(X_test_np)

        # Predictions on same feature space
        p_train = model.predict_proba(X_train_np)
        p_test = model.predict_proba(X_test_np)
        C = p_train.shape[1]

        # Global means (used for empty clusters)
        global_train = np.mean(p_train, axis=0)
        global_test = np.mean(p_test, axis=0)

        # Cluster-conditional mean probs
        mu_train = np.zeros((n_clusters, C))
        mu_test = np.zeros((n_clusters, C))

        for c in range(n_clusters):
            idx_tr = np.where(c_train == c)[0]
            idx_te = np.where(c_test == c)[0]
            mu_train[c] = np.mean(p_train[idx_tr], axis=0) if len(idx_tr) else global_train
            mu_test[c] = np.mean(p_test[idx_te], axis=0) if len(idx_te) else global_test

        # Flatten and JS distance
        v_train = np.clip(mu_train.flatten(), eps, 1.0)
        v_test = np.clip(mu_test.flatten(), eps, 1.0)
        v_train /= v_train.sum()
        v_test /= v_test.sum()

        js = jensenshannon(v_train, v_test)
        return 1.0 if (np.isnan(js) or np.isinf(js)) else js

    except Exception as e:
        print(f"Error computing concept shift: {e}")
        import traceback
        traceback.print_exc()
        return 1.0

# Data Loading and Preprocessing
print("Loading D-3 dataset...")
data_path = "/var/nfs_share/Overfitting/D-3/Intermediate/stress_binary_personal-full.pkl"
X, y, groups, t, datetimes = load(data_path)

# Remove PIF and ESM columns
pif_cols = [col for col in X.columns if "PIF" in col]
X = X.drop(columns=pif_cols)
X = X.drop(columns=[col for col in X.columns if col.startswith("ESM")])
user_ids = np.unique(groups)

print(f"Loaded data: {len(X)} samples from {len(user_ids)} users")
print(f"Features: {X.shape[1]}")

# Remove rows with NaN values
mask = X.notna().all(axis=1)
X = X.loc[mask].reset_index(drop=True)
y = y[mask]
groups = groups[mask]
if t is not None:
    t = t[mask]
if datetimes is not None:
    datetimes = datetimes[mask]

print(f"After removing NaNs: {len(X)} samples from {len(np.unique(groups))} users")

# Feature preprocessing
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
cat_zero_var_cols = [col for col in categorical_cols if X[col].nunique() <= 1]
categorical_cols = [col for col in categorical_cols if col not in cat_zero_var_cols]

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Prepare data for LOSO
print("\n=== Preparing LOSO Cross-Validation ===")
X_normalized = X.copy()

# User-wise normalization for numeric features
for user in np.unique(groups):
    user_mask = (groups == user)
    if np.sum(user_mask) > 0:
        scaler = StandardScaler()
        X_normalized.loc[user_mask, numeric_cols] = scaler.fit_transform(
            X.loc[user_mask, numeric_cols]
        )

print(f"Normalized data shape: {X_normalized.shape}")

# Filter users with sufficient data and class balance
valid_users = []
for user in np.unique(groups):
    user_mask = (groups == user)
    y_user = y[user_mask]
    if len(y_user) >= 20 and len(np.unique(y_user)) >= 2:
        valid_users.append(user)

print(f"Valid users for LOSO: {len(valid_users)}")

def train_and_evaluate_loso_with_ood_detection(
    X, y, groups, valid_users,
    use_ood_filtering=True, n_clusters=5, max_test_samples=None, random_state=42,
    top_k=40, shift_threshold_percentile=75
):
    """
    Perform LOSO cross-validation with per-fold OOD detection.
    
    For each fold:
    1. Feature selection: Select top-K features from TRAINING users only
    2. Threshold calibration: Compute shifts for all TRAINING users (nested LOSO), take percentile
    3. Model training: Train model on TRAINING users with selected features
    4. OOD detection: Compute shift for TEST user, compare to threshold
    5. Evaluation: If not OOD, evaluate on TEST user
    
    Args:
        X: Feature dataframe (all features, will be selected per-fold)
        y: Labels
        groups: User groups
        valid_users: List of valid users to include
        use_ood_filtering: If True, skip OOD users. If False, evaluate all users.
        n_clusters: Number of clusters for KMeans in shift computation
        max_test_samples: If set, only use first N test samples for shift computation
        random_state: Random seed
        top_k: Number of top features to select per fold
        shift_threshold_percentile: Percentile for threshold calibration (e.g., 75 for 75th percentile)
    
    Returns:
        Dictionary with results for each test user, and OOD statistics
    """
    loso = LeaveOneGroupOut()
    results = {}
    ood_stats = {'total': 0, 'ood_identified': 0}
    
    # Filter to valid users only
    valid_mask = np.isin(groups, valid_users)
    X_valid = X.loc[valid_mask].reset_index(drop=True)
    y_valid = y[valid_mask]
    groups_valid = groups[valid_mask]
    
    splits = list(loso.split(X_valid, y_valid, groups_valid))
    
    for train_idx, test_idx in tqdm(splits, desc="LOSO CV with per-fold feature selection and OOD detection"):
        test_user = groups_valid[test_idx[0]]
        ood_stats['total'] += 1
        
        # Get train and test data (all features)
        X_train_full = X_valid.iloc[train_idx]
        y_train = y_valid[train_idx]
        X_test_full = X_valid.iloc[test_idx]
        y_test = y_valid[test_idx]
        groups_train = groups_valid[train_idx]
        
        # Skip if insufficient data
        if len(X_train_full) < 10 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        try:
            # ============================================================
            # STEP 1: Feature Selection (TRAINING users only)
            # ============================================================
            fs_model = XGBClassifier(
                n_estimators=300,
                max_depth=3,
                eval_metric='auc',
                random_state=random_state,
                tree_method='hist',
                n_jobs=1  # Determinism
            )
            fs_model.fit(X_train_full, y_train)
            
            # Get feature importances
            importances = fs_model.feature_importances_
            feature_names = X_train_full.columns.to_numpy()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(top_k)['feature'].tolist()
            
            # Reduce feature space for this fold
            X_train = X_train_full[selected_features].copy()
            X_test = X_test_full[selected_features].copy()
            
            # Ensure numeric types for KMeans
            X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            
            # ============================================================
            # STEP 2: Train Model (TRAINING users, selected features)
            # ============================================================
            model = XGBClassifier(
                n_estimators=500,
                max_depth=3,
                eval_metric='auc',
                random_state=random_state,
                tree_method='hist',
                n_jobs=1  # Determinism
            )
            model.fit(X_train, y_train)
            
            # ============================================================
            # STEP 3: Threshold Calibration (TRAINING users only, using SAME model)
            # ============================================================
            # Compute shifts for all training users using the SAME fold model
            # NOTE: We compute shift against X_train_minus_user (not full X_train)
            # to make training-user shifts more comparable to held-out test-user shifts
            train_shifts = []
            train_users = np.unique(groups_train)
            
            for train_user in train_users:
                user_mask = (groups_train == train_user)
                if np.sum(user_mask) < 10:  # Skip if insufficient samples
                    continue
                
                X_train_user = X_train.loc[user_mask]
                X_train_minus_user = X_train.loc[~user_mask]
                
                # Skip if not enough samples after removing user
                if len(X_train_minus_user) < 10:
                    continue
                
                # Compute shift for this training user against remaining training users
                # This makes training-user shifts more comparable to test-user shifts
                js_shift_train = concept_shift_from_trained_model(
                    model, X_train_minus_user, X_train_user,
                    n_clusters=n_clusters, random_state=random_state,
                    max_test_samples=max_test_samples
                )
                
                if not np.isnan(js_shift_train):
                    train_shifts.append(js_shift_train)
            
            # Compute threshold from training user shifts
            if len(train_shifts) > 0:
                shift_threshold = np.percentile(train_shifts, shift_threshold_percentile)
            else:
                shift_threshold = 0.15  # Default threshold
            
            # ============================================================
            # STEP 4: OOD Detection (TEST user)
            # ============================================================
            js_shift = concept_shift_from_trained_model(
                model, X_train, X_test,
                n_clusters=n_clusters, random_state=random_state,
                max_test_samples=max_test_samples
            )
            
            is_ood = use_ood_filtering and not np.isnan(js_shift) and js_shift > shift_threshold
            if is_ood:
                ood_stats['ood_identified'] += 1
            
            # ============================================================
            # STEP 5: Evaluation (TEST user) - ALWAYS compute metrics
            # ============================================================
            # Always compute metrics, even if OOD, for proper comparison
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            try:
                auroc = roc_auc_score(y_test, y_pred_proba)
                prauc = average_precision_score(y_test, y_pred_proba)
            except ValueError as e:
                # Handle edge cases (e.g., single class in y_test)
                print(f"Warning: Could not compute metrics for user {test_user}: {e}")
                auroc = np.nan
                prauc = np.nan
            
            results[test_user] = {
                'auroc': auroc,
                'prauc': prauc,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'js_shift': js_shift,
                'is_ood': is_ood,
                'shift_threshold': shift_threshold,
                'n_train_shifts': len(train_shifts),
                'mean_train_shift': np.mean(train_shifts) if len(train_shifts) > 0 else np.nan
            }
            
        except Exception as e:
            print(f"Error for user {test_user}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results, ood_stats

# Run LOSO ONCE - compute everything, store is_ood flag
# This ensures paired comparison (same model for all users)
print("\n=== Running LOSO with OOD Detection (per-fold feature selection and threshold) ===")
print("NOTE: Computing metrics for ALL users, storing is_ood flag for comparison")
results_all, ood_stats = train_and_evaluate_loso_with_ood_detection(
    X=X_normalized,
    y=y,
    groups=groups,
    valid_users=valid_users,
    use_ood_filtering=True,  # Compute is_ood flag, but still evaluate all
    n_clusters=N_CLUSTERS,
    max_test_samples=MAX_TEST_SAMPLES_FOR_GATE,
    random_state=seed,
    top_k=TOP_K,
    shift_threshold_percentile=SHIFT_THRESHOLD_PERCENTILE
)

print(f"\nCompleted LOSO: {len(results_all)} users evaluated")
print(f"OOD users identified: {ood_stats['ood_identified']}/{ood_stats['total']} "
      f"({ood_stats['ood_identified']/ood_stats['total']*100:.1f}%)")

# Compare results: ALL users vs NON-OOD users only
print("\n=== Results Comparison: Including ALL users vs Excluding OOD users ===")

# All users (from single run)
all_users = list(results_all.keys())

# Identify OOD users using stored is_ood flag
ood_users = [u for u in all_users if results_all[u].get('is_ood', False)]
non_ood_users = [u for u in all_users if not results_all[u].get('is_ood', False)]

print(f"Total users evaluated: {len(all_users)}")
print(f"OOD users (is_ood=True): {len(ood_users)}")
print(f"Non-OOD users: {len(non_ood_users)}")
if len(ood_users) > 0 and len(ood_users) <= 20:
    print(f"OOD user names: {ood_users}")

# Extract metrics (from same run, ensuring paired comparison)
# Track counts after NaN removal for transparency
auroc_all = [results_all[u]['auroc'] for u in all_users if not np.isnan(results_all[u]['auroc'])]
auroc_non_ood = [results_all[u]['auroc'] for u in non_ood_users if not np.isnan(results_all[u]['auroc'])]
auroc_ood_only = [results_all[u]['auroc'] for u in ood_users if not np.isnan(results_all[u]['auroc'])]

prauc_all = [results_all[u]['prauc'] for u in all_users if not np.isnan(results_all[u]['prauc'])]
prauc_non_ood = [results_all[u]['prauc'] for u in non_ood_users if not np.isnan(results_all[u]['prauc'])]
prauc_ood_only = [results_all[u]['prauc'] for u in ood_users if not np.isnan(results_all[u]['prauc'])]

n_all_valid_auroc = len(auroc_all)
n_nonood_valid_auroc = len(auroc_non_ood)
n_ood_valid_auroc = len(auroc_ood_only)

n_all_valid_prauc = len(prauc_all)
n_nonood_valid_prauc = len(prauc_non_ood)
n_ood_valid_prauc = len(prauc_ood_only)

# Summary statistics
print("\n=== Performance Summary: Including ALL vs Excluding OOD ===")
print(f"\n{'Metric':<25} {'Including ALL':<25} {'Excluding OOD':<25} {'Difference':<20}")
print("-" * 95)
print(f"{'Sample sizes (after NaN removal)':<25} {'n=' + str(n_all_valid_auroc):<25} {'n=' + str(n_nonood_valid_auroc):<25}")
print("-" * 95)

mean_auroc_all = np.mean(auroc_all) if len(auroc_all) > 0 else np.nan
mean_auroc_non_ood = np.mean(auroc_non_ood) if len(auroc_non_ood) > 0 else np.nan
mean_prauc_all = np.mean(prauc_all) if len(prauc_all) > 0 else np.nan
mean_prauc_non_ood = np.mean(prauc_non_ood) if len(prauc_non_ood) > 0 else np.nan

diff_auroc = mean_auroc_non_ood - mean_auroc_all if not (np.isnan(mean_auroc_non_ood) or np.isnan(mean_auroc_all)) else np.nan
diff_prauc = mean_prauc_non_ood - mean_prauc_all if not (np.isnan(mean_prauc_non_ood) or np.isnan(mean_prauc_all)) else np.nan

print(f"{'Mean AUROC':<25} {mean_auroc_all:<25.4f} {mean_auroc_non_ood:<25.4f} {diff_auroc:<20.4f}")
std_auroc_all = np.std(auroc_all) if len(auroc_all) > 0 else np.nan
std_auroc_non_ood = np.std(auroc_non_ood) if len(auroc_non_ood) > 0 else np.nan
print(f"{'Std AUROC':<25} {std_auroc_all:<25.4f} {std_auroc_non_ood:<25.4f}")
print(f"{'Mean PRAUC':<25} {mean_prauc_all:<25.4f} {mean_prauc_non_ood:<25.4f} {diff_prauc:<20.4f}")
std_prauc_all = np.std(prauc_all) if len(prauc_all) > 0 else np.nan
std_prauc_non_ood = np.std(prauc_non_ood) if len(prauc_non_ood) > 0 else np.nan
print(f"{'Std PRAUC':<25} {std_prauc_all:<25.4f} {std_prauc_non_ood:<25.4f}")
print(f"{'Sample sizes (PRAUC)':<25} {'n=' + str(n_all_valid_prauc):<25} {'n=' + str(n_nonood_valid_prauc):<25}")

# Compare OOD vs Non-OOD user performance (already shown in main metrics, but kept for clarity)
if len(ood_users) > 0:
    print(f"\n--- OOD Users Performance (for reference) ---")
    if len(auroc_ood_only) > 0:
        print(f"Mean AUROC (OOD users only): {np.mean(auroc_ood_only):.4f} ± {np.std(auroc_ood_only):.4f} (n={n_ood_valid_auroc})")
    if len(prauc_ood_only) > 0:
        print(f"Mean PRAUC (OOD users only): {np.mean(prauc_ood_only):.4f} ± {np.std(prauc_ood_only):.4f} (n={n_ood_valid_prauc})")

# Statistical test: OOD vs Non-OOD (two independent groups)
# Also compute Cliff's delta effect size
def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size.
    Returns value in [-1, 1]:
    - -1: all values in x are greater than all values in y
    - 0: no effect
    - 1: all values in x are less than all values in y
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # Count how many times x[i] > y[j] and x[i] < y[j]
    greater = 0
    less = 0
    
    for xi in x_arr:
        greater += np.sum(xi > y_arr)
        less += np.sum(xi < y_arr)
    
    delta = (greater - less) / (n_x * n_y)
    return delta

if len(auroc_ood_only) > 0 and len(auroc_non_ood) > 0:
    from scipy.stats import mannwhitneyu
    try:
        stat_auroc, pval_auroc = mannwhitneyu(auroc_ood_only, auroc_non_ood, alternative='two-sided')
        cliff_delta_auroc = cliffs_delta(auroc_ood_only, auroc_non_ood)
        print(f"\n=== Statistical Tests (Mann-Whitney U: OOD vs Non-OOD) ===")
        print(f"AUROC: statistic={stat_auroc:.4f}, p-value={pval_auroc:.4f}")
        print(f"  OOD group: n={len(auroc_ood_only)}, mean={np.mean(auroc_ood_only):.4f} ± {np.std(auroc_ood_only):.4f}")
        print(f"  Non-OOD group: n={len(auroc_non_ood)}, mean={np.mean(auroc_non_ood):.4f} ± {np.std(auroc_non_ood):.4f}")
        print(f"  Cliff's delta (effect size): {cliff_delta_auroc:.4f}")
        print(f"    (negative = OOD worse, positive = OOD better, |d|>0.147 = small, |d|>0.33 = medium, |d|>0.474 = large)")
    except Exception as e:
        print(f"\nNote: AUROC statistical test not performed: {e}")

if len(prauc_ood_only) > 0 and len(prauc_non_ood) > 0:
    from scipy.stats import mannwhitneyu
    try:
        stat_prauc, pval_prauc = mannwhitneyu(prauc_ood_only, prauc_non_ood, alternative='two-sided')
        cliff_delta_prauc = cliffs_delta(prauc_ood_only, prauc_non_ood)
        print(f"PRAUC: statistic={stat_prauc:.4f}, p-value={pval_prauc:.4f}")
        print(f"  OOD group: n={len(prauc_ood_only)}, mean={np.mean(prauc_ood_only):.4f} ± {np.std(prauc_ood_only):.4f}")
        print(f"  Non-OOD group: n={len(prauc_non_ood)}, mean={np.mean(prauc_non_ood):.4f} ± {np.std(prauc_non_ood):.4f}")
        print(f"  Cliff's delta (effect size): {cliff_delta_prauc:.4f}")
        print(f"    (negative = OOD worse, positive = OOD better, |d|>0.147 = small, |d|>0.33 = medium, |d|>0.474 = large)")
    except Exception as e:
        print(f"Note: PRAUC statistical test not performed: {e}")

# Visualization
if HAS_MATPLOTLIB and len(all_users) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get AUROC and PRAUC values for all users (handle NaN)
    auroc_values = [results_all[u]['auroc'] if not np.isnan(results_all[u]['auroc']) else np.nan for u in all_users]
    prauc_values = [results_all[u]['prauc'] if not np.isnan(results_all[u]['prauc']) else np.nan for u in all_users]
    
    # AUROC: All vs Non-OOD comparison
    colors = ['red' if u in ood_users else 'blue' for u in all_users]
    axes[0].scatter(range(len(all_users)), auroc_values, c=colors, alpha=0.6, s=100)
    if len(non_ood_users) > 0 and not np.isnan(mean_auroc_non_ood):
        axes[0].axhline(y=mean_auroc_non_ood, color='green', linestyle='--', linewidth=2)
    if not np.isnan(mean_auroc_all):
        axes[0].axhline(y=mean_auroc_all, color='orange', linestyle='--', linewidth=2)
    axes[0].set_xlabel('User Index', fontsize=12)
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('AUROC: All Users vs Non-OOD Only (LOSO)', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(range(len(all_users)))
    axes[0].set_xticklabels([u.replace('P', '') for u in all_users], rotation=45, ha='right')
    
    # PRAUC: All vs Non-OOD comparison
    axes[1].scatter(range(len(all_users)), prauc_values, c=colors, alpha=0.6, s=100)
    if len(non_ood_users) > 0 and not np.isnan(mean_prauc_non_ood):
        axes[1].axhline(y=mean_prauc_non_ood, color='green', linestyle='--', linewidth=2)
    if not np.isnan(mean_prauc_all):
        axes[1].axhline(y=mean_prauc_all, color='orange', linestyle='--', linewidth=2)
    axes[1].set_xlabel('User Index', fontsize=12)
    axes[1].set_ylabel('PRAUC', fontsize=12)
    axes[1].set_title('PRAUC: All Users vs Non-OOD Only (LOSO)', fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(range(len(all_users)))
    axes[1].set_xticklabels([u.replace('P', '') for u in all_users], rotation=45, ha='right')
    
    # Add legend for colors and means
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements_auroc = [
        Patch(facecolor='blue', label='Non-OOD users'),
        Patch(facecolor='red', label='OOD users')
    ]
    if not np.isnan(mean_auroc_non_ood):
        legend_elements_auroc.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Mean (Non-OOD): {mean_auroc_non_ood:.4f}'))
    if not np.isnan(mean_auroc_all):
        legend_elements_auroc.append(Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Mean (All): {mean_auroc_all:.4f}'))
    axes[0].legend(handles=legend_elements_auroc, loc='best')
    
    legend_elements_prauc = [
        Patch(facecolor='blue', label='Non-OOD users'),
        Patch(facecolor='red', label='OOD users')
    ]
    if not np.isnan(mean_prauc_non_ood):
        legend_elements_prauc.append(Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Mean (Non-OOD): {mean_prauc_non_ood:.4f}'))
    if not np.isnan(mean_prauc_all):
        legend_elements_prauc.append(Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Mean (All): {mean_prauc_all:.4f}'))
    axes[1].legend(handles=legend_elements_prauc, loc='best')
    
    plt.tight_layout()
    plt.savefig('ood_detection_loso_comparison.png', dpi=300)
    plt.close()
    print("Saved: ood_detection_loso_comparison.png")

# Detailed results table - include ALL users
results_df = pd.DataFrame({
    'User': all_users,
    'AUROC': [results_all[u]['auroc'] for u in all_users],
    'PRAUC': [results_all[u]['prauc'] for u in all_users],
    'JS_Shift': [results_all[u]['js_shift'] for u in all_users],
    'IsOOD': [results_all[u]['is_ood'] for u in all_users],
    'ShiftThreshold': [results_all[u]['shift_threshold'] for u in all_users],
    'N_TrainShifts': [results_all[u]['n_train_shifts'] for u in all_users],
    'MeanTrainShift': [results_all[u]['mean_train_shift'] for u in all_users],
})

results_df = results_df.sort_values('JS_Shift', ascending=False)

print("\n=== Detailed Results (All users, sorted by JS Shift) ===")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('ood_detection_loso_results.csv', index=False)
print("\nResults saved to 'ood_detection_loso_results.csv'")

# Final summary report
print("\n" + "="*80)
print("FINAL SUMMARY REPORT - OOD Detection on D#4 Dataset (LOSO, per-fold feature selection and threshold)")
print("NOTE: Uses deployable concept shift (cluster-conditional prediction distribution shift)")
print("      - Feature selection: Top-40 features selected PER-FOLD from TRAINING users only")
print("      - Threshold calibration: PER-FOLD from TRAINING users only (using SAME fold model)")
print("      - Threshold uses X_train_minus_user (not full X_train) for better comparability to test shifts")
print("      - No y_test required, fits KMeans on train only; uses fold-trained model for predictions")
print("      - Each fold is completely independent (no data leakage)")
print("      - Single run ensures paired comparison (same model for all users)")
print("      - Deterministic XGBoost settings (n_jobs=1) for reproducibility")
print("="*80)
print(f"\nDataset: D#4")
print(f"Total users: {len(user_ids)}")
print(f"Valid users for LOSO: {len(valid_users)}")
print(f"Top-K features per fold: {TOP_K}")
print(f"Shift threshold percentile: {SHIFT_THRESHOLD_PERCENTILE}th")
print(f"OOD users identified: {ood_stats['ood_identified']}/{ood_stats['total']} ({ood_stats['ood_identified']/ood_stats['total']*100:.1f}%)")

print(f"\n--- Performance Metrics ---")
if len(auroc_all) > 0:
    print(f"Mean AUROC (Including ALL users): {mean_auroc_all:.4f} ± {std_auroc_all:.4f} (n={n_all_valid_auroc})")
if len(auroc_non_ood) > 0:
    print(f"Mean AUROC (Excluding OOD users): {mean_auroc_non_ood:.4f} ± {std_auroc_non_ood:.4f} (n={n_nonood_valid_auroc})")
    if len(auroc_all) > 0:
        print(f"AUROC Change: {mean_auroc_non_ood - mean_auroc_all:+.4f}")
if len(auroc_ood_only) > 0:
    print(f"Mean AUROC (OOD users only): {np.mean(auroc_ood_only):.4f} ± {np.std(auroc_ood_only):.4f} (n={n_ood_valid_auroc})")

if len(prauc_all) > 0:
    print(f"\nMean PRAUC (Including ALL users): {mean_prauc_all:.4f} ± {std_prauc_all:.4f} (n={n_all_valid_prauc})")
if len(prauc_non_ood) > 0:
    print(f"Mean PRAUC (Excluding OOD users): {mean_prauc_non_ood:.4f} ± {std_prauc_non_ood:.4f} (n={n_nonood_valid_prauc})")
    if len(prauc_all) > 0:
        print(f"PRAUC Change: {mean_prauc_non_ood - mean_prauc_all:+.4f}")
if len(prauc_ood_only) > 0:
    print(f"Mean PRAUC (OOD users only): {np.mean(prauc_ood_only):.4f} ± {np.std(prauc_ood_only):.4f} (n={n_ood_valid_prauc})")

if len(ood_users) > 0:
    print(f"\n--- OOD Filtering Impact ---")
    print(f"OOD users identified: {len(ood_users)}/{len(all_users)} ({len(ood_users)/len(all_users)*100:.1f}%)")
    print(f"Note: Threshold calibration uses X_train_minus_user (not full X_train) for better comparability")

print("\n" + "="*80)
print("\nAnalysis complete!")
