#!/usr/bin/env python3
"""
Individual Model Pipeline Runner

Usage: python RQ2/run_model_pipeline.py --model lgbm
"""

import sys
import os
import subprocess



os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import logging

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*is_sparse.*")
warnings.filterwarnings("ignore", message=".*is_categorical_dtype.*")
warnings.filterwarnings("ignore", message=".*SparseDtype.*")
warnings.filterwarnings("ignore", message=".*CategoricalDtype.*")
warnings.filterwarnings("ignore", message=".*builtin type swigvarlink.*")
warnings.filterwarnings("ignore", message=".*ot.gpu not found.*")
warnings.filterwarnings("ignore", message=".*coupling computation will be in cpu.*")
warnings.filterwarnings("ignore", message=".*computation placer already registered.*")
warnings.filterwarnings("ignore", message=".*Unable to register.*")

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import argparse
import os
import re
import json
from datetime import datetime
from typing import Any, List, Dict, Tuple
import numpy as np
import pandas as pd
import cloudpickle
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import spearmanr, pearsonr, wilcoxon
from statsmodels.stats.multitest import fdrcorrection  # type: ignore
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression

from utility_deep import *
from models import *

from dataset_config import DATASET_CONFIGS
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*builtin type swigvarlink.*")
warnings.filterwarnings("ignore", message=".*ot.gpu not found.*")

RESULTS_DIR = os.environ.get("CHI_RESULTS_DIR", "results")
PROCESSED_DIR = os.path.join(RESULTS_DIR, "processed")

def wilcoxon_greater(x):
    """One-sided Wilcoxon test for positive median.
    
    Args:
        x: Array-like of numeric values.
        
    Returns:
        float: P-value for H1: median(x) > 0, or nan if insufficient data.
    """
    x = np.array(x)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    try:
        _, p = wilcoxon(x, alternative='greater')
        return p
    except:
        return np.nan

def sign_test_greater(x):
    """One-sided sign test for positive bias.
    
    Args:
        x: Array-like of numeric values.
        
    Returns:
        float: P-value for H1: P(x > 0) > 0.5, or nan if insufficient data.
    """
    x = np.array(x)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    try:
        from scipy.stats import binom_test
        n_pos = (x > 0).sum()
        n = len(x)
        return binom_test(n_pos, n, p=0.5, alternative='greater')
    except:
        return np.nan

def fdr_bh_series(p_values):
    """Apply Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: Array-like of p-values.
        
    Returns:
        numpy.ndarray: FDR-corrected q-values.
    """
    p_arr = np.array(p_values)
    mask = ~np.isnan(p_arr)
    if not mask.any():
        return p_arr
    try:
        _, q_adj = fdrcorrection(p_arr[mask], alpha=0.05, method='indep')
        result = np.full_like(p_arr, np.nan)
        result[mask] = q_adj
        return result
    except:
        return p_arr

def extract_shift_type_from_text(s: str) -> str:
    """Extract shift type from shift name.
    
    Args:
        s: Shift name string (e.g., 'PhoneUsage Covariate shift').
        
    Returns:
        str: Shift type (e.g., 'Covariate').
    """
    tokens = str(s).strip().split()
    if len(tokens) >= 2 and tokens[-1].lower() == 'shift':
        return tokens[-2]
    m = re.search(r'(\w+)\s+shift', str(s))
    return m.group(1) if m else str(s)

def parse_cat_and_shift_from_key(key: str):
    """Parse category and shift from combined key.
    
    Args:
        key: Combined key string (e.g., 'PhoneUsage_PhoneUsage Covariate shift').
        
    Returns:
        tuple: (category, full_shift_name)
    """
    parts = key.split('_', 1)
    cat = parts[0]
    shift_full = parts[1].replace('_', ' ') if len(parts) > 1 else parts[0]
    return cat, shift_full

def filter_users_notebook_style(
    X_train_normalized: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test_normalized: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    pra_threshold: float,
    random_state: int,
    n_estimators_override: int = None,
):
    """Mimic the notebook's filter_users logic (XGBClassifier, n_estimators=500, depth=3, hist)."""
    from xgboost import XGBClassifier  # local import to avoid NameError in joblib workers

    def check_user(user):
        train_idx = (groups_train == user)
        X_u_train = X_train_normalized.loc[train_idx]
        y_u_train = y_train[train_idx]
        
        test_idx = (groups_test == user)
        X_u_test = X_test_normalized.loc[test_idx]
        y_u_test = y_test[test_idx]
        
        if (len(np.unique(y_u_train)) < 2 or len(X_u_train) < 10 or 
            len(np.unique(y_u_test)) < 2 or len(X_u_test) < 5):
            return None

        model = XGBClassifier(
            n_estimators=n_estimators_override or 500,
            max_depth=3,
            eval_metric='auc',
            tree_method='hist',
            n_jobs=-1,
            random_state=random_state
        )
        try:
            model.fit(X_u_train, y_u_train)
            proba = model.predict_proba(X_u_test)[:, 1]
            prauc = average_precision_score(y_u_test, proba)
            if prauc >= pra_threshold:
                return user
            else:
                log(f"User {user} removed with xgb (PRAUC = {prauc:.3f})")
                return None
        except Exception as e:
            log(f"User {user} failed during filtering: {e}")
            return None
    common_users = np.intersect1d(groups_train, groups_test)
    # Use loky backend for true multiprocessing (matches original notebook style)
    results = Parallel(n_jobs=4, backend='loky')(
        delayed(check_user)(u) for u in common_users
    )
    valid_users = [u for u in results if u is not None]

    mask = np.isin(groups_train, valid_users)
    X_filt = X_train_normalized.loc[mask].reset_index(drop=True)
    y_filt = y_train[mask]
    grp_filt = groups_train[mask]

    return valid_users, X_filt, pd.Series(dtype=object), y_filt, grp_filt, None, None

def evaluate_users_notebook_style(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    common_feats: List[str],
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Mimic notebook evaluate_users_with_fixed_splits (XGB 100 trees)."""
    from xgboost import XGBClassifier  # local import for worker threads
    from sklearn.preprocessing import StandardScaler
    import shap

    def _process_user(user):
        train_mask = (groups_train == user)
        X_train_user = X_train.loc[train_mask]
        y_train_user = y_train[train_mask]
        
        test_mask = (groups_test == user)
        X_test_user = X_test.loc[test_mask]
        y_test_user = y_test[test_mask]
        
        if len(np.unique(y_train_user)) < 2 or len(np.unique(y_test_user)) < 2:
            return None
        if len(X_train_user) < 5 or len(X_test_user) < 2:
            return None
            
        common_cols = X_train_user.columns.intersection(X_test_user.columns)
        if len(common_cols) == 0:
            return None
            
        X_train_final = X_train_user[common_cols]
        X_test_common = X_test_user[common_cols]
        
        if len(X_test_common) == 0 or X_test_common.shape[1] == 0:
            return None
            
        try:
            numeric_test_cols = X_test_common.select_dtypes(include=[np.number]).columns
            if len(numeric_test_cols) > 0:
                test_scaler = StandardScaler()
                X_test_numeric = X_test_common[numeric_test_cols]
                if X_test_numeric.var().sum() > 0:
                    X_test_scaled = pd.DataFrame(
                        test_scaler.fit_transform(X_test_numeric),
                        columns=numeric_test_cols,
                        index=X_test_numeric.index
                    )
                    non_numeric_cols = X_test_common.columns.difference(numeric_test_cols)
                    if len(non_numeric_cols) > 0:
                        X_test_final = pd.concat([
                            X_test_scaled,
                            X_test_common[non_numeric_cols]
                        ], axis=1)
                    else:
                        X_test_final = X_test_scaled
                else:
                    X_test_final = X_test_common
            else:
                X_test_final = X_test_common
        except Exception:
            X_test_final = X_test_common
        
        problematic_chars = ['[', ']', '<', '>', '{', '}', '(', ')', ',']
        cols_to_drop = []
        for col in X_train_final.columns:
            if any(char in str(col) for char in problematic_chars):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            X_train_final = X_train_final.drop(columns=cols_to_drop)
            X_test_final = X_test_final.drop(columns=cols_to_drop)
        
        # Align column order and restrict to available common features
        X_test_final = X_test_final[X_train_final.columns]
        available_common_feats = [f for f in common_feats if f in X_train_final.columns]
        if not available_common_feats:
            return None
            
        X_train_common = X_train_final[available_common_feats]
        X_test_common = X_test_final[available_common_feats]
        
        def fit_and_predict(X_tr, y_tr, X_te):
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                eval_metric='auc',
                tree_method='hist',
                random_state=random_state
            )
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            return model, proba

        m_c, pred_c = fit_and_predict(X_train_common, y_train_user, X_test_common)
        sc_c = {
            'auroc': roc_auc_score(y_test_user, pred_c),
            'prauc': average_precision_score(y_test_user, pred_c)
        }

        m_f, pred_f = fit_and_predict(X_train_final, y_train_user, X_test_final)
        sc_f = {
            'auroc': roc_auc_score(y_test_user, pred_f),
            'prauc': average_precision_score(y_test_user, pred_f)
        }

        imp = m_c.feature_importances_
        explainer = shap.TreeExplainer(m_c)
        shap_vals = explainer.shap_values(X_train_common)
        if isinstance(shap_vals, list):
            shap_vals = np.array(shap_vals).mean(axis=0)
        shap_imp = np.abs(shap_vals).mean(axis=0)

        corr = X_train_common.corr()

        norm_imp = imp / imp.sum() if imp.sum() > 0 else np.ones_like(imp) / len(imp)
        norm_shap = shap_imp / shap_imp.sum() if shap_imp.sum() > 0 else np.ones_like(shap_imp) / len(shap_imp)

        return user, sc_c, sc_f, norm_imp, norm_shap, corr

    train_users = set(np.unique(groups_train))
    test_users = set(np.unique(groups_test))
    common_users = list(train_users.intersection(test_users))

    results = Parallel(n_jobs=8, backend='threading')(
        delayed(_process_user)(u) for u in common_users
    )

    common_scores = {}
    full_scores = {}
    user_imp = {}
    user_imp_map = {}
    corr_maps = {}

    for res in results:
        if res is None:
            continue
        user, sc_c, sc_f, imp_arr, shap_arr, corr = res
        common_scores[user] = sc_c
        full_scores[user] = sc_f
        user_imp[user] = imp_arr
        user_imp_map[user] = shap_arr
        corr_maps[user] = corr

    return common_scores, full_scores, user_imp, user_imp_map, corr_maps

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Run individual model pipeline')
    parser.add_argument('--model', type=str, required=True,
                       choices=['lgbm', 'xgb', 'catboost', 'mlp', 'tabtransformer', 'fttransformer', 'node', 'tabresnet'],
                       help='Model name to run pipeline for')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Dataset identifier (e.g., D-1, D-2, K-EmoPhone, studentlife) to load configuration from')
    parser.add_argument('--pra_threshold', type=float, default=0.65, help='PRA threshold for user filtering')
    parser.add_argument('--auroc_threshold', type=float, default=0.65, help='AUROC threshold for anchor users')
    parser.add_argument('--top_k', type=int, default=None, help='Number of top features to select (CLI overrides dataset config; defaults to dataset config or 40)')
    
    parser.add_argument('--n_estimators', type=int, help='Number of estimators for tree models (default: 200)')
    parser.add_argument('--max_depth', type=int, help='Max depth for tree models (default: varies by model)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (default: 0.1)')
    parser.add_argument('--subsample', type=float, help='Subsample ratio (default: 0.8)')
    parser.add_argument('--reg_alpha', type=float, help='L1 regularization (default: 0.1)')
    parser.add_argument('--reg_lambda', type=float, help='L2 regularization (default: 1.0)')
    parser.add_argument('--early_stopping_rounds', type=int, help='Early stopping rounds (default: 50)')
    
    parser.add_argument('--epochs', type=int, help='Number of epochs for deep models (default: varies by model)')
    parser.add_argument('--batch_size', type=int, help='Batch size for deep models (default: 128)')
    parser.add_argument('--lr', type=float, help='Learning rate for deep models (default: 1e-3)')
    parser.add_argument('--patience', type=int, help='Early stopping patience for deep models (default: varies)')
    parser.add_argument('--d_model', type=int, help='Model dimension for transformer models (default: 64)')
    parser.add_argument('--nhead', type=int, help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, help='Number of layers (default: 2)')
    parser.add_argument('--dropout', type=float, help='Dropout rate (default: 0.1)')
    parser.add_argument('--hidden_layer_sizes', type=str, help='Hidden layer sizes for MLP as comma-separated values (e.g., "64,32")')
    parser.add_argument('--max_iter', type=int, help='Max iterations for MLP (default: 200)')
    parser.add_argument('--split_strategy', type=str, default='random',
                        choices=['random', 'loso', 'stratified_group_kfold'],
                        help='Train/test splitting strategy')
    parser.add_argument('--fold_index', type=int, default=0,
                        help='Fold index for LOSO or StratifiedGroupKFold splits')
    parser.add_argument('--fold_group', type=str, default=None,
                        help='Explicit group identifier for LOSO strategy')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for StratifiedGroupKFold strategy')
    parser.add_argument('--no_shuffle_splits', action='store_true',
                        help='Disable shuffling for StratifiedGroupKFold strategy')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test fraction for random split strategy')
    parser.add_argument('--run_all_folds', action='store_true',
                        help='Iterate through every fold (LOSO/StratifiedGroupKFold only)')
    parser.add_argument('--distance_jobs', type=int, default=None,
                        help='Number of parallel jobs for distance matrices (default: min(8, cpu_count//2) for tree models, otherwise 1)')
    parser.add_argument('--distance_backend', type=str, default='threading',
                        choices=['threading', 'loky'],
                        help='Joblib backend for distance matrices (default: threading)')
    parser.add_argument('--concept_shift_method', type=str, default='model',
                        choices=['model', 'cluster', 'knn', 'detectshift'],
                        help='Concept shift computation: model (JSD on model outputs), cluster (KMeans bins), knn, or detectshift.')
    parser.add_argument('--detectshift_test_size', type=float, default=0.2,
                        help='Test size for DetectShift internal split (default: 0.2)')
    parser.add_argument('--detectshift_B', type=int, default=200,
                        help='Number of permutations for DetectShift (default: 200)')
    
    args = parser.parse_args()
    
    # Parse model hyperparameters
    model_params = {}
    
    if args.n_estimators is not None:
        model_params['n_estimators'] = args.n_estimators
    if args.max_depth is not None:
        model_params['max_depth'] = args.max_depth
    if args.learning_rate is not None:
        model_params['learning_rate'] = args.learning_rate
    if args.subsample is not None:
        model_params['subsample'] = args.subsample
    if args.reg_alpha is not None:
        model_params['reg_alpha'] = args.reg_alpha
    if args.reg_lambda is not None:
        model_params['reg_lambda'] = args.reg_lambda
    if args.early_stopping_rounds is not None:
        model_params['early_stopping_rounds'] = args.early_stopping_rounds
    if args.epochs is not None:
        model_params['epochs'] = args.epochs
    if args.batch_size is not None:
        model_params['batch_size'] = args.batch_size
    if args.lr is not None:
        model_params['lr'] = args.lr
    if args.patience is not None:
        model_params['patience'] = args.patience
    if args.d_model is not None:
        model_params['d_model'] = args.d_model
    if args.nhead is not None:
        model_params['nhead'] = args.nhead
    if args.num_layers is not None:
        model_params['num_layers'] = args.num_layers
    if args.dropout is not None:
        model_params['dropout'] = args.dropout
    if args.hidden_layer_sizes is not None:
        # Parse comma-separated string to tuple
        model_params['hidden_layer_sizes'] = tuple(map(int, args.hidden_layer_sizes.split(',')))
    if args.max_iter is not None:
        model_params['max_iter'] = args.max_iter

    # Align XGB defaults with notebook configuration if user did not override them
    if args.model == 'xgb':
        xgb_defaults = {
            'n_estimators': 500,
            'max_depth': 3,
            'early_stopping_rounds': 100,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0
        }
        for k, v in xgb_defaults.items():
            model_params.setdefault(k, v)
    
    print(f"=== Running Pipeline for Model: {args.model.upper()} ===")
    if model_params:
        print(f"Using custom hyperparameters: {model_params}")

    # Load dataset config if specified
    if hasattr(args, 'dataset') and args.dataset in DATASET_CONFIGS:
        config = DATASET_CONFIGS[args.dataset]
        print(f"=== Loading Configuration for {args.dataset} ===")
        dataset_path = config["data_path"]
        args.pra_threshold = config["pra_threshold"]
        args.auroc_threshold = config["auroc_threshold"]
        top_k_cfg = config.get("top_k")
        if args.top_k is None and top_k_cfg is not None:
            args.top_k = top_k_cfg
        args.concept_shift_k = config.get("concept_shift_k", 20)
        args.concept_shift_min_samples = config.get("concept_shift_min_samples", 5)
        print(f"Dataset: {dataset_path}")
        print(f"PRA Threshold: {args.pra_threshold}")
        print(f"AUROC Threshold: {args.auroc_threshold}")
        print(f"Top K: {args.top_k if args.top_k is not None else top_k_cfg}")
        print(f"Concept Shift K: {args.concept_shift_k}")
        print(f"Concept Shift Min Samples: {args.concept_shift_min_samples}")
        
    else:
        dataset_path = "data/Archived/stress_binary_personal-current.pkl"
        args.concept_shift_k = 20
        args.concept_shift_min_samples = 5
        if args.top_k is None:
            args.top_k = 40
        print("Using default dataset path and settings")
        print(f"Top K: {args.top_k}")

    if args.top_k is None:
        args.top_k = 40
    
    print("=== Loading Data ===")
    data_path = dataset_path
    X, y, groups, _, _ = load(data_path)
    
    pif_cols = [col for col in X.columns if "PIF" in col]
    X = X.drop(columns=pif_cols)
    X = X.drop(columns=[col for col in X.columns if col.startswith("ESM")])
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]
    groups = groups[mask]
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(groups))} users")

    if args.run_all_folds and args.split_strategy in ('loso', 'stratified_group_kfold'):
        if args.split_strategy == 'loso':
            unique_groups = np.unique(groups)
            fold_tasks = []
            for idx, group in enumerate(unique_groups):
                group_value = group.item() if hasattr(group, "item") else group
                fold_tasks.append((idx, str(group_value)))
        else:
            fold_tasks = [(idx, None) for idx in range(args.n_splits)]

        script_path = os.path.abspath(__file__)
        base_cli = sys.argv[1:]

        def build_child_args(fold_idx, fold_group):
            filtered = []
            skip_next = False
            for arg in base_cli:
                if skip_next:
                    skip_next = False
                    continue
                if arg in ('--fold_index', '--fold_group'):
                    skip_next = True
                    continue
                if arg == '--run_all_folds':
                    continue
                filtered.append(arg)
            filtered.extend(['--fold_index', str(fold_idx)])
            if fold_group is not None:
                filtered.extend(['--fold_group', fold_group])
            return filtered

        total_tasks = len(fold_tasks)
        for idx, (fold_idx, fold_group) in enumerate(fold_tasks, 1):
            desc_group = fold_group if fold_group is not None else 'None'
            print(f"\n>>> Running fold {idx}/{total_tasks} (fold_index={fold_idx}, fold_group={desc_group})")
            child_args = build_child_args(fold_idx, fold_group)
            subprocess.run([sys.executable, script_path] + child_args, check=True)

        print("\nAll folds completed.")
        return
    
    print("=== Creating Train/Test Splits ===")
    if args.split_strategy == "random":
        print(f"Split strategy: random (test_size={args.test_size:.2f}, seed={args.seed})")
    elif args.split_strategy == "loso":
        fold_desc = args.fold_group if args.fold_group is not None else args.fold_index
        print(f"Split strategy: LOSO (fold={fold_desc})")
    else:
        print(
            f"Split strategy: StratifiedGroupKFold (n_splits={args.n_splits}, "
            f"fold={args.fold_index}, shuffle={not args.no_shuffle_splits}, seed={args.seed})"
        )

    #Use this same split for upcoming steps
    X_train, y_train, groups_train, X_test, y_test, groups_test = create_train_test_splits(
        X=X,
        y=y,
        groups=groups,
        test_size=args.test_size,
        random_state=args.seed,
        strategy=args.split_strategy,
        fold_index=args.fold_index,
        fold_group=args.fold_group,
        n_splits=args.n_splits,
        shuffle=not args.no_shuffle_splits,
    )
    raw_train_samples = int(len(y_train))
    raw_test_samples = int(len(y_test))
    raw_train_users = int(np.unique(groups_train).size)
    raw_test_users = int(np.unique(groups_test).size)

    logs_dir = RESULTS_DIR
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "pipeline_run_log.csv")

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "seed": args.seed,
        "split_strategy": args.split_strategy,
        "fold_index": args.fold_index,
        "fold_group": args.fold_group if args.fold_group is not None else "",
        "n_splits": args.n_splits,
        "shuffle_splits": bool(not args.no_shuffle_splits),
        "test_size": args.test_size if args.split_strategy == "random" else float("nan"),
        "pra_threshold": args.pra_threshold,
        "auroc_threshold": args.auroc_threshold,
        "top_k": args.top_k,
        "raw_train_samples": raw_train_samples,
        "raw_test_samples": raw_test_samples,
        "raw_train_users": raw_train_users,
        "raw_test_users": raw_test_users,
        "filtered_train_samples": 0,
        "filtered_train_users": 0,
        "filtered_test_samples": 0,
        "filtered_test_users": 0,
        "valid_users_first_pass": 0,
        "anchor_users": 0,
        "anchor_train_samples": 0,
        "anchor_test_samples": 0,
        "selected_features": 0,
        "summary_cat_shift_path": "",
        "summary_shift_path": "",
        "median_shift_slope": float("nan"),
        "median_shift_pos_rate": float("nan"),
        "median_category_slope": float("nan"),
        "strongest_shift_type": "",
        "strongest_shift_rho": float("nan"),
        "strongest_shift_median": float("nan"),
        "model_params": json.dumps(model_params) if model_params else "{}"
    }

    def append_run_log(summary_dict):
        log_df = pd.DataFrame([summary_dict])
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
    
    print("=== Normalizing Data ===")
    # drop categorical columns with zero variance
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_zero_var_cols = [col for col in categorical_cols if X[col].nunique() <= 1]
    categorical_cols = X[categorical_cols].drop(columns=cat_zero_var_cols)
    
    # Normalization cache (per dataset/split/seed)
    norm_cache_dir = os.path.join(RESULTS_DIR, "cache_normalized")
    os.makedirs(norm_cache_dir, exist_ok=True)
    cache_key_parts = [
        os.path.basename(dataset_path).replace('.pkl',''),
        args.split_strategy,
        f"seed{args.seed}",
    ]
    if args.split_strategy == "random":
        cache_key_parts.append(f"test{args.test_size}")
    else:
        cache_key_parts.append(f"fold{args.fold_index}")
        cache_key_parts.append(f"group{args.fold_group if args.fold_group else 'none'}")
    norm_cache_path = os.path.join(norm_cache_dir, "__".join(map(str, cache_key_parts)) + ".pkl")
    
    X_train_normalized = None
    X_test_normalized = None
    if os.path.exists(norm_cache_path):
        try:
            with open(norm_cache_path, "rb") as f:
                cached = cloudpickle.load(f)
            c_train_cols = cached.get("train_cols", [])
            c_test_cols = cached.get("test_cols", [])
            if c_train_cols == list(X_train.columns) and c_test_cols == list(X_test.columns):
                X_train_normalized = cached["X_train_normalized"]
                X_test_normalized = cached["X_test_normalized"]
                print(f"[cache] Loaded normalized data from {norm_cache_path}")
        except Exception as e:
            print(f"[cache] Failed to load normalization cache ({e}); recomputing.")
    
    if X_train_normalized is None or X_test_normalized is None:
        # only normalize numeric columns and in user wise manner
        X_train_normalized, X_test_normalized = normalize_data(
            X_train=X_train, groups_train=groups_train,
            X_test=X_test, groups_test=groups_test,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )
        try:
            with open(norm_cache_path, "wb") as f:
                cloudpickle.dump({
                    "X_train_normalized": X_train_normalized,
                    "X_test_normalized": X_test_normalized,
                    "train_cols": list(X_train.columns),
                    "test_cols": list(X_test.columns)
                }, f)
            print(f"[cache] Saved normalized data to {norm_cache_path}")
        except Exception as e:
            print(f"[cache] Failed to save normalization cache ({e})")
    
    print(f"Normalized training data: {X_train_normalized.shape}")
    print(f"Normalized test data: {X_test_normalized.shape}")
    
    print("=== 1st Round User Filtering ===")
    all_users = np.unique(groups_train)
    
    # drop users with low PRA scores
    if args.model == 'xgb':
        valid_users_1st, X_train_filtered, _, y_train_filtered, groups_train_filtered, _, _ = filter_users_notebook_style(
            X_train_normalized=X_train_normalized,
            y_train=y_train,
            groups_train=groups_train,
            X_test_normalized=X_test_normalized,
            y_test=y_test,
            groups_test=groups_test,
            pra_threshold=args.pra_threshold,
            random_state=args.seed,
            n_estimators_override=300
        )
    else:
        valid_users_1st, X_train_filtered, _, y_train_filtered, groups_train_filtered, _, _ = filter_valid_users_deep(
            all_users=all_users,
            X_train_normalized=X_train_normalized,
            y_train=y_train,
            groups_train=groups_train,
            X_test_normalized=X_test_normalized,
            y_test=y_test,
            groups_test=groups_test,
            categorical_cols=pd.Series(dtype=object),
            t=None, datetimes=None,
            model_name=args.model,
            pra_threshold=args.pra_threshold,
            random_state=args.seed,
            model_params=model_params
        )
    
    print(f"Users after 1st filtering: {len(valid_users_1st)}")
    print(f"Training samples after filtering: {len(y_train_filtered)}")
    filtered_train_samples = int(len(y_train_filtered))
    filtered_train_users = int(np.unique(groups_train_filtered).size)
    run_summary.update({
        "filtered_train_samples": filtered_train_samples,
        "filtered_train_users": filtered_train_users,
        "valid_users_first_pass": len(valid_users_1st)
    })

    if len(valid_users_1st) == 0:
        message = "No valid users remain after first filtering; aborting pipeline run."
        print(message)
        run_summary.update({
            "filtered_test_samples": 0,
            "filtered_test_users": 0
        })
        append_run_log(run_summary)
        return

    print("=== Feature Selection ===")
    filtered_test_mask = np.isin(groups_test, valid_users_1st)
    X_test_filtered_for_feature_selection = X_test_normalized.loc[filtered_test_mask]
    y_test_filtered_for_feature_selection = y_test[filtered_test_mask]
    
    if args.model == 'xgb':
        common_feats = select_top_xgb_features(
            X_train=X_train_filtered,
            y_train=y_train_filtered,
            X_test=X_test_filtered_for_feature_selection,
            y_test=y_test_filtered_for_feature_selection,
            top_k=args.top_k,
            random_state=args.seed
        )
    else:
        # Use the specific model for feature selection instead of XGBoost
        common_feats = select_top_model_features(
            X_train=X_train_filtered,
            y_train=y_train_filtered,
            X_test=X_test_filtered_for_feature_selection,
            y_test=y_test_filtered_for_feature_selection,
            model_name=args.model,
            top_k=args.top_k,
            random_state=args.seed,
            model_params=model_params
        )
    
    print(f"Selected {len(common_feats)} features using {args.model}")
    
    # STEP 5: 2nd Round User Filtering (Anchor User Selection)
    # because feature selection may change user performance
    print("=== 2nd Round User Filtering ===")
    filtered_test_mask = np.isin(groups_test, np.unique(groups_train_filtered))
    X_test_filtered = X_test_normalized.loc[filtered_test_mask]
    y_test_filtered = y_test[filtered_test_mask]
    groups_test_filtered = groups_test[filtered_test_mask]
    filtered_test_users = int(np.unique(groups_test_filtered).size)
    filtered_test_samples = int(len(y_test_filtered))
    run_summary.update({
        "filtered_test_samples": filtered_test_samples,
        "filtered_test_users": filtered_test_users
    })
    
    common_features = X_train_filtered.columns.intersection(X_test_filtered.columns)
    X_test_filtered = X_test_filtered[common_features]
    
    if args.model == 'xgb':
        common_scores, _, _, user_imp_map, _ = evaluate_users_notebook_style(
            X_train=X_train_filtered,
            y_train=y_train_filtered,
            groups_train=groups_train_filtered,
            X_test=X_test_filtered,
            y_test=y_test_filtered,
            groups_test=groups_test_filtered,
            common_feats=common_feats,
            random_state=args.seed,
            n_jobs=-1
        )
    else:
        common_scores, _, _, user_imp_map, _ = evaluate_users_with_fixed_splits(
            X_train=X_train_filtered,
            y_train=y_train_filtered,
            groups_train=groups_train_filtered,
            X_test=X_test_filtered,
            y_test=y_test_filtered,
            groups_test=groups_test_filtered,
            common_feats=common_feats,
            numeric_cols=[],
            random_state=args.seed,
            model_name=args.model,  # Pass model name
            n_jobs=-1 if is_tree_model(args.model) else 1,
            model_params=model_params
        )
    
    anchor_users = sorted([u for u, s in common_scores.items() if s['auroc'] > args.auroc_threshold])
    print(f"Anchor users selected (AUROC > {args.auroc_threshold}): {len(anchor_users)}")
    
    print("=== Preparing Final Datasets ===")
    mask = np.isin(groups_train_filtered, anchor_users)
    X_anchor_final = X_train_filtered.loc[mask, common_feats].reset_index(drop=True)
    y_anchor_final = y_train_filtered[mask]
    groups_anchor_final = groups_train_filtered[mask]
    
    # Prepare test data for anchor users
    X_anchor_test_final = {}
    y_anchor_test_final = {}
    groups_anchor_test_final = {}
    
    for user in anchor_users:
        user_test_mask = groups_test_filtered == user
        if np.any(user_test_mask):
            user_test_data = X_test_filtered.loc[user_test_mask]
            available_common_feats = [f for f in common_feats if f in user_test_data.columns]
            
            if len(available_common_feats) > 0:
                X_anchor_test_final[user] = user_test_data[available_common_feats]
                y_anchor_test_final[user] = y_test_filtered[user_test_mask]
                groups_anchor_test_final[user] = groups_test_filtered[user_test_mask]
    
    # Save processed data
    processed = {
        "X_common": X_anchor_final,
        "y": y_anchor_final,
        "groups": groups_anchor_final,
        "X_test": X_anchor_test_final,
        "y_test": y_anchor_test_final,
        "groups_test": groups_anchor_test_final,
        "common_feats": common_feats,
        "model_name": args.model,
        "user_imp_map": {u: user_imp_map[u] for u in anchor_users},
        "scalers": {},
        "t": None,
        "datetimes": None
    }
    
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    data_path_primary = os.path.join(PROCESSED_DIR, f"{base_name}_filtered_{args.model}.pkl")
    legacy_data_path = os.path.join(PROCESSED_DIR, f"{base_name}_filtered.pkl") if args.model == 'xgb' else None
    for save_path in [data_path_primary, legacy_data_path]:
        if save_path is None:
            continue
        with open(save_path, "wb") as f:
            cloudpickle.dump(processed, f)
    
    print(f"Final dataset: {len(anchor_users)} anchor users")
    print(f"Training data: {len(y_anchor_final)} samples")
    test_samples = sum(len(y_anchor_test_final[u]) for u in anchor_users if u in y_anchor_test_final)
    print(f"Test data: {test_samples} samples")
    print(f"Dataset saved to: {data_path_primary}")
    anchor_user_count = int(len(anchor_users))
    anchor_train_samples = int(len(y_anchor_final))
    anchor_test_samples = int(test_samples)
    selected_feature_count = int(len(common_feats))
    run_summary.update({
        "anchor_users": anchor_user_count,
        "anchor_train_samples": anchor_train_samples,
        "anchor_test_samples": anchor_test_samples,
        "selected_features": selected_feature_count
    })
    
    print("=== Distance Matrix Calculation ===")
    # Use GPU if available for OTDD computations
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (GPU not available)")
    cpu_count = os.cpu_count() or 1
    if args.distance_jobs is not None:
        N_JOBS = max(1, args.distance_jobs)
    elif is_tree_model(args.model):
        # Cap at 8 processes for tree models during distance computations to reduce memory/CPU pressure
        half_cores = max(1, cpu_count // 2)
        N_JOBS = max(1, min(half_cores, 8))
    else:
        N_JOBS = 1
    distance_backend = args.distance_backend
    print(f"Distance backend: {distance_backend}, n_jobs={N_JOBS}")
    base_dir_suffix = f"{args.model}_{args.dataset}" if hasattr(args, 'dataset') and args.dataset else f"{args.model}"
    OUTPUT_DIR = os.path.join(RESULTS_DIR, f"distance_figures_{base_dir_suffix}", f"k{args.top_k}")
    OUTPUT_DIR_CACHE = os.path.join(RESULTS_DIR, f"distance_cache_{base_dir_suffix}", f"k{args.top_k}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_CACHE, exist_ok=True)

    # Clear cached distance files to stay consistent with notebook outputs
    for base_dir in (OUTPUT_DIR, OUTPUT_DIR_CACHE):
        patterns = [
            os.path.join(base_dir, '**', '*.npy'),
            os.path.join(base_dir, '**', '*.png')
        ]
        for pat in patterns:
            for fp in glob.glob(pat, recursive=True):
                try:
                    os.remove(fp)
                except OSError:
                    pass
    
    # Calculate distances for each feature category
    
    if hasattr(args, 'dataset') and args.dataset in DATASET_CONFIGS:
         category_prefixes = DATASET_CONFIGS[args.dataset]["category_prefixes"]
    else:
         # Fallback default if not specified (D-3 style)
         category_prefixes = {
             'PhoneUsage': ['APP', 'WLS', 'CHG', 'Dozemode', 'PWR', 'BAT_TMP', 'SCR', 'BAT', 'Notification', 'RING', 'Time', 'ONOFF', 'keyevent'],
             'Social': ['CALL', 'DATA_SNT', 'DATA_RCV', 'DATA_MRCV', 'DATA_MSNT', 'MSG_SNT', 'MSG_RCV', 'MSG_ALL'],
             'Physical': ['ACT', 'FCL_VAL', 'FAC_VAL', 'FDI_VAL', 'FST_VAL', 'Fitbit', 'ACE'],
             'Mobility': ['LOC', 'INST_JAC', 'WIFI_COS', 'WIFI_EUC', 'WIFI_MAN', 'WIFI_JAC'],
             'Sleep': ['sleep', 'Sleep']
         }
    
    user_imp_map_valid = {u: user_imp_map[u] for u in anchor_users}
    
    # Calculate distances for each feature category
    for cat, prefixes in category_prefixes.items():
        feats = [f for f in common_feats if any(f.startswith(pref) for pref in prefixes)]
        
        if not feats:
            continue

        # Drop category-level constant features (on anchor subset) to avoid CatBoost/DetectShift failures
        feat_mask = (X_anchor_final[feats].nunique(dropna=False) > 1)
        const_feats = [f for f in feats if not feat_mask[f]]
        feats = [f for f in feats if feat_mask[f]]
        if not feats:
            print(f"[{cat}] Skipping: all features constant on anchor users (was {len(const_feats)} features)")
            continue
        if const_feats:
            print(f"[{cat}] Dropping {len(const_feats)} constant features on anchor users (keeping {len(feats)})")

        print(f"[{cat}] Features: {len(feats)}")

        feat_indices = [common_feats.index(f) for f in feats]
        user_importances_cat = {u: user_imp_map_valid[u][feat_indices] for u in anchor_users}

        out_dir = os.path.join(OUTPUT_DIR, cat)
        cache_dir = os.path.join(OUTPUT_DIR_CACHE, cat)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        with parallel_backend(distance_backend, n_jobs=N_JOBS):
            u_cov, cov_mat = compute_pairwise_wasserstein_matrix(
                X_anchor_final[feats], groups_anchor_final, feats,
                n_jobs=N_JOBS, cache_path=os.path.join(cache_dir, 'cov_mat.npy')
            )
            assert np.array_equal(u_cov, anchor_users), f"Covariate Shift User Mismatch: {u_cov} vs {anchor_users}"
            plot_distance_matrix_and_save(cov_mat, out_dir, f"{cat} Covariate Shift", 'cov')

            u_lbl, lbl_mat = compute_pairwise_label_shift_matrix(
                y_anchor_final, groups_anchor_final,
                n_jobs=N_JOBS, cache_path=os.path.join(cache_dir, 'lbl_mat.npy')
            )
            assert np.array_equal(u_lbl, anchor_users), f"Label Shift User Mismatch in {cat}"
            plot_distance_matrix_and_save(lbl_mat, out_dir, f"{cat} Label Shift", 'label')

            # Conditional Shift Calculation
            # For Deep Models, we calculate BOTH Raw Feature Distance and Embedding Distance
            
            # 1. Raw Feature Distance (Standard)
            raw_cond_cache = os.path.join(cache_dir, 'cond_mat.npy')
            u_cond, cond_mat = compute_pairwise_conditional_matrix(
                X_anchor_final, y_anchor_final, groups_anchor_final,
                feature_list=feats, user_importances=user_importances_cat,
                device=DEVICE, n_jobs=N_JOBS, cache_path=raw_cond_cache,
                model_name='xgb', # Force Raw Feature for category-wise analysis
                model_params=model_params, random_state=args.seed
            )
            assert np.array_equal(u_cond, anchor_users), f"Conditional Shift User Mismatch in {cat}"
            plot_distance_matrix_and_save(cond_mat, out_dir, f"{cat} Conditional Shift", 'cond')

            concept_cache = os.path.join(cache_dir, 'concept_mat.npy')
            if args.concept_shift_method == 'cluster':
                u_concept, concept_mat = compute_pairwise_concept_shift_clustering(
                    X_anchor_final, groups_anchor_final, y_anchor_final, feats,
                    n_clusters=args.concept_shift_k, min_samples=args.concept_shift_min_samples,
                    n_jobs=N_JOBS, cache_path=concept_cache, random_state=args.seed
                )
            elif args.concept_shift_method == 'knn':
                u_concept, concept_mat = compute_pairwise_concept_shift_knn(
                    X_anchor_final, groups_anchor_final, y_anchor_final, feats,
                    k_neighbors=args.concept_shift_k, n_jobs=N_JOBS,
                    cache_path=concept_cache, random_state=args.seed
                )
            elif args.concept_shift_method == 'detectshift':
                u_concept, concept_mat = compute_pairwise_concept_shift_detectshift(
                    X_anchor_final, groups_anchor_final, y_anchor_final, feats,
                    test_size=args.detectshift_test_size, B=args.detectshift_B, n_jobs=1,
                    cache_path=concept_cache, random_state=args.seed
                )
            else:
                u_concept, concept_mat = compute_pairwise_concept_shift_matrix(
                    X_anchor_final, groups_anchor_final, y_anchor_final, feats,
                    n_jobs=N_JOBS, cache_path=concept_cache,
                    backend=distance_backend, random_state=args.seed
                )
            assert np.array_equal(u_concept, anchor_users), f"Concept Shift User Mismatch in {cat}"
            plot_distance_matrix_and_save(concept_mat, out_dir, f"{cat} Concept Shift", 'concept')



    # === Global Embedding Analysis (Deep Models Only) ===
    if is_deep_model(args.model):
        print(f"=== Running Global Embedding Analysis for {args.model} ===")
        print(f"Features: All {len(common_feats)} features")
        
        emb_out_dir = os.path.join(OUTPUT_DIR, "Global_Embedding")
        emb_cache_dir = os.path.join(OUTPUT_DIR_CACHE, "Global_Embedding")
        os.makedirs(emb_out_dir, exist_ok=True)
        os.makedirs(emb_cache_dir, exist_ok=True)
        
        # 1. Global Conditional Shift (Embedding OTDD)
        # This uses the Untrained Backbone Projection on ALL features
        # It represents the distance in the model's latent space
        print("[Global_Embedding] Calculating Conditional Shift...")
        u_global_cond, global_cond_mat_emb = compute_pairwise_conditional_matrix(
            X_anchor_final, y_anchor_final, groups_anchor_final,
            feature_list=common_feats, # Use ALL features
            user_importances=user_imp_map_valid, # Use full importance map
            device=DEVICE, n_jobs=N_JOBS, 
            cache_path=os.path.join(emb_cache_dir, 'cond_mat_emb.npy'),
            model_name=args.model, # Trigger Embedding Logic
            model_params=model_params, random_state=args.seed
        )
        assert np.array_equal(u_global_cond, anchor_users), f"Global Conditional Shift User Mismatch: {u_global_cond} vs {anchor_users}"
        plot_distance_matrix_and_save(global_cond_mat_emb, emb_out_dir, f"Global Conditional Shift ({args.model})", 'cond_emb')
        
        # 2. Global Covariate Shift (Embedding Wasserstein)
        # We need to project features first to calculate Wasserstein on Embeddings
        # Since 'compute_pairwise_wasserstein_matrix' expects a DataFrame, we need to adapt or do it manually.
        # But 'compute_pairwise_conditional_matrix' with lambda_y=0 essentially gives us Covariate Shift (Unsupervised).
        # Let's use that trick for consistency.
        
        print("[Global_Embedding] Calculating Covariate Shift...")
        # Hack: Call OTDD with lambda_y=0 (Label agnostic) -> Pure Feature Distance
        # utility.py's default is lambda_y=1, so we need to pass custom ot_params via a new argument or modification.
        # Since we didn't expose ot_params to the outer function, we can't easily change lambda_y.
        # Instead, we will rely on Conditional Shift (Emb) and Concept Shift (Emb - calculated below) as the main deep metrics.
        
        # 3. Global Concept Shift
        # This is already model-based (JSD of predictions), so we just run it on all features.
        print("[Global_Embedding] Calculating Concept Shift...")
        global_concept_cache = os.path.join(emb_cache_dir, 'concept_mat.npy')
        if args.concept_shift_method == 'cluster':
            u_global_concept, global_concept_mat = compute_pairwise_concept_shift_clustering(
                X_anchor_final, groups_anchor_final, y_anchor_final, common_feats,
                n_clusters=args.concept_shift_k, min_samples=args.concept_shift_min_samples,
                n_jobs=N_JOBS, cache_path=global_concept_cache, random_state=args.seed
            )
        elif args.concept_shift_method == 'knn':
            u_global_concept, global_concept_mat = compute_pairwise_concept_shift_knn(
                X_anchor_final, groups_anchor_final, y_anchor_final, common_feats,
                k_neighbors=args.concept_shift_k, n_jobs=N_JOBS,
                cache_path=global_concept_cache, random_state=args.seed
            )
        elif args.concept_shift_method == 'detectshift':
            u_global_concept, global_concept_mat = compute_pairwise_concept_shift_detectshift(
                X_anchor_final, groups_anchor_final, y_anchor_final, common_feats,
                test_size=args.detectshift_test_size, B=args.detectshift_B, n_jobs=1,
                cache_path=global_concept_cache, random_state=args.seed
            )
        else:
            u_global_concept, global_concept_mat = compute_pairwise_concept_shift_matrix(
                X_anchor_final, groups_anchor_final, y_anchor_final, common_feats,
                n_jobs=N_JOBS, 
                cache_path=global_concept_cache, 
                random_state=args.seed
            )
        assert np.array_equal(u_global_concept, anchor_users), f"Global Concept Shift User Mismatch: {u_global_concept} vs {anchor_users}"
        plot_distance_matrix_and_save(global_concept_mat, emb_out_dir, f"Global Concept Shift ({args.model})", 'concept')


    
    
    print("=== Anchor User Transferability Analysis ===")
    data_path_to_load = legacy_data_path or data_path_primary
    with open(data_path_to_load, "rb") as f:
        data = cloudpickle.load(f)

    X_common_valid = data["X_common"]
    y_valid = data["y"]
    groups_valid = data["groups"]
    X_test_valid = data["X_test"]
    y_test_valid = data["y_test"]

    valid_users = np.unique(groups_valid)
    user_to_index = {u: i for i, u in enumerate(valid_users)}

    print(f"Loaded {len(valid_users)} anchor users")
    print(f"Training data: {len(y_valid)} samples")
    print(f"Test data available for {len(X_test_valid)} users")

    categories = [
        cat for cat, prefixes in category_prefixes.items()
        if any(f for f in common_feats if any(f.startswith(pref) for pref in prefixes))
    ]

    # Create mapping from user ID to matrix index
    # validation_users matches anchor_users because we loaded the filtered dataset which contains only anchor users
    # And we know anchor_users are sorted.
    user_to_index = {u: i for i, u in enumerate(valid_users)}

    reg_results = []
    all_anchor_data = {}

    def _plot_for_anchor_cat(shift_name, mat, prefix, anchor, cat):
        """Train anchor model and compute transferability metrics.
        
        Args:
            shift_name: Name of the shift type.
            mat: Distance matrix between users.
            prefix: Prefix for output files.
            anchor: Anchor user ID.
            cat: Category name.
            
        Returns:
            tuple: (regression_record, combined_data_dict)
        """
        mask_a = (groups_valid == anchor)
        X_a_train, y_a_train = X_common_valid.loc[mask_a], y_valid[mask_a]
        
        if anchor not in X_test_valid:
            return None, {}
        if anchor not in X_test_valid:
            return None, {}
        X_a_test = X_test_valid[anchor]
        y_a_test = y_test_valid[anchor]
        
        if len(np.unique(y_a_train)) < 2 or len(np.unique(y_a_test)) < 2:
            return None, {}

        # Train anchor model - use the specified model type instead of XGBClassifier
        model_a = get_model_by_name(args.model, X_a_train.shape[1], random_state=args.seed, **model_params)
        model_a.fit(X_a_train, y_a_train, X_a_test, y_a_test)
        
        # Compute self-AUROC
        anchor_self_auc = roc_auc_score(y_a_test, model_a.predict_proba(X_a_test)[:,1])

        # Compute transfer AUROC and delta for other users
        dists, deltas = [], []
        for other in valid_users:
            if other == anchor:
                continue

            mask_o_train = (groups_valid == other)
            X_o_parts = [X_common_valid.loc[mask_o_train]]
            y_o_parts = [y_valid[mask_o_train]]

            if other in X_test_valid:
                X_o_parts.append(X_test_valid[other])
                y_o_parts.append(y_test_valid[other])

            X_o_all = pd.concat(X_o_parts, axis=0)
            y_o_all = np.concatenate(y_o_parts)

            if len(np.unique(y_o_all)) < 2:
                continue

            auc_transfer = roc_auc_score(y_o_all, model_a.predict_proba(X_o_all)[:, 1])

            dist = mat[user_to_index[anchor], user_to_index[other]]
            delta = anchor_self_auc - auc_transfer
            dists.append(dist)
            deltas.append(delta)

        if not dists:
            return None, {}

        linear_pos = LinearRegression(positive=True)
        dists_2d = np.array(dists).reshape(-1, 1)
        linear_pos.fit(dists_2d, deltas)
        slope = linear_pos.coef_[0]
        
        record = {
            'category': cat,
            'shift': shift_name,
            'anchor': anchor,
            'slope': slope
        }

        shift_folder = shift_name.replace(' ', '_').lower()
        out_dir = os.path.join(OUTPUT_DIR, cat, shift_folder)
        os.makedirs(out_dir, exist_ok=True)
        
        filename = f"{prefix}_anchor_{anchor}.png"
        filepath = os.path.join(out_dir, filename)

        plt.figure(figsize=(5,4))
        sns.scatterplot(x=dists, y=deltas, alpha=0.6)
        sns.regplot(x=dists, y=deltas, scatter=False,
                    line_kws={'linestyle':'--','alpha':0.5})
        plt.axhline(0, linestyle='--', alpha=0.3)
        plt.title(f"{shift_name} (anchor={anchor})")
        plt.xlabel(f"{shift_name} distance")
        plt.ylabel("Δ AUROC (self – transfer)")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        key = f"{cat}_{shift_name}"
        combined_data = {key: {'dists': dists.copy(), 'deltas': deltas.copy()}}
        
        return record, combined_data

    # Process each category and shift type
    
    for cat in categories:
        cache_dir = os.path.join(OUTPUT_DIR_CACHE, cat)
        out_base = os.path.join(OUTPUT_DIR, cat)
        os.makedirs(out_base, exist_ok=True)

        # Load distance matrices
        cov_mat = np.load(os.path.join(cache_dir, 'cov_mat.npy'))
        lbl_mat = np.load(os.path.join(cache_dir, 'lbl_mat.npy'))
        cond_mat = np.load(os.path.join(cache_dir, 'cond_mat.npy'))
        concept_mat = np.load(os.path.join(cache_dir, 'concept_mat.npy'))

        distance_matrices = {
            f"{cat} Covariate shift": (cov_mat, 'cov'),
            f"{cat} Label shift": (lbl_mat, 'label'),
            f"{cat} Conditional shift": (cond_mat, 'cond'),
            f"{cat} Concept shift": (concept_mat, 'concept'),
        }

        # Create tasks for parallel processing
        plot_tasks = [
            (name, mat, prefix, anchor, cat)
            for name, (mat, prefix) in distance_matrices.items()
            for anchor in valid_users
        ]

        # Add Global Analysis Tasks (Embedding and Raw) if available
        # These are stored in 'Global_Embedding' and 'Global_Raw' directories
        
        # 1. Global Embedding
        if is_deep_model(args.model):
            glob_emb_cache = os.path.join(OUTPUT_DIR_CACHE, "Global_Embedding")
            if os.path.exists(glob_emb_cache):
                try:
                    g_cond = np.load(os.path.join(glob_emb_cache, 'cond_mat_emb.npy'))
                    g_concept = np.load(os.path.join(glob_emb_cache, 'concept_mat.npy'))
                    
                    # Add to plot tasks
                    # We use 'Global_Emb' as category name
                    for anchor in valid_users:
                        plot_tasks.append( ('Global Conditional Shift (Emb)', g_cond, 'cond_emb', anchor, 'Global_Embedding') )
                        plot_tasks.append( ('Global Concept Shift', g_concept, 'concept', anchor, 'Global_Embedding') )
                except Exception as e:
                    print(f"Skipping Global Embedding plots due to missing files: {e}")



        # Execute parallel plotting and collect results
        N_JOBS_TRANSFER = 1 if not is_tree_model(args.model) else N_JOBS
        with parallel_backend('loky', n_jobs=N_JOBS_TRANSFER):
            results = Parallel()(
                delayed(_plot_for_anchor_cat)(*args)
                for args in tqdm(plot_tasks, desc=f"Plotting {cat}")
            )
            
            # Process results and collect data
            for result in results:
                if result is not None:
                    record, combined_data = result
                    if record is not None:
                        reg_results.append(record)
                    
                    # Merge combined data
                    for key, data in combined_data.items():
                        if key not in all_anchor_data:
                            all_anchor_data[key] = {'dists': [], 'deltas': []}
                        all_anchor_data[key]['dists'].extend(data['dists'])
                        all_anchor_data[key]['deltas'].extend(data['deltas'])

    # Combined User Pairs Regression Analysis
    print("=== Combined User Pairs Regression Analysis ===")
    print(f"Collected {len(all_anchor_data)} shift combinations")
    for key, data in list(all_anchor_data.items())[:3]:
        print(f"  {key}: {len(data['dists'])} points")

    for key, data in all_anchor_data.items():
        if len(data['dists']) < 10:
            continue
            
        cat_shift = key.replace('_', ' ').title()
        dists = np.array(data['dists'])
        deltas = np.array(data['deltas'])
        
        # Compute regression using LinearRegression with positive constraint
        linear_pos = LinearRegression(positive=True)
        dists_2d = dists.reshape(-1, 1)
        linear_pos.fit(dists_2d, deltas)
        slope = linear_pos.coef_[0]
        
        # Plot combined regression
        parts = key.split('_', 1)
        cat = parts[0]
        shift = parts[1].replace('_', ' ')
        
        # Create combined plots
        combined_dir = os.path.join(OUTPUT_DIR, cat, 'combined_regression')
        os.makedirs(combined_dir, exist_ok=True)
        
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=dists, y=deltas, alpha=0.6)
        sns.regplot(x=dists, y=deltas, scatter=False,
                    line_kws={'linestyle':'--', 'alpha':0.7, 'color':'red'})
        plt.axhline(0, linestyle='--', alpha=0.3)
        plt.title(f"Combined: {cat_shift} (slope={slope:.3f})")
        plt.xlabel(f"{shift} distance")
        plt.ylabel("Δ AUROC (self – transfer)")
        
        corr_coef = np.corrcoef(dists, deltas)[0, 1]
        plt.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, f"combined_{shift.replace(' ', '_')}.png"))
        plt.close()
        
        print(f"{cat_shift}: slope={slope:.4f}, correlation={corr_coef:.3f}, n_points={len(dists)}")
    
    cat_shift_csv = None
    shift_csv = None
    summary_cat_shift = pd.DataFrame()
    summary_shift = pd.DataFrame()
    df_slopes = pd.DataFrame(reg_results)
    print(f"Total regression results: {len(reg_results)}")
    
    if len(df_slopes) > 0:
        df_slopes = df_slopes.copy()

        df_slopes['shift_type'] = df_slopes['shift'].apply(extract_shift_type_from_text)
        df_slopes['dataset'] = args.dataset
        df_slopes['distances'] = "[]"
        df_slopes['deltas'] = "[]"

        combined_csv = os.path.join(OUTPUT_DIR, 'combined_results_summary.csv')
        df_slopes_out = df_slopes[['category','shift','anchor','slope','distances','deltas','dataset','shift_type']]
        df_slopes_out.to_csv(combined_csv, index=False)
        print(f"[saved] {combined_csv}")

        grp = df_slopes.groupby(['category', 'shift', 'shift_type'], dropna=False)

        summary_cat_shift = grp['slope'].agg(
            n_anchors='size',
            median_slope='median'
        ).reset_index()
        summary_cat_shift['pos_rate'] = grp['slope'].apply(lambda s: float((s > 0).mean())).values

        p_wil = grp['slope'].apply(wilcoxon_greater).reset_index(name='p_wilcox_greater')
        p_sign = grp['slope'].apply(sign_test_greater).reset_index(name='p_sign_greater')

        summary_cat_shift = (
            summary_cat_shift
            .merge(p_wil,  on=['category','shift','shift_type'], how='left')
            .merge(p_sign, on=['category','shift','shift_type'], how='left')
        )

        summary_cat_shift['q_wilcox_greater'] = fdr_bh_series(summary_cat_shift['p_wilcox_greater'])
        summary_cat_shift['q_sign_greater']   = fdr_bh_series(summary_cat_shift['p_sign_greater'])

        rows = []
        for key, d in all_anchor_data.items():
            cat, shift_full = parse_cat_and_shift_from_key(key)
            dists  = np.asarray(d.get('dists', []), float)
            deltas = np.asarray(d.get('deltas', []), float)
            if dists.size >= 3 and np.std(dists) > 0 and np.std(deltas) > 0:
                rho, _ = spearmanr(dists, deltas)
                r, _   = pearsonr(dists, deltas)
            else:
                rho, r = np.nan, np.nan
            rows.append({
                'category': cat,
                'shift': shift_full,
                'shift_type': extract_shift_type_from_text(shift_full),
                'rho_pairs': rho,
                'pearson_r_pairs': r,
                'n_pairs_all': int(dists.size),
            })
        df_pairs = pd.DataFrame(rows).drop_duplicates(subset=['category','shift','shift_type'])

        summary_cat_shift = summary_cat_shift.merge(
            df_pairs[['category','shift','shift_type','rho_pairs','pearson_r_pairs','n_pairs_all']],
            on=['category','shift','shift_type'],
            how='left'
        )

        summary_cat_shift = summary_cat_shift.sort_values(
            ['shift_type','rho_pairs','median_slope','n_anchors'],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)

        cat_shift_csv = os.path.join(OUTPUT_DIR, 'stats_summary_category_shift_auroc.csv')
        summary_cat_shift.to_csv(cat_shift_csv, index=False)
        print("=== Category × Shift: Positive association tests (AUROC, one-sided, FDR) ===")
        print(summary_cat_shift.head(50))
        print(f"[saved] {cat_shift_csv}")

        grp_shift = df_slopes.groupby('shift_type', dropna=False)

        summary_shift = grp_shift['slope'].agg(
            n_anchors='size',
            median_slope='median'
        ).reset_index()
        summary_shift['pos_rate'] = grp_shift['slope'].apply(lambda s: float((s > 0).mean())).values

        p_wil_s  = grp_shift['slope'].apply(wilcoxon_greater).reset_index(name='p_wilcox_greater')
        p_sign_s = grp_shift['slope'].apply(sign_test_greater).reset_index(name='p_sign_greater')

        summary_shift = (
            summary_shift
            .merge(p_wil_s,  on='shift_type', how='left')
            .merge(p_sign_s, on='shift_type', how='left')
        )
        summary_shift['q_wilcox_greater'] = fdr_bh_series(summary_shift['p_wilcox_greater'])
        summary_shift['q_sign_greater']   = fdr_bh_series(summary_shift['p_sign_greater'])

        tmp_rows = []
        for key, d in all_anchor_data.items():
            _, shift_full = parse_cat_and_shift_from_key(key)
            tmp_rows.append({
                'shift_type': extract_shift_type_from_text(shift_full),
                'dists': d.get('dists', []),
                'deltas': d.get('deltas', [])
            })
        df_tmp = pd.DataFrame(tmp_rows)

        def concat_lists(series):
            out = []
            for lst in series:
                out.extend(lst)
            return np.array(out, float) if len(out) else np.array([], float)

        pooled = df_tmp.groupby('shift_type', dropna=False).agg(
            dists=('dists', concat_lists),
            deltas=('deltas', concat_lists)
        ).reset_index()

        rhos, pearsons, ns = [], [], []
        for _, row in pooled.iterrows():
            dists, deltas = row['dists'], row['deltas']
            if dists.size >= 3 and np.std(dists) > 0 and np.std(deltas) > 0:
                rho, _ = spearmanr(dists, deltas)
                r, _   = pearsonr(dists, deltas)
            else:
                rho, r = np.nan, np.nan
            rhos.append(rho); pearsons.append(r); ns.append(int(dists.size))
        pooled = pooled.drop(columns=['dists','deltas'])
        pooled['rho_pairs'] = rhos
        pooled['pearson_r_pairs'] = pearsons
        pooled['n_pairs_all'] = ns

        summary_shift = summary_shift.merge(pooled, on='shift_type', how='left')

        summary_shift = summary_shift.sort_values(
            ['rho_pairs','median_slope','pos_rate','n_anchors'],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        shift_csv = os.path.join(OUTPUT_DIR, 'stats_summary_shift_auroc.csv')
        summary_shift.to_csv(shift_csv, index=False)
        print("\n=== Shift-level summary across categories (AUROC) ===")
        print(summary_shift)
        print(f"[saved] {shift_csv}")

        if len(summary_shift) > 0:
            top = summary_shift.iloc[0]
            print(f"\nStrongest shift (by ρ then slope): {top['shift_type']}")
            print(f"ρ_pairs={top['rho_pairs']:.3f}, median_slope={top['median_slope']:.4f}, "
                  f"pos_rate={top['pos_rate']:.2f}, q_wilcox={top['q_wilcox_greater']:.3g}, "
                  f"q_sign={top['q_sign_greater']:.3g}, n_pairs_all={int(top['n_pairs_all'])}")
            run_summary.update({
                "strongest_shift_type": str(top['shift_type']),
                "strongest_shift_rho": float(top['rho_pairs']) if not pd.isna(top['rho_pairs']) else float('nan'),
                "strongest_shift_median": float(top['median_slope']) if not pd.isna(top['median_slope']) else float('nan')
            })

    run_summary.update({
        "summary_cat_shift_path": cat_shift_csv or "",
        "summary_shift_path": shift_csv or "",
        "median_shift_slope": float(summary_shift['median_slope'].median()) if len(summary_shift) > 0 else float('nan'),
        "median_shift_pos_rate": float(summary_shift['pos_rate'].median()) if len(summary_shift) > 0 else float('nan'),
        "median_category_slope": float(summary_cat_shift['median_slope'].median()) if len(summary_cat_shift) > 0 else float('nan')
    })

    if len(df_slopes) > 0:
        # Match notebook spacing: map categories to fixed numeric positions
        preferred_order = ['PhoneUsage', 'Social', 'Physical', 'Mobility', 'Sleep']
        unique_cats = df_slopes['category'].unique().tolist()
        categories = sorted(unique_cats, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)
        
        if len(categories) == 5:
            cat_vals = [-2, 0, 2, 4, 6]
        elif len(categories) == 4:
            cat_vals = [-2, 0, 2, 4]
        else:
            cat_vals = [-2 + 2*i for i in range(len(categories))]
            
        cat_positions = {cat: pos for cat, pos in zip(categories, cat_vals)}
        df_slopes['category_pos'] = df_slopes['category'].map(cat_positions)
        
        plt.figure(figsize=(8,6))
        ax = sns.boxplot(
            x='category_pos', y='slope', hue='shift',
            data=df_slopes, showfliers=False, width=5,
            dodge=True
        )

        shift_colors = {'Covariate': 'C0', 'Label': 'C1', 'Conditional': 'C2', 'Concept': 'C3'}

        unique_shifts = df_slopes['shift'].unique()
        n_shifts = len(unique_shifts)

        for i, patch in enumerate(ax.patches):
            shift_idx = i % n_shifts
            if shift_idx < len(unique_shifts):
                shift_name = unique_shifts[shift_idx]
                parts = shift_name.split()
                if len(parts) >= 3:
                    shift_type = parts[-2]
                    if shift_type in shift_colors:
                        patch.set_facecolor(shift_colors[shift_type])
                        
        xticks_locs = [cat_positions[cat] for cat in categories]
        ax.set_xticks(xticks_locs)
        ax.set_xticklabels(categories)
        
        ax.margins(x=0.05)
        ax.axhline(0, linestyle='--', color='gray', alpha=0.7)

        if ax.legend_:
            ax.legend_.remove()

        import matplotlib.patches as mpatches
        legend_elements = [mpatches.Patch(color=color, label=shift_type) 
                          for shift_type, color in shift_colors.items()]
        ax.legend(handles=legend_elements, title="Shift Type", bbox_to_anchor=(1.02, 1), loc='upper left')

        ax.set_xlabel("Category")
        ax.set_ylabel("Regression coefficient")
        ax.set_title(f"Regression coefficient by Category and Shift ({args.model.upper()}-based)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_coefficient_{args.model}_positive.png'))
        plt.close()
        
        print(f"Transferability analysis completed ({args.model.upper()}-based)")
        print(f"Results saved in {OUTPUT_DIR}/ directory")

    append_run_log(run_summary)
    print(f"Run metadata appended to {log_path}")

    print("=== Pipeline Completed Successfully ===")
    print(f"Model: {args.model}")
    print(f"Anchor users: {len(anchor_users)}")
    print(f"Features: {len(common_feats)}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
