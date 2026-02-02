"""
Deep Learning Enhanced Utility Functions

This module extends the base utility functions with deep learning model support.
Organized to match the structure of utility.py with additional deep learning capabilities.

Key Features:
- GPU-accelerated deep learning models
- Parallel computing for tree-based models  
- Standardized model parameters for fair comparison
- Comprehensive evaluation pipeline
"""

import os
import time
import warnings
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

# Suppress OT GPU warnings
warnings.filterwarnings("ignore", message=".*ot.gpu not found.*")
warnings.filterwarnings("ignore", message=".*coupling computation will be in cpu.*")

# Suppress pandas deprecation warnings from XGBoost and other libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is_sparse.*")
warnings.filterwarnings("ignore", message=".*is_categorical_dtype.*")
warnings.filterwarnings("ignore", message=".*SparseDtype.*")
warnings.filterwarnings("ignore", message=".*CategoricalDtype.*")

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import ray
import seaborn as sns
import shap
import scipy.stats as st
import torch
from torch.utils.data import DataLoader, TensorDataset

from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from otdd.pytorch.distance import DatasetDistance
from scipy.spatial.distance import jensenshannon
from scipy import stats
from scipy.stats import wilcoxon, binomtest, spearmanr, pearsonr

from splitters import (
    SplitMetadata,
    loso_split_indices,
    stratified_group_kfold_indices,
)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

# Import our organized models
from models import (
    XGBModel, LGBMModel, CatBoostModel,  # Tree-based (parallel)
    MLPModel, TabTransformerModel, FTTransformerModel, NODEModel, TabResNetModel  # Deep (GPU)
)
from utils.io import load, dump, log, summary
from utils.models import get_model_by_name, is_deep_model, is_tree_model
from utils.splits import create_train_test_splits
from utils.stats import wilcoxon_greater, sign_test_greater, fdr_bh_series

__all__ = [
    # Core utilities (inherited from utility.py)
    "load", "dump", "log", "summary",
    
    # Model evaluation functions
    "filter_valid_users_deep",
    
    # Visualization and analysis
    "save_correlation_heatmaps",
    
    # Distance computations
    "compute_pairwise_wasserstein_matrix",
    "compute_pairwise_label_shift_matrix", 
    "compute_pairwise_conditional_matrix",
    "compute_pairwise_concept_shift_matrix",
    "compute_pairwise_concept_shift_clustering",
    "compute_pairwise_concept_shift_knn",
    "compute_pairwise_concept_shift_detectshift",
    "plot_distance_matrix_and_save",
    
    # Data preprocessing
    "create_train_test_splits",
    "normalize_data",
    "evaluate_users_with_fixed_splits",
    "select_top_model_features",
    "select_top_xgb_features",
    
    # Statistical tests
    "wilcoxon_greater",
    "sign_test_greater", 
    "fdr_bh_series",
    
    # Model utilities
    "get_model_by_name",
    "is_deep_model",
    "is_tree_model",
    "loso_split_indices",
    "stratified_group_kfold_indices",
]

# Constants (inherited from utility.py)
DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

# Repo-local path defaults (override via environment if needed)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(REPO_ROOT, "data", "raw", "D-1")
PATH_DATA = os.environ.get("CHI_RAW_DATA_DIR", RAW_ROOT)
PATH_ESM = os.path.join(PATH_DATA, "EsmResponse.csv")
PATH_PARTICIPANT = os.path.join(PATH_DATA, "UserInfo.csv")
PATH_SENSOR = os.path.join(PATH_DATA, "Sensor")
PATH_RESULTS = os.path.join(REPO_ROOT, "results")
PATH_INTERMEDIATE = os.path.join(REPO_ROOT, "data", "Archived")
RANDOM_STATE = 42

seed = RANDOM_STATE

# Data type mappings (inherited)
DATA_TYPES = {
    'Acceleration': 'ACC', 'AmbientLight': 'AML', 'Calorie': 'CAL',
    'Distance': 'DST', 'EDA': 'EDA', 'HR': 'HRT', 'RRI': 'RRI',
    'SkinTemperature': 'SKT', 'StepCount': 'STP', 'UltraViolet': 'ULV',
    'ActivityEvent': 'ACE', 'ActivityTransition': 'ACT', 'AppUsageEvent': 'APP',
    'BatteryEvent': 'BAT', 'CallEvent': 'CAE', 'Connectivity': 'CON',
    'DataTraffic': 'DAT', 'InstalledApp': 'INS', 'Location': 'LOC',
    'MediaEvent': 'MED', 'MessageEvent': 'MSG', 'WiFi': 'WIF',
    'ScreenEvent': 'SCR', 'RingerModeEvent': 'RNG', 'ChargeEvent': 'CHG',
    'PowerSaveEvent': 'PWS', 'OnOffEvent': 'ONF'
}

# App category mappings (inherited)
transform = {
    'GAME': 'ENTER', 'GAME_TRIVIA': 'ENTER', 'GAME_CASINO': 'ENTER',
    'GAME-ACTION': 'ENTER', 'GAME_SPORTS': 'ENTER', 'GAME_PUZZLE': 'ENTER',
    'GAME_SIMULATION': 'ENTER', 'GAME_STRATEGY': 'ENTER', 'GAME_ROLE_PLAYING': 'ENTER',
    'GAME_ACTION': 'ENTER', 'GAME_ARCADE': 'ENTER', 'GAME_RACING': 'ENTER',
    'GAME_CASUAL': 'ENTER', 'GAME_MUSIC': 'ENTER', 'GAME_CARD': 'ENTER',
    'GAME_ADVENTURE': 'ENTER', 'GAME_BOARD': 'ENTER', 'GAME_EDUCATIONAL': 'ENTER',
    'PHOTOGRAPHY': 'ENTER', 'ENTERTAINMENT': 'ENTER', 'SPORTS': 'ENTER',
    'MUSIC_AND_AUDIO': 'ENTER', 'COMICS': 'ENTER', 'VIDEO_PLAYERS_AND_EDITORS': 'ENTER',
    'VIDEO_PLAYERS': 'ENTER', 'ART_AND_DESIGN': 'ENTER',
    'TRAVEL_AND_LOCAL': 'INFO', 'FOOD_AND_DRINK': 'INFO', 'NEWS_AND_MAGAZINES': 'INFO',
    'MAPS_AND_NAVIGATION': 'INFO', 'WEATHER': 'INFO', 'HOUSE_AND_HOME': 'INFO',
    'BOOKS_AND_REFERENCE': 'INFO', 'SHOPPING': 'INFO', 'LIBRARIES_AND_DEMO': 'INFO',
    'BEAUTY': 'INFO', 'AUTO_AND_VEHICLES': 'INFO', 'LIFESTYLE': 'INFO',
    'PERSONALIZATION': 'SYSTEM', 'TOOLS': 'SYSTEM', 'SYSTEM': 'SYSTEM', 'MISC': 'SYSTEM',
    'COMMUNICATION': 'SOCIAL', 'SOCIAL': 'SOCIAL', 'DATING': 'SOCIAL', 'PARENTING': 'SOCIAL',
    'FINANCE': 'WORK', 'BUSINESS': 'WORK', 'PRODUCTIVITY': 'WORK', 'EDUCATION': 'WORK',
    'HEALTH_AND_FITNESS': 'HEALTH', 'MEDICAL': 'HEALTH',
    None: 'UNKNOWN', 'UNKNOWN': 'UNKNOWN'
}

# ===============================================================================
# Core Utility Functions (from utility.py)
# ===============================================================================

@contextmanager
def on_ray(*args, **kwargs):
    """Context manager for Ray initialization and cleanup."""
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(*args, **kwargs)
        yield None
    finally:
        ray.shutdown()

# ===============================================================================
# Model Management Functions
# ===============================================================================

# ===============================================================================
# Enhanced Evaluation Functions
# ===============================================================================

def filter_valid_users_deep(
    all_users: List[Any],
    X_train_normalized: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test_normalized: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    categorical_cols: pd.Series,
    t: np.ndarray,
    datetimes: np.ndarray,
    model_name: str = 'xgb',
    pra_threshold: float = 0.7,
    random_state: int = 42,
    model_params: dict = None
) -> Tuple[List, pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter users using specified model with train/test splits.
    
    Enhanced version that supports both tree-based and deep learning models.
    """
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

        # Get model instance
        params = model_params or {}
        model = get_model_by_name(model_name, X_u_train.shape[1], random_state=random_state, **params)
        
        # Train with validation
        model.fit(X_u_train, y_u_train, X_u_test, y_u_test)
        proba = model.predict_proba(X_u_test)[:, 1]
        prauc = average_precision_score(y_u_test, proba)

        if prauc >= pra_threshold:
            return user
        else:
            log(f"User {user} removed with {model_name} (PRAUC = {prauc:.3f})")
            return None

    # Use appropriate parallelization based on model type
    common_users = np.intersect1d(groups_train, groups_test)
    
    if is_deep_model(model_name):
        # Deep models: Use GPU, no joblib parallelization to avoid conflicts
        log(f"Filtering users with {model_name} using GPU (sequential processing)")
        results = [check_user(u) for u in common_users]
    elif is_tree_model(model_name):
        # Tree models: Use parallel processing, let model handle internal parallelization
        backend = 'threading'
        n_jobs = -1
        log(f"Filtering users with {model_name} using {backend} backend, n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=1)(
            delayed(check_user)(u) for u in common_users
        )
    else:
        # Default fallback
        log(f"Filtering users with {model_name} using sequential processing (unknown model type)")
        results = [check_user(u) for u in common_users]
    valid_users = [u for u in results if u is not None]

    # Filter datasets
    mask = np.isin(groups_train, valid_users)
    X_filt = X_train_normalized.loc[mask].reset_index(drop=True)
    
    if categorical_cols.empty:
        cat_filt = pd.Series(dtype=object)
    else:
        cat_filt = categorical_cols.loc[mask].reset_index(drop=True) if len(categorical_cols) > 0 else pd.Series(dtype=object)
        
    y_filt = y_train[mask]
    grp_filt = groups_train[mask]
    t_filt = t[mask] if t is not None else None
    dt_filt = datetimes[mask] if datetimes is not None else None

    log(f"Remaining users: {len(valid_users)}, samples: {X_filt.shape[0]}")
    return valid_users, X_filt, cat_filt, y_filt, grp_filt, t_filt, dt_filt

# ===============================================================================
# Inherited Functions from utility.py (with enhanced documentation)
# ===============================================================================

# Import all the distance computation, data preprocessing, and statistical test functions
# from the original utility.py. These functions remain the same but can now work with
# the enhanced model evaluation pipeline.

def compute_pairwise_wasserstein_matrix(data, groups, feats, n_jobs=1, cache_path=None):
    """Compute pairwise Wasserstein distance matrix between users."""
    def _wass_pair(i, j, data, groups, feats):
        xi = data.loc[groups == i, feats]
        xj = data.loc[groups == j, feats]
        dists = [wasserstein_distance(xi[col], xj[col]) for col in feats]
        return np.mean(dists)
    
    user_ids = np.unique(groups)
    n = len(user_ids)
    mat = np.zeros((n, n))
    
    # Use parallel processing for distance computations
    results = Parallel(n_jobs=n_jobs)(
        delayed(_wass_pair)(ui, uj, data, groups, feats)
        for idx, ui in enumerate(user_ids)
        for uj in user_ids[idx+1:]
    )
    
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = results[k]
            k += 1
            
    if cache_path:
        np.save(cache_path, mat)
    return user_ids, mat

def compute_pairwise_label_shift_matrix(labels, groups, n_jobs=1, cache_path=None):
    """Compute pairwise label distribution shift matrix between users."""
    user_ids = np.unique(groups)
    max_label = labels.max()
    probs = {}
    
    for u in user_ids:
        y_u = labels[groups == u]
        counts = np.bincount(y_u, minlength=max_label+1)
        probs[u] = counts / counts.sum()
    
    n = len(user_ids)
    mat = np.zeros((n, n))
    
    for i, ui in enumerate(user_ids):
        for j in range(i+1, n):
            uj = user_ids[j]
            mat[i, j] = mat[j, i] = np.linalg.norm(probs[ui] - probs[uj])
    
    if cache_path:
        np.save(cache_path, mat)
    return user_ids, mat

# Statistical test functions (inherited from utility.py; imported from utils.stats)

# ===============================================================================
# Additional utility functions inherited from utility.py
# ===============================================================================

def normalize_data(
    X_train: pd.DataFrame,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    groups_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize features using user-wise normalization."""
    X_numeric_train = X_train[numeric_cols]
    train_vars = X_numeric_train.var()
    valid_features = train_vars[train_vars > 0].index.tolist()
    
    user_ids = np.unique(groups_train)
    train_normalized_parts = []
    test_normalized_parts = []
    
    for user in user_ids:
        user_train_mask = (groups_train == user)
        X_user_train = X_train.loc[user_train_mask, valid_features]
        
        if len(X_user_train) == 0:
            continue
            
        scaler = StandardScaler()
        X_user_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_user_train),
            columns=X_user_train.columns,
            index=X_user_train.index
        )
        train_normalized_parts.append(X_user_train_scaled)
        
        user_test_mask = (groups_test == user)
        X_user_test = X_test.loc[user_test_mask]
        
        if len(X_user_test) == 0:
            continue
            
        common_features = [f for f in valid_features if f in X_user_test.columns]
        
        if len(common_features) == 0:
            continue
            
        X_user_test_common = X_user_test[common_features].copy()
        
        common_numeric_cols = [col for col in common_features if col in valid_features]
        
        if len(common_numeric_cols) > 0:
            user_train_numeric = X_user_train_scaled[common_numeric_cols]
            user_test_numeric = X_user_test_common[common_numeric_cols]
            
            X_user_test_common[common_numeric_cols] = scaler.transform(user_test_numeric)
        
        test_normalized_parts.append(X_user_test_common)
    
    X_train_normalized = pd.concat(train_normalized_parts)
    X_test_normalized = pd.concat(test_normalized_parts) if test_normalized_parts else pd.DataFrame()
    
    if not categorical_cols.empty:
        cat_train_mask = categorical_cols.index.isin(X_train.index)
        cat_filtered = categorical_cols.loc[cat_train_mask]
        
        X_train_final = pd.concat([
            X_train_normalized.reset_index(drop=True),
            cat_filtered.reset_index(drop=True)
        ], axis=1)
        
        if not X_test_normalized.empty:
            cat_test_mask = categorical_cols.index.isin(X_test.index)
            cat_test_filtered = categorical_cols.loc[cat_test_mask]
            
            X_test_final = pd.concat([
                X_test_normalized.reset_index(drop=True),
                cat_test_filtered.reset_index(drop=True)
            ], axis=1)
        else:
            X_test_final = pd.DataFrame()
    else:
        X_train_final = X_train_normalized.reset_index(drop=True)
        X_test_final = X_test_normalized.reset_index(drop=True) if not X_test_normalized.empty else pd.DataFrame()
    
    problematic_chars = ['[', ']', '<', '>', '{', '}', '(', ')', ',']
    
    cols_to_drop_train = []
    for col in X_train_final.columns:
        if any(char in str(col) for char in problematic_chars):
            cols_to_drop_train.append(col)
    
    if cols_to_drop_train:
        print(f"Dropping {len(cols_to_drop_train)} columns with problematic characters from training data: {cols_to_drop_train[:5]}...")
        X_train_final = X_train_final.drop(columns=cols_to_drop_train)
    
    if not X_test_final.empty:
        cols_to_drop_test = []
        for col in X_test_final.columns:
            if any(char in str(col) for char in problematic_chars):
                cols_to_drop_test.append(col)
        
        if cols_to_drop_test:
            print(f"Dropping {len(cols_to_drop_test)} columns with problematic characters from test data: {cols_to_drop_test[:5]}...")
            X_test_final = X_test_final.drop(columns=cols_to_drop_test)
    
    print(f"Normalized features: {len(valid_features)}")
    print(f"Final normalized training shape: {X_train_final.shape}")
    print(f"Final normalized test shape: {X_test_final.shape}")
    
    return X_train_final, X_test_final

def evaluate_users_with_fixed_splits(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    common_feats: List[str],
    numeric_cols: List[str],
    model_name: str = 'xgb',
    random_state: int = 42,
    n_jobs: int = -1,
    model_params: dict = None
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Evaluate users using pre-split train/test data."""
    from sklearn.preprocessing import StandardScaler
    
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
                
        except Exception as e:
            X_test_final = X_test_common
        
        problematic_chars = ['[', ']', '<', '>', '{', '}', '(', ')', ',']
        cols_to_drop = []
        for col in X_train_final.columns:
            if any(char in str(col) for char in problematic_chars):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            X_train_final = X_train_final.drop(columns=cols_to_drop)
            X_test_final = X_test_final.drop(columns=cols_to_drop)
        
        available_common_feats = [f for f in common_feats if f in X_train_final.columns]
        if not available_common_feats:
            return None
            
        X_train_common = X_train_final[available_common_feats]
        X_test_common = X_test_final[available_common_feats]
        
        def fit_and_predict(X_tr, y_tr, X_te):
            params = model_params or {}
            model = get_model_by_name(model_name, X_tr.shape[1], random_state=random_state, **params)
            model.fit(X_tr, y_tr, X_te, y_test_user)
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

        # Base importances from model (fallback to uniform if missing)
        if m_c.feature_importances_ is not None:
            imp = m_c.feature_importances_
        else:
            imp = np.ones(len(available_common_feats)) / len(available_common_feats)
        
        # SHAP: only attempt TreeExplainer for tree models; otherwise reuse base importances
        if is_tree_model(model_name):
            try:
                base_model = m_c.model if hasattr(m_c, 'model') else m_c
                explainer = shap.TreeExplainer(base_model)
                shap_vals = explainer.shap_values(
                    X_train_common.values if hasattr(X_train_common, 'values') else X_train_common
                )
                shap_imp = np.abs(shap_vals).mean(axis=0)
            except Exception as e:
                shap_imp = imp
        else:
            shap_imp = imp

        corr = X_train_common.corr()

        norm_imp = imp / imp.sum() if imp.sum() > 0 else np.ones_like(imp) / len(imp)
        norm_shap = shap_imp / shap_imp.sum() if shap_imp.sum() > 0 else np.ones_like(shap_imp) / len(shap_imp)

        return user, sc_c, sc_f, norm_imp, norm_shap, corr

    train_users = set(np.unique(groups_train))
    test_users = set(np.unique(groups_test))
    common_users = list(train_users.intersection(test_users))
    
    print(f"Evaluating {len(common_users)} users with consistent train/test splits using {model_name}")
    
    # Use appropriate parallelization based on model type
    if is_deep_model(model_name):
        # Deep models: Use GPU, no joblib parallelization to avoid conflicts
        print(f"Using {model_name} with GPU (sequential processing)")
        results = [_process_user(u) for u in common_users]
    elif is_tree_model(model_name):
        # Tree models: Use parallel processing, let model handle internal parallelization
        backend = 'threading'
        n_jobs = -1
        print(f"Using {model_name} with {backend} backend, n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_process_user)(u) for u in common_users
        )
    else:
        # Default fallback
        print(f"Using {model_name} with sequential processing (unknown model type)")
        results = [_process_user(u) for u in common_users]

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

def select_top_model_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    top_k: int = 10,
    random_state=42,
    model_params: dict = None
) -> List[str]:
    """Select top features using the specified model."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train must contain at least two classes")
    if len(np.unique(y_test)) < 2:
        raise ValueError("y_test must contain at least two classes")

    # Get model instance
    params = model_params or {}
    model = get_model_by_name(model_name, X_train.shape[1], random_state=random_state, **params)
    
    # Train model with validation
    model.fit(X_train, y_train, X_test, y_test)
    
    # Get feature importances - calculate if not already done
    if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
        importances = model.feature_importances_
    elif hasattr(model, 'calc_feature_importance'):
        # For deep models with separate importance calculation
        importances = model.calc_feature_importance(X_train, y_train, max_samples=1000)
    else:
        # Fallback to permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            model, X_train, y_train, n_repeats=5, random_state=random_state
        )
        importances = perm_importance.importances_mean
    
    # Get top features
    top_indices = np.argsort(importances)[-top_k:]
    return X_train.columns[top_indices].tolist()

def save_correlation_heatmaps(
    corr_maps: Dict[Any, pd.DataFrame],
    common_feats: List[str],
    feature_imps: Dict[Any, np.ndarray],
    shap_imps: Dict[Any, np.ndarray],
    out_dir: str = 'correlation_heatmaps'
):
    """Generate and save heatmap PNG files for each user based on correlation matrices."""
    os.makedirs(out_dir, exist_ok=True)
    for user, corr in corr_maps.items():
        labels = [
            f"{feat}\n({g:.2f}/{s:.2f})"
            for feat, g, s in zip(common_feats,
                                   feature_imps[user],
                                   shap_imps[user])
        ]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr, cmap='coolwarm', center=0,
            annot=True, fmt='.2f',
            xticklabels=labels, yticklabels=labels
        )
        plt.title(f'Feature Corr Heatmap (User {user})')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Correlation_{user}.png")
        plt.close()

def compute_pairwise_conditional_matrix(
    features: pd.DataFrame,
    labels: np.ndarray,
    group_indices: np.ndarray,
    feature_list: List[str],
    user_importances: Dict[Any, np.ndarray],
    device: str = "cpu",
    n_jobs: int = -1,
    cache_path: str = "otdd_w.npy",
    model_name: str = 'xgb',
    model_params: dict = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise OTDD distances between users and return user IDs with the distance matrix.
    Results are cached to disk to avoid recomputation.

    Args:
        features: Dataframe of all user features.
        labels: Array of all labels corresponding to features.
        group_indices: Array of user identifiers for each row in features.
        feature_list: List of features to include.
        user_importances: Mapping from user ID to feature importances array.
        device: PyTorch device for OTDD computation.
        n_jobs: Number of parallel jobs for distance computation.
        cache_path: File path for caching the distance matrix.
        model_name: Name of the model (default: 'xgb'). If deep model, uses embedding projection.
        model_params: Dictionary of model parameters for embedding generation.
    """
    
    def calculate_weighted_otdd_distance(
        features_u: np.ndarray,
        labels_u: np.ndarray,
        features_v: np.ndarray,
        labels_v: np.ndarray,
        importances_u: np.ndarray,
        importances_v: np.ndarray,
        ot_params: Dict[str, Any]
    ) -> float:
        """Calculate the OTDD distance between two user datasets with weighted features."""
        # Use simple unweighted average if sum is 0
        weights = importances_u
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / (len(weights) * 4)
            
        # Apply weights as scaling factor to features
        # sqrt(w) because OT uses L2 distance: ||sqrt(w)x - sqrt(w)y||^2 = w||x-y||^2
        scale = np.sqrt(weights)[None, :]
        Xu = (features_u * scale).astype(np.float32)
        Xv = (features_v * scale).astype(np.float32)
        
        # Add small jitter to prevent numerical instability
        Xu += np.random.normal(0, 1e-3, Xu.shape)
        Xv += np.random.normal(0, 1e-3, Xv.shape)
        
        yu = labels_u.astype(np.int64)
        yv = labels_v.astype(np.int64)
        
        # Move tensors to the specified device
        device_obj = torch.device(ot_params.get('device', 'cpu'))
        ds_u = TensorDataset(torch.from_numpy(Xu).to(device_obj), torch.from_numpy(yu).to(device_obj))
        ds_v = TensorDataset(torch.from_numpy(Xv).to(device_obj), torch.from_numpy(yv).to(device_obj))
        
        loader_u = DataLoader(ds_u, batch_size=len(Xu), shuffle=False)
        loader_v = DataLoader(ds_v, batch_size=len(Xv), shuffle=False)
        
        try:
            ot = DatasetDistance(loader_u, loader_v, **ot_params)
            dist = ot.distance()
            # Handle both tensor and scalar returns
            if isinstance(dist, torch.Tensor):
                return dist.cpu().item()
            return float(dist)
        except Exception as e:
            # If GPU fails, try CPU fallback
            if device_obj.type == 'cuda':
                try:
                    cpu_device = torch.device('cpu')
                    ot_params_cpu = ot_params.copy()
                    ot_params_cpu['device'] = 'cpu'
                    ds_u_cpu = TensorDataset(torch.from_numpy(Xu), torch.from_numpy(yu))
                    ds_v_cpu = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
                    loader_u_cpu = DataLoader(ds_u_cpu, batch_size=len(Xu), shuffle=False)
                    loader_v_cpu = DataLoader(ds_v_cpu, batch_size=len(Xv), shuffle=False)
                    ot = DatasetDistance(loader_u_cpu, loader_v_cpu, **ot_params_cpu)
                    dist = ot.distance()
                    if isinstance(dist, torch.Tensor):
                        return dist.cpu().item()
                    return float(dist)
                except:
                    pass
            print(f"OTDD Error: {e}")
            return 10.0 # High distance for failure

    def _compute_otdd_for_pair(
        i: int,
        j: int,
        user_ids: np.ndarray,
        data: Dict[Any, Tuple[np.ndarray, np.ndarray]],
        feature_list: List[str],
        user_importances: Dict[Any, np.ndarray],
        ot_params: Dict[str, Any],
        model_name: str = 'xgb',
        model_params: Dict = None,
        random_state: int = 42
    ) -> Tuple[int, int, float]:
        """Compute OTDD distance for a single pair of users."""
        from utility_deep import is_deep_model, get_model_by_name
        
        u_id = user_ids[i]
        v_id = user_ids[j]
        Xu, yu = data[u_id]
        Xv, yv = data[v_id]
        
        # Logic for Deep Models: Project features using Randomly Initialized Backbone
        if is_deep_model(model_name):
            try:
                # Initialize untrained model
                model = get_model_by_name(model_name, Xu.shape[1], random_state=random_state, **(model_params or {}))
                
                # Get embeddings - this projects Xu/Xv into latent space Z
                Xu = model.get_embedding(Xu)
                Xv = model.get_embedding(Xv)
                
                # For embeddings, assume uniform importance (or dimensionality based)
                imp_u = np.ones(Xu.shape[1]) / Xu.shape[1]
                imp_v = np.ones(Xv.shape[1]) / Xv.shape[1]
            except Exception as e:
                # Fallback to feature space if embedding fails
                # print(f"Embedding failed: {e}")
                imp_u = user_importances.get(u_id, np.ones(len(feature_list)) / len(feature_list))
                imp_v = user_importances.get(v_id, np.ones(len(feature_list)) / len(feature_list))
        else:
            # Standard raw feature space + weighted importance
            imp_u = user_importances.get(u_id, np.ones(len(feature_list)) / len(feature_list))
            imp_v = user_importances.get(v_id, np.ones(len(feature_list)) / len(feature_list))

        dist = calculate_weighted_otdd_distance(
            Xu, yu, Xv, yv, imp_u, imp_v, ot_params
        )
        return i, j, dist

    user_ids = np.unique(group_indices)
    if os.path.exists(cache_path):
        try:
            D = np.load(cache_path)
            if D.shape == (len(user_ids), len(user_ids)):
                return user_ids, D
        except:
            pass

    data = {
        u: (
            features.loc[group_indices == u, feature_list].values,
            labels[group_indices == u]
        )
        for u in user_ids
    }

    # Parameters specifically for Conditional Shift (P(X|Y)) analysis
    ot_params = {
        "inner_ot_method": "exact",
        "debiased_loss": True,
        "p": 2,
        "λ_x": 0, # Ignore feature distance (X) themselves, focus on mapping
        "λ_y": 1, # Enforce label matching (Y) -> P(X|Y)
        "entreg": 1e-5,
        "device": device
    }

    pairs = [(i, j) for i in range(len(user_ids)) for j in range(i + 1, len(user_ids))]
    
    # Use loky backend for better compatibility with deep learning libraries if needed, 
    # but threading is often faster for pure numpy/torch CPU work if GIL isn't bottleneck.
    # Here we stick to default or joblib's preference.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_otdd_for_pair)(
            i, j, user_ids, data, feature_list, user_importances, 
            ot_params, model_name, model_params, random_state
        )
        for i, j in pairs
    )

    U = len(user_ids)
    D = np.zeros((U, U), dtype=float)
    for i, j, dist in results:
        D[i, j] = D[j, i] = dist

    if np.isnan(D).any():
        max_val = np.nanmax(D)
        D[np.isnan(D)] = max_val * 1.5 # Penalize missing values
        
    np.save(cache_path, D)
    return user_ids, D



def plot_distance_matrix_and_save(mat, out_dir, title, fname_prefix, users=None):
    """Generate and save distance matrix visualizations."""
    flat = mat[np.triu_indices_from(mat, k=1)]
    plt.figure(figsize=(6, 4))
    sns.histplot(flat, bins=50, kde=True)
    plt.title(f"{title} Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_prefix}_hist.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, square=True, cbar_kws={'label':'Distance'}, cmap="viridis")
    plt.title(f"{title} Matrix")
    plt.xlabel("User Index")
    plt.ylabel("User Index")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_prefix}_matrix.png")
    plt.close()

# ===============================================================================
# Transferred Functions from utility.py (Consolidation)
# ===============================================================================

def _train_single_model(X, y, random_state=42):
    """Train a single XGBoost model.
    
    Args:
        X: Features.
        y: Labels.
        random_state: Random seed.
    
    Returns:
        Trained model.
    """
    model = XGBClassifier(
        n_estimators=100,  # Reduced estimators for speed as we train many models
        max_depth=3,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=1,  # Sequential inside parallel worker
        random_state=random_state,
        gpu_id=-1
    )
    model.fit(X, y)
    return model

def _compute_concept_shift_pair(i, j, user_ids, data, feats, random_state=42):
    """Compute concept shift by comparing predictions of two user-specific models on the union of their data.
    
    Method:
    1. Train Model_i on User_i data.
    2. Train Model_j on User_j data.
    3. Create Validation Set X_val = User_i_X U User_j_X.
    4. Predict P_i = Model_i(X_val) and P_j = Model_j(X_val).
    5. Compute Jensen-Shannon Divergence between P_i and P_j.
    """
    ui, uj = user_ids[i], user_ids[j]
    
    try:
        # 1. Prepare Data
        Xi = data.loc[data.index.get_level_values(0) == ui, feats].values
        yi = data.loc[data.index.get_level_values(0) == ui, 'label'].values
        Xj = data.loc[data.index.get_level_values(0) == uj, feats].values
        yj = data.loc[data.index.get_level_values(0) == uj, 'label'].values
        
        # Check sufficient data
        if len(Xi) < 10 or len(Xj) < 10 or len(np.unique(yi)) < 2 or len(np.unique(yj)) < 2:
            return 1.0  # Max divergence if data insufficient
        
        # 2. Train Models separately
        model_i = _train_single_model(Xi, yi, random_state)
        model_j = _train_single_model(Xj, yj, random_state)
        
        # 3. Create Union Validation Set (X only)
        X_val = np.vstack([Xi, Xj])
        
        # 4. Predict
        # Get probability of positive class (class 1)
        prob_i = model_i.predict_proba(X_val)[:, 1]
        prob_j = model_j.predict_proba(X_val)[:, 1]
        
        # 5. Compute JSD
        def safe_kl(p, q):
            # p, q are arrays of shape (N,) - probability of class 1
            epsilon = 1e-10
            p = np.clip(p, epsilon, 1 - epsilon)
            q = np.clip(q, epsilon, 1 - epsilon)
            
            # KL divergence for binary variable: p*log(p/q) + (1-p)*log((1-p)/(1-q))
            return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

        m = 0.5 * (prob_i + prob_j)
        jsd_pointwise = 0.5 * safe_kl(prob_i, m) + 0.5 * safe_kl(prob_j, m)
        return np.mean(jsd_pointwise)
        
    except Exception as e:
        # print(f"Error computing concept shift for users {ui}-{uj}: {e}")
        return 1.0

def compute_pairwise_concept_shift_matrix(
    data,
    groups,
    labels,
    feats,
    n_jobs=1,
    cache_path=None,
    backend="loky",
    random_state=42
):
    """Compute pairwise concept shift matrix using Jensen-Shannon divergence on P(Y|X).
    
    Args:
        data: DataFrame with features.
        groups: Array of user identifiers.
        labels: Array of labels.
        feats: List of feature names to use.
        n_jobs (int): Number of parallel jobs.
        cache_path: Path to cache results.
        backend: Joblib backend ('loky' or 'threading').
        random_state: Random seed.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: User IDs and concept shift matrix.
    """
    if cache_path and os.path.exists(cache_path):
        user_ids = np.unique(groups)
        return user_ids, np.load(cache_path)
    
    user_ids = np.unique(groups)
    n = len(user_ids)
    
    combined_data = data[feats].copy()
    combined_data['label'] = labels
    combined_data.index = pd.MultiIndex.from_arrays([groups, combined_data.index], names=['user', 'sample'])
    
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_compute_concept_shift_pair)(i, j, user_ids, combined_data, feats, random_state)
        for i in range(n)
        for j in range(i+1, n)
    )
    
    mat = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = results[k]
            k += 1
    
    if cache_path:
        np.save(cache_path, mat)
    
    return user_ids, mat

def _compute_concept_shift_clustering_pair(i, j, user_ids, data, feats, n_clusters=20, min_samples=5, random_state=42):
    """
    Compute concept shift using Clustering-based adaptive binning.
    
    1. Cluster the Union Data (User i + User j) into K clusters.
    2. For each cluster C_k:
       - Calculate P(Y=1 | k, User i)
       - Calculate P(Y=1 | k, User j)
    3. Compute weighted divergence (RMSE or JSD).
    """
    ui, uj = user_ids[i], user_ids[j]
    
    try:
        Xi = data.loc[data.index.get_level_values(0) == ui, feats].values
        yi = data.loc[data.index.get_level_values(0) == ui, 'label'].values
        Xj = data.loc[data.index.get_level_values(0) == uj, feats].values
        yj = data.loc[data.index.get_level_values(0) == uj, 'label'].values
        
        if len(Xi) < 20 or len(Xj) < 20: 
            return 1.0
            
        # Combine for clustering
        X_all = np.vstack([Xi, Xj])
        
        # Train KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(X_all)//10), random_state=random_state, n_init='auto')
        kmeans.fit(X_all)
        
        # Get labels
        cli = kmeans.predict(Xi)
        clj = kmeans.predict(Xj)
        
        diffs = []
        weights = []
        
        # Iterate over clusters
        unique_clusters = np.unique(kmeans.labels_)
        
        for k in unique_clusters:
            # Mask for current cluster
            mask_i = (cli == k)
            mask_j = (clj == k)
            
            # Count samples in this cluster
            ni = np.sum(mask_i)
            nj = np.sum(mask_j)
            
            # Weight by total volume in this region (P(x in C_k))
            w = (ni / len(Xi) + nj / len(Xj)) / 2
            
            if ni < min_samples or nj < min_samples:
                continue
            
            pi = np.mean(yi[mask_i])
            pj = np.mean(yj[mask_j])
            
            # Metric: Squared Difference or JSD (Bernoulli)
            # Let's use Absolute Difference for interpretability
            d = abs(pi - pj)
            
            diffs.append(d)
            weights.append(w)
            
        if sum(weights) == 0:
            return 1.0 # Failed to find common support
            
        return np.average(diffs, weights=weights)

    except Exception as e:
        # print(f"Clustering shift error {ui}-{uj}: {e}")
        return 1.0

def compute_pairwise_concept_shift_clustering(data, groups, labels, feats, n_clusters=20, min_samples=5, n_jobs=1, cache_path=None, random_state=42):
    user_ids = np.unique(groups)
    n = len(user_ids)
    
    combined_data = data.copy()
    if isinstance(combined_data, pd.DataFrame):
        combined_data = combined_data[feats]
    combined_data['label'] = labels
    combined_data.index = pd.MultiIndex.from_arrays([groups, combined_data.index], names=['user', 'sample'])
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_concept_shift_clustering_pair)(i, j, user_ids, combined_data, feats, n_clusters, min_samples, random_state)
        for i in range(n)
        for j in range(i+1, n)
    )
    
    mat = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = results[k]
            k += 1
            
    if cache_path:
        np.save(cache_path, mat)
    return user_ids, mat

def _compute_concept_shift_knn_pair(i, j, user_ids, data, feats, k_neighbors=10, random_state=42):
    """
    Compute concept shift using k-Nearest Neighbors.
    
    For each point x in User i's Test set:
        Find k neighbors in User j's Train set.
        Compare y_true_i(x) (which is just y_i) with Average(y_true_j(neighbors)).
    
    Symmetrized: (Shift(i->j) + Shift(j->i)) / 2
    """
    ui, uj = user_ids[i], user_ids[j]
    
    try:
        Xi = data.loc[data.index.get_level_values(0) == ui, feats].values
        yi = data.loc[data.index.get_level_values(0) == ui, 'label'].values
        Xj = data.loc[data.index.get_level_values(0) == uj, feats].values
        yj = data.loc[data.index.get_level_values(0) == uj, 'label'].values
        
        # Subsample if too large for speed
        if len(Xi) > 1000:
            Xi, yi = resample(Xi, yi, n_samples=1000, random_state=random_state)
        if len(Xj) > 1000:
            Xj, yj = resample(Xj, yj, n_samples=1000, random_state=random_state)

        def one_way_shift(X_source, y_source, X_ref, y_ref):
            if len(X_ref) < k_neighbors:
                return 1.0
            
            # Train KNN on Ref
            knn = KNeighborsClassifier(n_neighbors=min(k_neighbors, len(X_ref)))
            knn.fit(X_ref, y_ref)
            
            # Predict P(y=1|x) based on Ref's data
            if hasattr(knn, "predict_proba"):
                probs_ref = knn.predict_proba(X_source)[:, 1]
            else:
                 preds = knn.predict(X_source)
                 probs_ref = preds
            
            # Train KNN on Source too (LOO prediction)
            knn_source = KNeighborsClassifier(n_neighbors=min(k_neighbors, len(X_source)))
            knn_source.fit(X_source, y_source)
            probs_source = knn_source.predict_proba(X_source)[:, 1] # This is P_i(x)
            
            # Calculate difference
            return np.mean(np.abs(probs_source - probs_ref))

        s1 = one_way_shift(Xi, yi, Xj, yj)
        s2 = one_way_shift(Xj, yj, Xi, yi)
        
        return (s1 + s2) / 2

    except Exception as e:
        # print(f"KNN shift error {ui}-{uj}: {e}")
        return 1.0

def compute_pairwise_concept_shift_knn(data, groups, labels, feats, k_neighbors=20, n_jobs=1, cache_path=None, random_state=42):
    user_ids = np.unique(groups)
    n = len(user_ids)
    
    combined_data = data.copy()
    if isinstance(combined_data, pd.DataFrame):
        combined_data = combined_data[feats]
    combined_data['label'] = labels
    combined_data.index = pd.MultiIndex.from_arrays([groups, combined_data.index], names=['user', 'sample'])
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_concept_shift_knn_pair)(i, j, user_ids, combined_data, feats, k_neighbors, random_state)
        for i in range(n)
        for j in range(i+1, n)
    )
    
    mat = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = results[k]
            k += 1
            
    if cache_path:
        np.save(cache_path, mat)
    return user_ids, mat


# ========================= DetectShift-based Concept Shift ==========================

def _compute_detectshift_pair(
    i: int,
    j: int,
    user_ids: np.ndarray,
    data: pd.DataFrame,
    labels: np.ndarray,
    feats: List[str],
    test_size: float,
    B: int,
    random_state: int
) -> float:
    """Concept shift using DetectShift (concept shift type 2: conditional randomization).

    Steps mirror the DetectShift demo:
      1) prep_data to split source/target into train/test
      2) train KL models for total shift (Z) and covariate shift (X)
      3) train conditional density model
      4) ShiftDiagnostics → use conc2 (concept shift type 2) KL as distance
    """
    import sys
    import os
    # Ensure detectshift path is available for parallel workers
    detectshift_path = os.path.join(os.path.dirname(__file__), 'detectshift')
    if detectshift_path not in sys.path:
        sys.path.insert(0, detectshift_path)
    import detectshift as ds

    ui, uj = user_ids[i], user_ids[j]

    Xs = data.loc[data.index.get_level_values(0) == ui, feats]
    ys = pd.DataFrame(labels[data.index.get_level_values(0) == ui])
    Xt = data.loc[data.index.get_level_values(0) == uj, feats]
    yt = pd.DataFrame(labels[data.index.get_level_values(0) == uj])

    # Drop columns that are constant across both users to avoid CatBoost errors.
    combined = pd.concat([Xs, Xt], axis=0)
    varying_cols = [c for c in combined.columns if combined[c].nunique(dropna=False) > 1]
    if len(varying_cols) == 0:
        # All features are flat for both users (e.g., a category with a single always-zero column).
        # No signal to estimate shift; let caller handle via fallback distance.
        raise ValueError(f"DetectShift: all features constant for users {ui}-{uj}")
    Xs = Xs[varying_cols]
    Xt = Xt[varying_cols]

    # Require variability - need at least 20 samples per user for train/test split
    min_samples = max(20, int(10 / test_size))  # Ensure enough samples after split
    if len(Xs) < min_samples or len(Xt) < min_samples:
        raise ValueError(f"Insufficient data for users {ui}-{uj}: {len(Xs)}/{len(Xt)} samples")
    if ys.nunique().item() < 2 or yt.nunique().item() < 2:
        raise ValueError(f"Insufficient label variability for users {ui}-{uj}")

    try:
        Xs_train, Xs_test, ys_train, ys_test, Zs_train, Zs_test, \
        Xt_train, Xt_test, yt_train, yt_test, Zt_train, Zt_test = ds.tools.prep_data(
            Xs, ys, Xt, yt, test=test_size, task='class', random_state=random_state
        )
        
        # Validate that test sets have data and both classes
        if len(ys_test) == 0 or len(yt_test) == 0:
            raise ValueError(f"Empty test sets after split for users {ui}-{uj}")
        if ys_test.nunique().item() < 2 or yt_test.nunique().item() < 2:
            raise ValueError(f"Test sets lack both classes for users {ui}-{uj}")
    except (IndexError, ValueError, AttributeError) as e:
        raise ValueError(f"DetectShift prep_data failed for users {ui}-{uj}: {e}")

    # Helper to train KL with CatBoost and fall back to logistic regression when CatBoost rejects constant/ignored features.
    def _fit_kl(Zs, Zt):
        try:
            mdl = ds.tools.KL(boost=True)
            mdl.fit(Zs, Zt)
            return mdl
        except Exception as e:
            if "constant" in str(e).lower() or "ignored" in str(e).lower():
                mdl = ds.tools.KL(boost=False)
                mdl.fit(Zs, Zt)
                return mdl
            raise

    # KL estimators
    totshift_model = _fit_kl(Zs_train, Zt_train)
    covshift_model = _fit_kl(Xs_train, Xt_train)

    # Conditional model (fallback to logistic regression if CatBoost fails)
    try:
        cd_model = ds.cdist.cde_class(boost=True)
        cd_model.fit(pd.concat([Xs_train, Xt_train], axis=0),
                     pd.concat([ys_train, yt_train], axis=0))
    except Exception as e:
        if "constant" in str(e).lower() or "ignored" in str(e).lower():
            cd_model = ds.cdist.cde_class(boost=False)
            cd_model.fit(pd.concat([Xs_train, Xt_train], axis=0),
                         pd.concat([ys_train, yt_train], axis=0))
        else:
            raise

    try:
        out = ds.tests.ShiftDiagnostics(
            Xs_test, ys_test, Xt_test, yt_test,
            totshift_model=totshift_model,
            covshift_model=covshift_model,
            labshift_model=None,
            cd_model=cd_model,
            task='class',
            B=B,
            verbose=False
        )

        conc2 = out.get('conc2', {})
        val = conc2.get('kl', np.nan)
        
        # Handle scalar value issue: ensure val is a scalar, not tensor/array
        if hasattr(val, 'item'):  # PyTorch tensor or numpy array with .item() method
            val = val.item()
        elif isinstance(val, (np.ndarray, list, tuple)):
            val = float(val[0] if len(val) > 0 else np.nan)
        elif not np.isscalar(val) and not isinstance(val, (int, float, np.number)):
            # Try to convert to float
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan
        
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Invalid DetectShift KL for users {ui}-{uj}")
        return float(abs(val))  # Use absolute value for consistency
    except (ValueError, AttributeError, KeyError) as e:
        # Handle detectshift library bugs (e.g., column length mismatch in CondRand)
        error_msg = str(e)
        if "Length mismatch" in error_msg or "Expected axis" in error_msg:
            # This is a known bug in detectshift when feature/label dimensions don't match
            # Return high distance to indicate shift, but log it
            if not hasattr(_compute_detectshift_pair, '_warned_pairs'):
                _compute_detectshift_pair._warned_pairs = set()
            pair_key = f"{ui}-{uj}"
            if pair_key not in _compute_detectshift_pair._warned_pairs:
                import warnings
                warnings.warn(f"DetectShift column mismatch for {pair_key}, using fallback distance", UserWarning)
                _compute_detectshift_pair._warned_pairs.add(pair_key)
            # Use a fallback: compute simple KL divergence on predictions if possible
            try:
                # Fallback: use conditional model predictions to estimate shift
                pred_s = cd_model.predict(Xs_test)
                pred_t = cd_model.predict(Xt_test)
                # Simple distance metric: mean absolute difference
                fallback_dist = float(np.mean(np.abs(pred_s - pred_t)))
                return min(fallback_dist, 1.0)  # Cap at 1.0
            except:
                return 1.0  # Maximum distance if fallback also fails
        else:
            # Re-raise other errors
            raise ValueError(f"DetectShift failed for users {ui}-{uj}: {error_msg}")


def compute_pairwise_concept_shift_detectshift(
    data: pd.DataFrame,
    groups: np.ndarray,
    labels: np.ndarray,
    feats: List[str],
    test_size: float = 0.2,
    B: int = 200,
    n_jobs: int = 1,
    cache_path: str = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise concept shift matrix using DetectShift (concept shift type 2).

    Args:
        data: Feature dataframe (all users).
        groups: User ids per row.
        labels: Array of labels aligned to data.
        feats: Feature list.
        test_size: Fraction for DetectShift internal split.
        B: Number of permutations for DetectShift.
        n_jobs: Parallel jobs.
        cache_path: Optional cache path.
        random_state: Seed.
    """
    user_ids = np.unique(groups)
    if cache_path and os.path.exists(cache_path):
        try:
            mat = np.load(cache_path)
            if mat.shape == (len(user_ids), len(user_ids)):
                return user_ids, mat
        except Exception:
            pass

    combined = data[feats].copy()
    combined['label'] = labels
    combined.index = pd.MultiIndex.from_arrays([groups, combined.index], names=['user', 'sample'])

    def _safe_compute_pair(args):
        """Wrapper to catch exceptions in parallel execution."""
        try:
            return _compute_detectshift_pair(*args)
        except (ValueError, AttributeError, KeyError, IndexError, Exception) as e:
            # Handle detectshift library bugs and data issues
            error_msg = str(e)
            if "Length mismatch" in error_msg or "Expected axis" in error_msg:
                # Known detectshift bug - return high distance
                return 1.0
            elif "Insufficient" in error_msg or "Invalid" in error_msg:
                # Data quality issues - return high distance
                return 1.0
            else:
                # Unknown error - return high distance
                return 1.0

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_safe_compute_pair)(
            (i, j, user_ids, combined, labels, feats, test_size, B, random_state)
        )
        for i in range(len(user_ids))
        for j in range(i + 1, len(user_ids))
    )

    mat = np.zeros((len(user_ids), len(user_ids)))
    k = 0
    valid_count = 0
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            val = results[k]
            # Ensure valid float value
            if isinstance(val, (int, float)) and not (np.isnan(val) or np.isinf(val)):
                mat[i, j] = mat[j, i] = float(abs(val))  # Use absolute value
                valid_count += 1
            else:
                mat[i, j] = mat[j, i] = 1.0  # Fallback to high distance
            k += 1
    
    if valid_count < len(results) * 0.5:  # Warn if less than 50% succeeded
        import warnings
        warnings.warn(f"DetectShift: Only {valid_count}/{len(results)} pairs computed successfully", UserWarning)

    if cache_path:
        np.save(cache_path, mat)

    return user_ids, mat

def select_top_xgb_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_k: int = 10,
    random_state=42
) -> List[str]:
    """Select top features using SHAP values from an XGBoost classifier.

    Args:
        X_train (pd.DataFrame): Training features dataframe.
        y_train (pd.Series): Training target labels series.
        X_test (pd.DataFrame): Test features dataframe.
        y_test (pd.Series): Test target labels series.
        top_k (int): Number of top features to select.
        random_state: Random seed for reproducibility.

    Returns:
        List[str]: List of selected feature names.

    Raises:
        ValueError: If y_train or y_test contains fewer than two classes.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train must contain at least two classes")
    if len(np.unique(y_test)) < 2:
        raise ValueError("y_test must contain at least two classes")

    model = XGBClassifier(
        n_estimators=500, max_depth=4, random_state=random_state,
        eval_metric='auc', early_stopping_rounds=100, n_jobs= -1,
        tree_method='hist', gpu_id=-1  # Force CPU usage
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list): 
        shap_values = np.array(shap_values).mean(axis=0)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:]

    return X_train.columns[top_indices].tolist()
