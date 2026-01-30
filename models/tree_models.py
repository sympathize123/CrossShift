"""
Tree-based Models for Tabular Data

These models use parallel computing (n_jobs=-1) for CPU optimization.
All models are configured with similar parameter counts for fair comparison.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  
from catboost import CatBoostClassifier


class XGBModel:
    """XGBoost Classifier optimized for parallel computing"""
    
    def __init__(
        self,
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=100,
        **kwargs
    ):
        # User-configurable parameters with reasonable defaults
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=kwargs.get('min_child_weight', 1),
            gamma=kwargs.get('gamma', 0),
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
            enable_categorical=False
        )
        self.feature_importances_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Convert to numpy arrays to ensure compatibility across all models
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        if X_val is not None:
            X_val_arr = np.asarray(X_val)
            y_val_arr = np.asarray(y_val)
            self.model.fit(X_arr, y_arr, eval_set=[(X_val_arr, y_val_arr)], verbose=False)
        else:
            self.model.fit(X_arr, y_arr, verbose=False)
        
        self.feature_importances_ = self.model.feature_importances_
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))
    
    def predict(self, X):
        return self.model.predict(X)


class LGBMModel:
    """LightGBM Classifier optimized for parallel computing"""
    
    def __init__(self, random_state=42, n_jobs=-1, n_estimators=200, max_depth=5,
                 learning_rate=0.1, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=50, **kwargs):
        # User-configurable parameters with reasonable defaults
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=kwargs.get('min_child_samples', 5),
            min_child_weight=kwargs.get('min_child_weight', 1e-3),
            min_split_gain=kwargs.get('min_split_gain', 0.0),
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbosity=-1,
            force_col_wise=True,
            importance_type='gain',
            n_jobs=n_jobs
        )
        self.feature_importances_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Convert to numpy arrays to ensure compatibility across all models
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        if X_val is not None:
            X_val_arr = np.asarray(X_val)
            y_val_arr = np.asarray(y_val)
            self.model.fit(X_arr, y_arr, eval_set=[(X_val_arr, y_val_arr)])
        else:
            self.model.fit(X_arr, y_arr)
        self.feature_importances_ = self.model.feature_importances_
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))
    
    def predict(self, X):
        return self.model.predict(X)


class CatBoostModel:
    """CatBoost Classifier optimized for parallel computing"""
    
    def __init__(self, random_state=42, n_jobs=-1, iterations=200, depth=6,
                 learning_rate=0.1, subsample=0.8, colsample_bylevel=0.8,
                 reg_lambda=1.0, early_stopping_rounds=50, **kwargs):
        # User-configurable parameters with reasonable defaults
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            reg_lambda=reg_lambda,
            min_data_in_leaf=kwargs.get('min_data_in_leaf', 20),
            early_stopping_rounds=early_stopping_rounds,
            objective='Logloss',
            eval_metric='AUC',
            task_type='CPU',
            thread_count=n_jobs,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        )
        self.feature_importances_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Convert to numpy arrays to ensure compatibility across all models
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        if X_val is not None:
            X_val_arr = np.asarray(X_val)
            y_val_arr = np.asarray(y_val)
            self.model.fit(X_arr, y_arr, eval_set=[(X_val_arr, y_val_arr)])
        else:
            self.model.fit(X_arr, y_arr)
            
        self.feature_importances_ = self.model.feature_importances_
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))
    
    def predict(self, X):
        return self.model.predict(X)
