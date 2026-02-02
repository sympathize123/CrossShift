"""
Machine Learning Models Package

This package contains implementations of various machine learning models
organized into tree-based and deep learning models.
"""

from .tree_models import XGBModel, LGBMModel, CatBoostModel

try:
    from .deep_models import (
        MLPModel,
        TabTransformerModel,
        FTTransformerModel,
        NODEModel,
        TabResNetModel
    )
    DEEP_MODELS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DEEP_MODELS_AVAILABLE = False

    class _MissingDeepModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Deep learning models require additional dependencies (e.g., torch). "
                "Install the necessary packages to enable these models."
            )

    MLPModel = TabTransformerModel = FTTransformerModel = NODEModel = TabResNetModel = _MissingDeepModel  # type: ignore

__all__ = [
    # Tree-based models (parallel computing)
    'XGBModel',
    'LGBMModel', 
    'CatBoostModel',
    
    # Deep learning models (GPU)
    'MLPModel',
    'TabTransformerModel',
    'FTTransformerModel', 
    'NODEModel',
    'TabResNetModel'
]
