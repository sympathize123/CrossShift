from __future__ import annotations

try:
    from ..models import (
        XGBModel,
        LGBMModel,
        CatBoostModel,
        MLPModel,
        TabTransformerModel,
        FTTransformerModel,
        NODEModel,
        TabResNetModel,
    )
except ImportError:
    from models import (
        XGBModel,
        LGBMModel,
        CatBoostModel,
        MLPModel,
        TabTransformerModel,
        FTTransformerModel,
        NODEModel,
        TabResNetModel,
    )


def get_model_by_name(model_name: str, num_features: int, random_state: int = 42, **kwargs):
    """Return a model instance by name with optional hyperparameters."""
    model_map = {
        "xgb": XGBModel,
        "lgbm": LGBMModel,
        "catboost": CatBoostModel,
        "mlp": MLPModel,
        "tabtransformer": TabTransformerModel,
        "fttransformer": FTTransformerModel,
        "node": NODEModel,
        "tabresnet": TabResNetModel,
    }

    key = model_name.lower()
    if key not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")

    model_cls = model_map[key]

    if key in {"xgb", "lgbm", "catboost"}:
        return model_cls(random_state=random_state, n_jobs=kwargs.get("n_jobs", -1), **kwargs)
    return model_cls(num_features=num_features, random_state=random_state, **kwargs)


def is_deep_model(model_name: str) -> bool:
    """Check if a model is a deep learning model (uses GPU)."""
    return model_name.lower() in {"mlp", "tabtransformer", "fttransformer", "node", "tabresnet"}


def is_tree_model(model_name: str) -> bool:
    """Check if a model is a tree-based model (uses parallel computing)."""
    return model_name.lower() in {"xgb", "lgbm", "catboost"}
