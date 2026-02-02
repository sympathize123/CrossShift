from __future__ import annotations

import logging
import os
import warnings


def configure_runtime():
    """Configure environment variables and warning filters for reproducible runs."""
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*is_sparse.*")
    warnings.filterwarnings("ignore", message=".*is_categorical_dtype.*")
    warnings.filterwarnings("ignore", message=".*SparseDtype.*")
    warnings.filterwarnings("ignore", message=".*CategoricalDtype.*")
    warnings.filterwarnings("ignore", message=".*swigvarlink.*")
    warnings.filterwarnings("ignore", message=".*ot.gpu not found.*")
    warnings.filterwarnings("ignore", message=".*coupling computation will be in cpu.*")
    warnings.filterwarnings("ignore", message=".*computation placer already registered.*")
    warnings.filterwarnings("ignore", message=".*Unable to register.*")

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
