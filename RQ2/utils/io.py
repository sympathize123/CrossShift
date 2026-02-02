from __future__ import annotations

from datetime import datetime
import warnings

import cloudpickle
import numpy as np
import pandas as pd
import scipy.stats as st


def load(path: str):
    """Load an object from a pickle file."""
    with open(path, mode="rb") as f:
        return cloudpickle.load(f)


def dump(obj, path: str):
    """Dump an object to a pickle file."""
    with open(path, mode="wb") as f:
        cloudpickle.dump(obj, f)


def log(msg):
    """Log a message with timestamp."""
    print(f"[{datetime.now().strftime('%y-%m-%d %H:%M:%S')}] {msg}")


def summary(x):
    """Generate summary statistics for numeric or categorical data."""
    x = np.asarray(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        n = len(x)
        if x.dtype.kind.isupper() or x.dtype.kind == "b":
            cnt = pd.Series(x).value_counts(dropna=False)
            card = len(cnt)
            cnt = cnt[:20]
            cnt_str = ", ".join([f"{u}:{c}" for u, c in zip(cnt.index, cnt)])
            if card > 30:
                cnt_str = f"{cnt_str}, ..."
            return {"n": n, "cardinality": card, "value_count": cnt_str}

        x_nan = x[np.isnan(x)]
        x_norm = x[~np.isnan(x)]

        if len(x_norm) == 0:
            return {"n": n, "nan_count": n}

        tot = np.sum(x_norm)
        m = np.mean(x_norm)
        me = np.median(x_norm)
        s = np.std(x_norm, ddof=1)
        l, u = np.min(x_norm), np.max(x_norm)
        conf_l, conf_u = st.t.interval(0.95, len(x_norm) - 1, loc=m, scale=st.sem(x_norm))
        n_nan = len(x_nan)

        return {
            "n": n,
            "sum": tot,
            "mean": m,
            "SD": s,
            "med": me,
            "range": (l, u),
            "conf.": (conf_l, conf_u),
            "nan_count": n_nan,
        }
