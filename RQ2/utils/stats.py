from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def fdr_bh_series(p_values):
    """Benjamini–Hochberg q-values (Series-like input)."""
    p = pd.Series(p_values, dtype=float)
    mask = p.notna()
    m = int(mask.sum())
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if m == 0:
        return q
    order = p[mask].sort_values().index
    ranks = np.arange(1, m + 1)
    adj = p[mask].loc[order].values * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q.loc[order] = np.minimum(adj, 1.0)
    return q


def wilcoxon_greater(x):
    """Wilcoxon signed-rank test for greater than zero."""
    x = np.asarray(list(x), dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 10 or np.allclose(x, 0):
        return np.nan
    try:
        return wilcoxon(x, alternative="greater").pvalue
    except ValueError:
        return np.nan


def sign_test_greater(x):
    """Sign test for greater than zero."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    k_pos = int((x > 0).sum())
    return binomtest(k_pos, n, p=0.5, alternative="greater").pvalue
