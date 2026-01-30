"""Reusable dataset splitting utilities for group-aware evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple, Union

import numpy as np

SplitIndex = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SplitMetadata:
    """Container for a single train/test index split."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    info: dict


def _as_numpy(array: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Ensure the input is a 1D numpy array."""
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def loso_split_indices(groups: Sequence) -> Iterator[SplitMetadata]:
    """Yield Leave-One-Subject-Out (LOSO) split indices.

    Args:
        groups: Sequence of group identifiers aligned with the dataset rows.

    Yields:
        SplitMetadata entries containing train/test indices and metadata with the held-out group.
    """
    groups_arr = _as_numpy(groups)
    unique_groups = np.unique(groups_arr)

    for fold, group in enumerate(unique_groups):
        test_idx = np.where(groups_arr == group)[0]
        train_idx = np.where(groups_arr != group)[0]

        if test_idx.size == 0:
            continue

        yield SplitMetadata(
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            info={"group": group},
        )


def stratified_group_kfold_indices(
    y: Sequence,
    groups: Sequence,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> Iterator[SplitMetadata]:
    """Yield Stratified Group K-Fold indices using scikit-learn's implementation.

    Args:
        y: Target labels aligned with the dataset rows.
        groups: Group identifiers aligned with the dataset rows.
        n_splits: Number of folds.
        shuffle: Whether to shuffle the groups before splitting.
        random_state: Random seed used when shuffling.

    Yields:
        SplitMetadata entries with fold index.

    Raises:
        ImportError: If StratifiedGroupKFold is not available in the current sklearn version.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "StratifiedGroupKFold requires scikit-learn >= 1.1. "
            "Please upgrade scikit-learn to use stratified group splits."
        ) from exc

    y_arr = _as_numpy(y)
    groups_arr = _as_numpy(groups)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    dummy_X = np.zeros((len(y_arr), 1), dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy_X, y_arr, groups=groups_arr)):
        yield SplitMetadata(
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            info={},
        )


__all__ = [
    "SplitMetadata",
    "loso_split_indices",
    "stratified_group_kfold_indices",
]
