from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from ..splitters import loso_split_indices, stratified_group_kfold_indices
except ImportError:
    from splitters import loso_split_indices, stratified_group_kfold_indices


def _assemble_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Helper to assemble datasets from provided indices."""
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]
    groups_test = groups[test_idx]

    return X_train, y_train, groups_train, X_test, y_test, groups_test


def create_train_test_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    strategy: str = "random",
    fold_index: int = 0,
    fold_group: Any = None,
    n_splits: int = 5,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create train/test splits with configurable strategies (random, LOSO, stratified group)."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    X = X.reset_index(drop=True)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if groups_arr.ndim != 1:
        raise ValueError("groups must be a 1D array.")
    if len(X) != len(y_arr) or len(y_arr) != len(groups_arr):
        raise ValueError("Lengths of X, y, and groups must match.")

    strategy_lower = strategy.lower()

    if strategy_lower == "random":
        unique_groups = np.unique(groups_arr)
        train_indices: List[np.ndarray] = []
        test_indices: List[np.ndarray] = []

        for group in unique_groups:
            group_mask = np.where(groups_arr == group)[0]
            y_group = y_arr[group_mask]

            if np.unique(y_group).size < 2:
                print(f"Skipping user {group}: insufficient class diversity")
                continue

            try:
                local_indices = np.arange(group_mask.size)
                train_local, test_local = train_test_split(
                    local_indices,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_group,
                    shuffle=True,
                )
            except ValueError as exc:
                print(f"Skipping user {group}: unable to stratify ({exc})")
                continue

            if (
                np.unique(y_group[train_local]).size < 2
                or np.unique(y_group[test_local]).size < 2
            ):
                print(f"Skipping user {group}: split results in insufficient class diversity")
                continue

            train_indices.append(group_mask[train_local])
            test_indices.append(group_mask[test_local])

        if not train_indices:
            raise ValueError("No valid users found for train/test split.")

        train_idx = np.concatenate(train_indices)
        test_idx = np.concatenate(test_indices)

    elif strategy_lower == "temporal":
        if time_values is None:
            raise ValueError("Temporal split requires time_values.")
        time_series = pd.to_datetime(time_values, errors="coerce")
        if time_series.isna().all():
            time_key = np.asarray(time_values)
        else:
            time_key = time_series.to_numpy()

        if time_key.ndim != 1:
            raise ValueError("time_values must be a 1D array.")
        if len(time_key) != len(y_arr):
            raise ValueError("Lengths of time_values and labels must match.")

        unique_groups = np.unique(groups_arr)
        train_indices = []
        test_indices = []

        for group in unique_groups:
            group_mask = np.where(groups_arr == group)[0]
            y_group = y_arr[group_mask]

            if np.unique(y_group).size < 2:
                print(f"Skipping user {group}: insufficient class diversity")
                continue

            order = np.argsort(time_key[group_mask])
            split_point = int(np.floor(group_mask.size * (1 - test_size)))

            if split_point <= 0 or split_point >= group_mask.size:
                print(f"Skipping user {group}: invalid temporal split point")
                continue

            train_local = order[:split_point]
            test_local = order[split_point:]

            if (
                np.unique(y_group[train_local]).size < 2
                or np.unique(y_group[test_local]).size < 2
            ):
                print(f"Skipping user {group}: split results in insufficient class diversity")
                continue

            train_indices.append(group_mask[train_local])
            test_indices.append(group_mask[test_local])

        if not train_indices:
            raise ValueError("No valid users found for temporal train/test split.")

        train_idx = np.concatenate(train_indices)
        test_idx = np.concatenate(test_indices)

    elif strategy_lower == "loso":
        splits = list(loso_split_indices(groups_arr))
        if not splits:
            raise ValueError("Cannot perform LOSO split: no groups available.")

        if fold_group is not None:
            group_to_fold = {meta.info.get("group"): meta for meta in splits}
            if fold_group not in group_to_fold:
                raise ValueError(f"Group {fold_group} not found for LOSO split.")
            selected = group_to_fold[fold_group]
        else:
            if fold_index < 0 or fold_index >= len(splits):
                raise ValueError(
                    f"fold_index {fold_index} out of range for LOSO splits (0-{len(splits) - 1})."
                )
            selected = splits[fold_index]

        train_idx, test_idx = selected.train_idx, selected.test_idx

    elif strategy_lower == "stratified_group_kfold":
        splits = list(
            stratified_group_kfold_indices(
                y_arr,
                groups_arr,
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        )
        if not splits:
            raise ValueError("Stratified group K-Fold produced no splits.")
        if fold_index < 0 or fold_index >= len(splits):
            raise ValueError(
                f"fold_index {fold_index} out of range for StratifiedGroupKFold (0-{len(splits) - 1})."
            )
        selected = splits[fold_index]
        train_idx, test_idx = selected.train_idx, selected.test_idx

    else:
        raise ValueError(f"Unknown splitting strategy '{strategy}'.")

    (
        X_train_all,
        y_train_all,
        groups_train_all,
        X_test_all,
        y_test_all,
        groups_test_all,
    ) = _assemble_split(X, y_arr, groups_arr, train_idx, test_idx)

    print(
        f"Training data: {len(X_train_all)} samples from {len(np.unique(groups_train_all))} users"
    )
    print(
        f"Test data: {len(X_test_all)} samples from {len(np.unique(groups_test_all))} users"
    )

    return (
        X_train_all,
        y_train_all,
        groups_train_all,
        X_test_all,
        y_test_all,
        groups_test_all,
    )
