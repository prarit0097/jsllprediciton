from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: Optional[pd.DataFrame]


def walk_forward_split(df: pd.DataFrame, train_days: int, val_hours: int, test_hours: int, use_test: bool) -> Split:
    if df.empty:
        raise ValueError("Empty dataframe")
    start = df.index.min()
    end = df.index.max()
    total_hours = (end - start).total_seconds() / 3600.0

    desired_val = float(val_hours)
    desired_test = float(test_hours) if use_test else 0.0
    desired_train = float(train_days) * 24.0

    if total_hours < desired_train + desired_val + desired_test:
        desired_test = min(desired_test, max(0.0, total_hours * 0.1))
        desired_val = min(desired_val, max(1.0, total_hours * 0.2))
        desired_train = max(1.0, total_hours - desired_val - desired_test)

    test = None
    val_end = end
    if use_test and desired_test > 0:
        test_start = end - timedelta(hours=desired_test)
        test = df.loc[test_start:end].copy()
        val_end = test_start
    val_start = val_end - timedelta(hours=desired_val)
    train_end = val_start
    train_start = train_end - timedelta(hours=desired_train)

    train = df.loc[train_start:train_end].copy()
    val = df.loc[val_start:val_end].copy()
    if not train.empty:
        train = train.loc[train.index < val_start].copy()

    return Split(train=train, val=val, test=test)


def walk_forward_cv_splits(
    df: pd.DataFrame,
    train_days: int,
    val_hours: int,
    test_hours: int,
    use_test: bool,
    folds: int = 3,
) -> List[Split]:
    if df.empty:
        raise ValueError("Empty dataframe")
    folds = max(1, int(folds))
    base = walk_forward_split(df, train_days, val_hours, test_hours, use_test)
    if folds == 1 or base.val.empty:
        return [base]

    out: List[Split] = []
    step = timedelta(hours=max(1, int(val_hours)))
    end_anchor = base.val.index.max()
    for i in range(folds):
        fold_end = end_anchor - (step * i)
        if fold_end <= df.index.min():
            break
        fold_df = df.loc[:fold_end].copy()
        if fold_df.empty:
            continue
        split_i = walk_forward_split(fold_df, train_days, val_hours, test_hours, use_test)
        if split_i.train.empty or split_i.val.empty:
            continue
        out.append(split_i)
    if not out:
        return [base]
    return list(reversed(out))
