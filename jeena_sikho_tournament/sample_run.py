import time
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

from .features import feature_sets, make_supervised
from .models_zoo import get_candidates
from .metrics import accuracy, f1_score, mae, pinball_loss, coverage
from .backtest import trading_score
from .splits import walk_forward_split
from .storage import Storage


def _subset_candidates(task: str, max_count: int, enable_dl: bool):
    specs = get_candidates(task, max_count, enable_dl, candle_minutes=config.candle_minutes, strict_horizon_pool=True)
    return specs[:max_count]


def run_dry_tournament(config) -> bool:
    storage = Storage(config.db_path, config.ohlcv_table)
    df = storage.load()
    if df.empty:
        return False

    sup = make_supervised(df, candle_minutes=config.candle_minutes, feature_windows_hours=config.feature_windows)
    if sup.empty:
        return False

    # Use last 30 days train, last 48 hours validation
    config_train_days = 30
    config_val_hours = 48
    split = walk_forward_split(sup, config_train_days, config_val_hours, 0, False)

    fs_map = feature_sets(sup)
    fs_id = "minimal" if "minimal" in fs_map else list(fs_map.keys())[0]
    cols = fs_map[fs_id]

    start = time.time()
    for task in ["direction", "return", "range"]:
        specs = _subset_candidates(task, 5, False)
        X_train = split.train[cols]
        X_val = split.val[cols]
        if task == "direction":
            y_train = split.train["y_dir"].values
            y_val = split.val["y_dir"].values
            for spec in specs:
                model = spec.model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                _ = accuracy(y_val, y_pred)
                _ = f1_score(y_val, y_pred)
        elif task == "return":
            y_train = split.train["y_ret"].values
            y_val = split.val["y_ret"].values
            for spec in specs:
                model = spec.model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                _ = mae(y_val, y_pred)
        else:
            y_train = split.train["y_ret"].values
            y_val = split.val["y_ret"].values
            for spec in specs[:2]:
                model = spec.model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                if y_pred.ndim == 1:
                    p10 = p50 = p90 = y_pred
                else:
                    p10, p50, p90 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
                _ = coverage(y_val, p10, p90)
                _ = pinball_loss(y_val, p50, 0.5)

    elapsed = time.time() - start
    return elapsed < 120
