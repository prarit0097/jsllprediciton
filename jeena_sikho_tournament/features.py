import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_rsi(series: pd.Series, bars: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / max(1, bars), adjust=False, min_periods=bars).mean()
    avg_loss = loss.ewm(alpha=1 / max(1, bars), adjust=False, min_periods=bars).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss > 0, np.where(avg_gain > 0, 100.0, 50.0))
    return pd.Series(rsi, index=series.index, dtype=float)


def _bars_for_hours(hours: int, candle_minutes: int) -> int:
    if candle_minutes <= 0:
        return max(1, hours)
    bars = int(round((hours * 60) / candle_minutes))
    return max(1, bars)


def _target_label_for_minutes(candle_minutes: int) -> str:
    minutes = max(1, int(candle_minutes))
    if minutes % 1440 == 0:
        days = max(1, minutes // 1440)
        return f"y_ret_{days}d"
    if minutes % 60 == 0:
        hours = max(1, minutes // 60)
        return f"y_ret_{hours}h"
    return f"y_ret_{minutes}m"


def resolve_feature_windows_for_horizon(
    candle_minutes: int,
    feature_windows_hours: Optional[Iterable[int]],
) -> list[int]:
    base = list(feature_windows_hours) if feature_windows_hours is not None else [2, 4, 8, 12, 24, 48, 72, 96, 168]
    minutes = max(1, int(candle_minutes))
    if minutes <= 120:
        preferred = [2, 4, 8, 12, 24, 48, 72, 96]
    elif minutes >= 1440:
        preferred = [24, 48, 72, 96, 120, 168, 240, 336, 504]
    else:
        preferred = [4, 8, 12, 24, 48, 72, 96, 168, 240]
    seen = set()
    merged: list[int] = []
    for w in preferred + base:
        if w <= 0 or w in seen:
            continue
        seen.add(w)
        merged.append(int(w))
    return merged


def allowed_feature_sets_for_horizon(candle_minutes: int) -> set[str]:
    minutes = max(1, int(candle_minutes))
    if minutes <= 120:
        return {"minimal", "base", "momentum", "micro_momentum", "session", "vwap_flow", "signal", "trend", "volatility", "context", "price_action"}
    if minutes >= 1440:
        return {"minimal", "base", "long", "trend_longer", "volatility", "session", "signal", "trend", "context", "price_action"}
    return {"minimal", "base", "momentum", "signal", "trend", "volatility", "session", "vwap_flow", "trend_longer", "context", "price_action"}


def compute_features(
    df: pd.DataFrame,
    candle_minutes: int = 60,
    feature_windows_hours: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    log_close = np.log(close)
    df["ret_1c"] = log_close.diff(1)
    bars_1h = _bars_for_hours(1, candle_minutes)
    df["ret_1h"] = log_close.diff(bars_1h)
    df["ret_4h"] = log_close.diff(_bars_for_hours(4, candle_minutes))
    df["ret_24h"] = log_close.diff(_bars_for_hours(24, candle_minutes))

    windows = resolve_feature_windows_for_horizon(candle_minutes, feature_windows_hours)
    for w in windows:
        bars = _bars_for_hours(w, candle_minutes)
        df[f"roll_mean_{w}"] = df["ret_1c"].rolling(bars, min_periods=bars).mean()
        # ddof=0 avoids all-NaN std when bars == 1 on higher timeframes (e.g. 2h+).
        df[f"roll_std_{w}"] = df["ret_1c"].rolling(bars, min_periods=bars).std(ddof=0)
        df[f"z_{w}"] = df["ret_1c"] / (df[f"roll_std_{w}"] + 1e-9)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_window = _bars_for_hours(14, candle_minutes)
    df["atr_14"] = tr.rolling(atr_window, min_periods=atr_window).mean()
    vol_window = _bars_for_hours(24, candle_minutes)
    df["vol_24"] = df["ret_1c"].rolling(vol_window, min_periods=vol_window).std(ddof=0)

    for w in [14, 21]:
        bars = _bars_for_hours(w, candle_minutes)
        df[f"rsi_{w}"] = _wilder_rsi(close, bars)

    ema12 = _ema(close, _bars_for_hours(12, candle_minutes))
    ema26 = _ema(close, _bars_for_hours(26, candle_minutes))
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], _bars_for_hours(9, candle_minutes))
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_window = _bars_for_hours(20, candle_minutes)
    ma20 = close.rolling(bb_window, min_periods=bb_window).mean()
    std20 = close.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma20 + 1e-9)

    typical = (high + low + close) / 3
    vwap_window = _bars_for_hours(24, candle_minutes)
    vwap_num = (typical * volume).rolling(vwap_window, min_periods=vwap_window).sum()
    vwap_den = volume.rolling(vwap_window, min_periods=vwap_window).sum()
    df["vwap_24"] = vwap_num / (vwap_den + 1e-9)
    df["vwap_dist"] = (close - df["vwap_24"]) / (df["vwap_24"] + 1e-9)

    df["vol_chg"] = volume.pct_change(1, fill_method=None)
    df["vol_mean_24"] = volume.rolling(vwap_window, min_periods=vwap_window).mean()
    df["vol_std_24"] = volume.rolling(vwap_window, min_periods=vwap_window).std(ddof=0)

    fast = _ema(close, _bars_for_hours(12, candle_minutes))
    slow = _ema(close, _bars_for_hours(48, candle_minutes))
    trend_strength = (fast - slow).abs() / (close + 1e-9)
    df["trend_flag"] = (trend_strength > 0.002).astype(int)
    df["range_flag"] = (trend_strength <= 0.002).astype(int)

    vol_med_window = _bars_for_hours(168, candle_minutes)
    vol_med = df["vol_24"].rolling(vol_med_window, min_periods=vol_med_window).median()
    df["high_vol_flag"] = (df["vol_24"] > vol_med).astype(int)
    df["low_vol_flag"] = (df["vol_24"] <= vol_med).astype(int)

    idx_local = df.index.tz_convert("Asia/Kolkata")
    minute_of_day = (idx_local.hour * 60) + idx_local.minute
    open_min = 9 * 60 + 15
    close_min = 15 * 60 + 30
    minutes_from_open = minute_of_day - open_min
    df["minutes_from_open"] = minutes_from_open.astype(float)
    df["is_opening_hour"] = ((minutes_from_open >= 0) & (minutes_from_open < 60)).astype(int)
    df["is_closing_hour"] = ((minute_of_day >= (close_min - 60)) & (minute_of_day <= close_min)).astype(int)
    df["day_of_week"] = idx_local.weekday.astype(int)

    date_series = pd.Series(idx_local.date, index=df.index)
    new_day = date_series != date_series.shift(1)
    prev_close = close.shift(1)
    gap = (df["open"] - prev_close) / (prev_close + 1e-9)
    df["gap_from_prev_close"] = np.where(new_day, gap, 0.0)

    # Candle-structure and session-context features.
    candle_range = (high - low).replace(0.0, np.nan)
    candle_top = pd.concat([df["open"], close], axis=1).max(axis=1)
    candle_bottom = pd.concat([df["open"], close], axis=1).min(axis=1)
    df["body_pct"] = (close - df["open"]) / (df["open"].abs() + 1e-9)
    df["range_pct"] = (high - low) / (close.abs() + 1e-9)
    df["upper_wick_pct"] = (high - candle_top) / (close.abs() + 1e-9)
    df["lower_wick_pct"] = (candle_bottom - low) / (close.abs() + 1e-9)
    df["close_location"] = (close - low) / (candle_range + 1e-9)
    df["ret_2c"] = log_close.diff(2)
    df["ret_3c"] = log_close.diff(3)

    session_open = df["open"].groupby(date_series).transform("first")
    df["day_open_dist"] = (close - session_open) / (session_open + 1e-9)

    daily_high = high.groupby(date_series).max()
    daily_low = low.groupby(date_series).min()
    prev_day_high = date_series.map(daily_high.shift(1))
    prev_day_low = date_series.map(daily_low.shift(1))
    df["prev_day_high_break"] = (high > prev_day_high).astype(int)
    df["prev_day_low_break"] = (low < prev_day_low).astype(int)

    vol_168_window = _bars_for_hours(168, candle_minutes)
    df["vol_168"] = df["ret_1c"].rolling(vol_168_window, min_periods=vol_168_window).std(ddof=0)
    df["vol_ratio_24_168"] = df["vol_24"] / (df["vol_168"] + 1e-9)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _target_quantile_defaults(candle_minutes: int) -> tuple[float, float]:
    if candle_minutes <= 120:
        return 0.02, 0.98
    if candle_minutes >= 1440:
        return 0.005, 0.995
    return 0.01, 0.99


def _event_target_defaults(candle_minutes: int) -> tuple[float, float]:
    if candle_minutes <= 120:
        return 0.06, 0.035
    if candle_minutes >= 1440:
        return 0.10, 0.07
    return 0.08, 0.05


def _target_scale_series(df: pd.DataFrame) -> pd.Series:
    vol_floor = float(os.getenv("TARGET_VOL_FLOOR", "0.001"))
    vol_cap = float(os.getenv("TARGET_VOL_CAP", "0.08"))
    target_scale = df.get("vol_24", pd.Series(1.0, index=df.index))
    return target_scale.clip(lower=max(1e-6, vol_floor), upper=max(vol_floor, vol_cap))


def _annotate_model_context(df: pd.DataFrame, candle_minutes: int) -> pd.DataFrame:
    out = df.copy()
    default_event_gap, _ = _event_target_defaults(candle_minutes)
    event_gap = float(os.getenv("EVENT_GAP_THRESHOLD", str(default_event_gap)))
    event_mask = pd.Series(False, index=out.index)
    if "gap_from_prev_close" in out.columns:
        event_mask = out["gap_from_prev_close"].abs() >= event_gap
    out["is_event_day"] = event_mask.astype(int)
    out["target_scale"] = _target_scale_series(out)
    return out


def make_inference_frame(
    df: pd.DataFrame,
    candle_minutes: int = 60,
    feature_windows_hours: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    frame = compute_features(df, candle_minutes=candle_minutes, feature_windows_hours=feature_windows_hours)
    frame = _annotate_model_context(frame, candle_minutes)
    if "volume" in frame.columns:
        frame = frame.loc[frame["volume"] > 0]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def feature_sets(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    base = [
        "ret_1c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "rsi_14",
        "rsi_21",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "vwap_24",
        "vwap_dist",
    ]
    vol = [
        "atr_14",
        "vol_24",
        "bb_width",
        "trend_flag",
        "range_flag",
        "high_vol_flag",
        "low_vol_flag",
    ]
    momentum = [
        "ret_1c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "z_4",
        "z_12",
        "z_24",
        "macd",
        "macd_hist",
    ]
    volume = [
        "vol_chg",
        "vol_mean_24",
        "vol_std_24",
        "vwap_24",
        "vwap_dist",
    ]
    trend = [
        "macd",
        "macd_signal",
        "macd_hist",
        "trend_flag",
        "range_flag",
        "bb_width",
        "vwap_dist",
        "atr_14",
    ]
    volatility = [
        "vol_24",
        "bb_width",
        "atr_14",
        "roll_std_4",
        "roll_std_12",
        "roll_std_24",
        "z_4",
        "z_12",
        "z_24",
        "high_vol_flag",
        "low_vol_flag",
    ]
    long = [
        "ret_24h",
        "roll_mean_24",
        "roll_mean_72",
        "roll_mean_168",
        "z_24",
        "z_72",
        "z_168",
        "vwap_24",
        "vwap_dist",
        "vol_24",
    ]
    signal = [
        "ret_1c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "macd",
        "macd_hist",
        "rsi_14",
        "rsi_21",
    ]
    price_action = [
        "ret_1c",
        "ret_2c",
        "ret_3c",
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "roll_std_4",
        "roll_std_12",
        "roll_std_24",
        "body_pct",
        "range_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "close_location",
        "day_open_dist",
    ]
    mean_revert = [
        "z_4",
        "z_12",
        "z_24",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "bb_width",
    ]
    vwap_flow = [
        "vwap_24",
        "vwap_dist",
        "vol_chg",
        "vol_mean_24",
        "vol_std_24",
        "ret_1h",
        "ret_4h",
    ]
    trend_longer = [
        "roll_mean_72",
        "roll_mean_168",
        "z_72",
        "z_168",
        "macd",
        "macd_signal",
        "trend_flag",
        "range_flag",
        "day_of_week",
        "gap_from_prev_close",
        "vol_ratio_24_168",
    ]
    micro_momentum = [
        "ret_1c",
        "ret_1h",
        "roll_mean_4",
        "roll_std_4",
        "z_4",
        "macd_hist",
    ]
    minimal = ["ret_1c", "ret_1h", "ret_4h", "ret_24h"]
    session = [
        "minutes_from_open",
        "is_opening_hour",
        "is_closing_hour",
        "day_of_week",
        "gap_from_prev_close",
        "day_open_dist",
        "prev_day_high_break",
        "prev_day_low_break",
        "ret_1c",
        "ret_1h",
        "vol_24",
    ]
    context = [
        "body_pct",
        "range_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "close_location",
        "day_open_dist",
        "prev_day_high_break",
        "prev_day_low_break",
        "vol_ratio_24_168",
        "gap_from_prev_close",
        "ret_2c",
        "ret_3c",
    ]

    sets = {
        "base": base,
        "vol": vol,
        "momentum": momentum,
        "volume": volume,
        "trend": trend,
        "volatility": volatility,
        "long": long,
        "signal": signal,
        "price_action": price_action,
        "mean_revert": mean_revert,
        "vwap_flow": vwap_flow,
        "trend_longer": trend_longer,
        "micro_momentum": micro_momentum,
        "minimal": minimal,
        "session": session,
        "context": context,
    }

    cleaned = {}
    for k, v in sets.items():
        cleaned[k] = [c for c in v if c in cols]
    return cleaned


def make_supervised(
    df: pd.DataFrame,
    candle_minutes: int = 60,
    feature_windows_hours: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    df = compute_features(df, candle_minutes=candle_minutes, feature_windows_hours=feature_windows_hours)
    df = _annotate_model_context(df, candle_minutes)
    target_label = _target_label_for_minutes(candle_minutes)
    df[target_label] = df["ret_1c"].shift(-1)

    # Keep raw log-return target for evaluation/trading math.
    df["y_ret_raw"] = df[target_label]
    df["y_ret"] = df["y_ret_raw"]
    df["y_dir"] = (df["y_ret_raw"] > 0).astype(int)

    # Optional modeling target normalization by recent volatility.
    target_mode = os.getenv("RETURN_TARGET_MODE", "volnorm_logret").strip().lower()
    if target_mode in {"volnorm", "volnorm_logret", "normalized"}:
        df["y_ret_model"] = df["y_ret_raw"] / (df["target_scale"] + 1e-12)
    else:
        df["y_ret_model"] = df["y_ret_raw"]

    # Synthetic repair/gap-fill rows can carry zero volume. Keep them available for
    # storage continuity, but exclude them from supervised training samples.
    if "volume" in df.columns:
        df = df.loc[df["volume"] > 0]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df
