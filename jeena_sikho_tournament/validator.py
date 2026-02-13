from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Set

import pandas as pd

from .market_calendar import IST, is_nse_trading_day


@dataclass
class DataQualityReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


def _expected_index_24x7(df: pd.DataFrame, candle_minutes: int) -> pd.DatetimeIndex:
    freq = f"{max(1, int(candle_minutes))}min"
    return pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")


def _expected_index_nse(df: pd.DataFrame, candle_minutes: int, holidays: Set) -> pd.DatetimeIndex:
    step = max(1, int(candle_minutes))
    start = df.index.min().tz_convert(IST).date()
    end = df.index.max().tz_convert(IST).date()
    all_slots = []
    cur = pd.Timestamp(start, tz=IST)
    end_ts = pd.Timestamp(end, tz=IST)
    open_min = 9 * 60 + 15
    close_min = 15 * 60 + 30
    while cur.date() <= end_ts.date():
        if is_nse_trading_day(cur.to_pydatetime(), holidays):
            day_open = cur.replace(hour=9, minute=15, second=0, microsecond=0)
            for mins in range(open_min, close_min + 1, step):
                hh, mm = divmod(mins, 60)
                slot = day_open.replace(hour=hh, minute=mm)
                all_slots.append(slot.tz_convert("UTC"))
        cur = cur + timedelta(days=1)
    if not all_slots:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(all_slots, tz="UTC")


def validate_ohlcv_quality(
    df: pd.DataFrame,
    candle_minutes: int,
    *,
    nse_mode: bool,
    holidays: Set,
    max_missing_ratio: float = 0.15,
) -> DataQualityReport:
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, float] = {}

    if df.empty:
        return DataQualityReport(False, errors=["empty_dataset"], warnings=warnings, stats=stats)
    if df.index.tz is None:
        errors.append("timestamp_index_must_be_tz_aware_utc")
        return DataQualityReport(False, errors=errors, warnings=warnings, stats=stats)

    if df.index.duplicated().any():
        errors.append("duplicate_timestamps_found")
    if not df.index.is_monotonic_increasing:
        errors.append("timestamps_not_monotonic")

    required = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        errors.append(f"missing_columns:{','.join(missing_cols)}")
        return DataQualityReport(False, errors=errors, warnings=warnings, stats=stats)

    if df[required].isna().any().any():
        errors.append("ohlcv_contains_nan")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("non_positive_price_detected")

    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    if bad_high > 0:
        errors.append(f"high_below_ohlc:{int(bad_high)}")
    if bad_low > 0:
        errors.append(f"low_above_ohlc:{int(bad_low)}")

    if nse_mode:
        idx_local = df.index.tz_convert(IST)
        minute_of_day = idx_local.hour * 60 + idx_local.minute
        valid_trading = (minute_of_day >= 9 * 60 + 15) & (minute_of_day <= 15 * 60 + 30)
        valid_weekday = idx_local.weekday < 5
        valid_holiday = ~pd.Series(idx_local.date).isin(holidays).to_numpy()
        if not (valid_trading & valid_weekday & valid_holiday).all():
            errors.append("nse_session_boundary_violation")

        step = max(1, int(candle_minutes))
        if ((minute_of_day - (9 * 60 + 15)) % step != 0).any():
            errors.append("nse_interval_alignment_violation")

    expected = _expected_index_nse(df, candle_minutes, holidays) if nse_mode else _expected_index_24x7(df, candle_minutes)
    if len(expected) > 0:
        missing = len(expected.difference(df.index))
        ratio = float(missing / max(1, len(expected)))
        stats["missing_intervals"] = float(missing)
        stats["expected_intervals"] = float(len(expected))
        stats["missing_ratio"] = ratio
        if ratio > max_missing_ratio:
            errors.append(f"missing_ratio_too_high:{ratio:.4f}")
        elif ratio > max_missing_ratio * 0.5:
            warnings.append(f"missing_ratio_warning:{ratio:.4f}")

    return DataQualityReport(ok=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)
