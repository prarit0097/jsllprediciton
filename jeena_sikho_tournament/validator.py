from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set

import pandas as pd

from .market_calendar import IST, NSE_OPEN_MIN, is_nse_trading_day, last_nse_slot_minute


@dataclass
class DataQualityReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


def expected_latest_timestamp(
    now_utc: datetime,
    candle_minutes: int,
    *,
    nse_mode: bool,
    holidays: Set,
) -> datetime:
    step = max(1, int(candle_minutes))
    if not nse_mode:
        ts = now_utc.astimezone(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=step)
        minute = (ts.minute // step) * step
        return ts.replace(minute=minute, second=0, microsecond=0)

    now_ist = now_utc.astimezone(IST).replace(second=0, microsecond=0)
    slot_min = last_nse_slot_minute(step)
    close_hh, close_mm = divmod(slot_min, 60)

    if not is_nse_trading_day(now_ist, holidays):
        cur = (now_ist - timedelta(days=1)).replace(hour=close_hh, minute=close_mm, second=0, microsecond=0)
        while not is_nse_trading_day(cur, holidays):
            cur = (cur - timedelta(days=1)).replace(hour=close_hh, minute=close_mm, second=0, microsecond=0)
        return cur.astimezone(timezone.utc)

    minute_of_day = now_ist.hour * 60 + now_ist.minute
    if minute_of_day < NSE_OPEN_MIN:
        cur = (now_ist - timedelta(days=1)).replace(hour=close_hh, minute=close_mm, second=0, microsecond=0)
        while not is_nse_trading_day(cur, holidays):
            cur = (cur - timedelta(days=1)).replace(hour=close_hh, minute=close_mm, second=0, microsecond=0)
        return cur.astimezone(timezone.utc)

    if minute_of_day > slot_min:
        return now_ist.replace(hour=close_hh, minute=close_mm, second=0, microsecond=0).astimezone(timezone.utc)

    ts = now_ist - timedelta(minutes=step)
    start_hh, start_mm = divmod(NSE_OPEN_MIN, 60)
    day_open = now_ist.replace(hour=start_hh, minute=start_mm, second=0, microsecond=0)
    delta = int((ts - day_open).total_seconds() // 60)
    slot = (max(0, delta) // step) * step
    expected = day_open + timedelta(minutes=slot)
    return expected.astimezone(timezone.utc)


def assess_freshness(
    df: pd.DataFrame,
    candle_minutes: int,
    *,
    nse_mode: bool,
    holidays: Set,
    now_utc: datetime | None = None,
) -> Dict[str, object]:
    now = now_utc or datetime.now(timezone.utc)
    expected_latest = expected_latest_timestamp(now, candle_minutes, nse_mode=nse_mode, holidays=holidays)
    latest = None
    if not df.empty:
        latest = df.index.max()
        if latest is not None and latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
    stale = latest is None or latest < expected_latest
    lag_slots = None
    if latest is not None:
        lag_seconds = max(0.0, (expected_latest - latest).total_seconds())
        lag_slots = int(round(lag_seconds / max(60.0, candle_minutes * 60.0)))
    return {
        "expected_latest_timestamp": expected_latest.isoformat(),
        "latest_timestamp": latest.isoformat() if latest is not None else None,
        "stale": stale,
        "lag_slots": lag_slots,
    }


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
            if step >= 1440:
                slot = day_open.replace(hour=15, minute=30)
                all_slots.append(slot.tz_convert("UTC"))
            else:
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
        if step >= 1440:
            align_ok = minute_of_day == (15 * 60 + 30)
        else:
            align_ok = ((minute_of_day - (9 * 60 + 15)) % step) == 0
        if (~align_ok).any():
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

    freshness = assess_freshness(df, candle_minutes, nse_mode=nse_mode, holidays=holidays)
    stats["expected_latest_timestamp"] = freshness["expected_latest_timestamp"]
    stats["latest_timestamp"] = freshness["latest_timestamp"]
    stats["stale"] = float(1.0 if freshness["stale"] else 0.0)
    if freshness.get("lag_slots") is not None:
        stats["lag_slots"] = float(freshness["lag_slots"])
    if freshness["stale"]:
        errors.append("latest_bar_stale")

    return DataQualityReport(ok=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)
