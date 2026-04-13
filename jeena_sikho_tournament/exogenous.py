from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_sources import _aggregate_ohlcv, fetch_yfinance
from .market_calendar import load_nse_holidays
from .validator import assess_freshness


@dataclass(frozen=True)
class PublicSignalSpec:
    alias: str
    symbol: str
    market: str = "global"
    auto_adjust: bool = False


def exogenous_feeds_enabled() -> bool:
    return _truthy(os.getenv("EXOGENOUS_FEEDS_ENABLE"))


def event_features_enabled() -> bool:
    return _truthy(os.getenv("EVENT_FEATURES_ENABLE"))


def resolve_public_signal_specs() -> List[PublicSignalSpec]:
    if not exogenous_feeds_enabled():
        return []
    specs = [
        PublicSignalSpec("nifty", os.getenv("EXOGENOUS_NIFTY_SYMBOL", "^NSEI").strip(), market="nse"),
        PublicSignalSpec("vix", os.getenv("EXOGENOUS_VIX_SYMBOL", "^INDIAVIX").strip(), market="nse"),
        PublicSignalSpec("usdinr", os.getenv("EXOGENOUS_USDINR_SYMBOL", "INR=X").strip(), market="global"),
    ]
    sector_symbol = os.getenv("EXOGENOUS_SECTOR_SYMBOL", "").strip()
    if sector_symbol:
        sector_market = os.getenv("EXOGENOUS_SECTOR_MARKET", "nse").strip().lower() or "nse"
        specs.append(PublicSignalSpec("sector", sector_symbol, market=sector_market))
    return [spec for spec in specs if spec.symbol]


def build_exogenous_feature_frame(
    index: pd.DatetimeIndex,
    candle_minutes: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    target_index = _coerce_utc_index(index)
    empty = pd.DataFrame(index=target_index)
    metadata: Dict[str, object] = {
        "signals": {},
        "event_calendar": {"enabled": event_features_enabled()},
        "enabled": exogenous_feeds_enabled() or event_features_enabled(),
    }
    if target_index.empty:
        return empty, metadata

    out = pd.DataFrame(index=target_index)

    for spec in resolve_public_signal_specs():
        signal_frame, signal_meta = load_public_signal_frame(spec, target_index, candle_minutes)
        metadata["signals"][spec.alias] = signal_meta
        if signal_frame.empty:
            continue
        signal_features = _featurize_signal_frame(signal_frame, spec.alias, candle_minutes)
        if signal_features.empty:
            continue
        aligned = signal_features.reindex(target_index).ffill()
        base_col = next((col for col in aligned.columns if col.endswith("_ret_1c")), None)
        signal_meta["joined_coverage_ratio"] = float(aligned[base_col].notna().mean()) if base_col else 0.0
        out = out.join(aligned, how="left")

    event_frame, event_meta = load_event_calendar_features(target_index)
    metadata["event_calendar"] = event_meta
    if not event_frame.empty:
        out = out.join(event_frame, how="left")

    return out, metadata


def refresh_exogenous_caches(timeframes: List[str], candle_minutes_map: Dict[str, int]) -> Dict[str, object]:
    target_now = pd.Timestamp.now(tz="UTC").ceil("min")
    report: Dict[str, object] = {
        "generated_at": target_now.isoformat(),
        "signals_enabled": exogenous_feeds_enabled(),
        "event_features_enabled": event_features_enabled(),
        "timeframes": {},
    }
    if not exogenous_feeds_enabled():
        return report

    lookback_hours = _int_env("EXOGENOUS_LOOKBACK_HOURS", 24 * 14)
    for timeframe in timeframes:
        candle_minutes = int(candle_minutes_map.get(timeframe, 60))
        start = target_now - pd.Timedelta(hours=max(24, lookback_hours))
        idx = pd.date_range(start=start, end=target_now, freq=f"{max(1, candle_minutes)}min", tz="UTC")
        timeframe_report: Dict[str, object] = {"signals": {}}
        for spec in resolve_public_signal_specs():
            _, meta = load_public_signal_frame(spec, idx, candle_minutes)
            timeframe_report["signals"][spec.alias] = meta
        _, event_meta = load_event_calendar_features(idx)
        timeframe_report["event_calendar"] = event_meta
        report["timeframes"][timeframe] = timeframe_report
    return report


def refresh_public_signal_caches(
    candle_minutes_list: List[int] | int,
    *,
    now_utc: pd.Timestamp | None = None,
) -> Dict[str, object]:
    """Refresh cached public-signal files ahead of training.

    This keeps exogenous feeds outside the core OHLCV store and primes the
    per-feed cache/meta files used later at feature-build time. Event-calendar
    validation is also exercised here so nightly runs can surface contract
    issues early without mutating the core market dataset.
    """
    if isinstance(candle_minutes_list, int):
        candle_minutes_values = [int(candle_minutes_list)]
    else:
        candle_minutes_values = sorted({max(1, int(value)) for value in candle_minutes_list})
    ts = pd.Timestamp(now_utc or pd.Timestamp.utcnow())
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    lookback_hours = max(24, _int_env("EXOGENOUS_LOOKBACK_HOURS", 24 * 14))
    start = ts - pd.Timedelta(hours=lookback_hours)
    target_index_by_candle = {
        candle_minutes: pd.date_range(start=start, end=ts, freq=f"{max(1, int(candle_minutes))}min", tz="UTC")
        for candle_minutes in candle_minutes_values
    }

    report: Dict[str, object] = {
        "generated_at": ts.isoformat(),
        "enabled": exogenous_feeds_enabled() or event_features_enabled(),
        "signals": {},
        "event_calendar": {"enabled": event_features_enabled()},
    }

    for candle_minutes, target_index in target_index_by_candle.items():
        signal_reports: Dict[str, object] = {}
        for spec in resolve_public_signal_specs():
            _, meta = load_public_signal_frame(spec, target_index, candle_minutes)
            signal_reports[spec.alias] = meta
        report["signals"][str(candle_minutes)] = signal_reports

    if event_features_enabled():
        daily_index = pd.date_range(start=start.normalize(), end=ts.normalize(), freq="1D", tz="UTC")
        _, event_meta = load_event_calendar_features(daily_index)
        event_meta["contract"] = {
            "required": ["event_date"],
            "optional": ["symbol", "severity"],
        }
        report["event_calendar"] = event_meta
    return report


def load_public_signal_frame(
    spec: PublicSignalSpec,
    index: pd.DatetimeIndex,
    candle_minutes: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    target_index = _coerce_utc_index(index)
    lookback_hours = _int_env("EXOGENOUS_LOOKBACK_HOURS", 24 * 14)
    start = target_index.min() - pd.Timedelta(hours=max(24, lookback_hours))
    end = target_index.max() + pd.Timedelta(minutes=max(60, int(candle_minutes)))

    csv_path, meta_path = _cache_paths(spec.alias, candle_minutes)
    cached = _load_cached_signal_frame(csv_path)
    cache_status = "cache_hit"
    needs_refresh = (
        cached.empty
        or cached.index.min() > start
        or cached.index.max() < (end - pd.Timedelta(minutes=max(60, int(candle_minutes))))
    )

    frame = cached
    if needs_refresh:
        refreshed = _fetch_signal_frame(spec, start, end, candle_minutes)
        if not refreshed.empty:
            frame = refreshed
            cache_status = "refreshed"
            _write_cached_signal_frame(frame, csv_path)
        elif not cached.empty:
            cache_status = "cache_fallback"
        else:
            cache_status = "unavailable"

    frame = frame.sort_index()
    if not frame.empty:
        frame = frame.loc[(frame.index >= start) & (frame.index <= end)]

    freshness = _signal_freshness(frame, candle_minutes, spec.market == "nse")
    metadata = {
        "alias": spec.alias,
        "symbol": spec.symbol,
        "market": spec.market,
        "enabled": True,
        "cache_status": cache_status,
        "cache_path": str(csv_path),
        "meta_path": str(meta_path),
        "record_count": int(len(frame)),
        "source": "yfinance",
        **freshness,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame, metadata


def load_event_calendar_features(index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load optional event features from a flat JSON/CSV calendar contract.

    Contract:
    - required field: ``event_date``
    - optional fields: ``symbol`` and ``severity``

    JSON may be either a top-level list of rows or ``{"events": [...]}``.
    CSV/JSON rows outside the active symbol filter are ignored.
    Missing ``severity`` defaults to ``1.0``.
    """
    target_index = _coerce_utc_index(index)
    meta: Dict[str, object] = {
        "enabled": event_features_enabled(),
        "available": False,
        "path": None,
        "record_count": 0,
        "matched_days": 0,
        "contract": {
            "required": ["event_date"],
            "optional": ["symbol", "severity"],
        },
    }
    if not event_features_enabled() or target_index.empty:
        return pd.DataFrame(index=target_index), meta

    raw_path = os.getenv("EVENT_CALENDAR_FILE", "").strip()
    if not raw_path:
        return pd.DataFrame(index=target_index), meta
    path = Path(raw_path)
    meta["path"] = str(path)
    if not path.exists():
        return pd.DataFrame(index=target_index), meta

    events = _read_event_calendar(path)
    if events.empty:
        return pd.DataFrame(index=target_index), meta

    symbol_filter = {
        token
        for token in [
            os.getenv("MARKET_YFINANCE_SYMBOL", "").strip().upper(),
            os.getenv("MARKET_SYMBOL", "").strip().upper(),
        ]
        if token
    }
    if "symbol" in events.columns and symbol_filter:
        normalized = events["symbol"].astype(str).str.upper()
        filtered = events.loc[normalized.isin(symbol_filter)]
        if not filtered.empty:
            events = filtered

    if events.empty:
        return pd.DataFrame(index=target_index), meta

    date_col = "event_date"
    events[date_col] = pd.to_datetime(events[date_col], errors="coerce")
    events = events.dropna(subset=[date_col])
    if events.empty:
        return pd.DataFrame(index=target_index), meta

    if events[date_col].dt.tz is None:
        events[date_col] = events[date_col].dt.tz_localize("Asia/Kolkata")
    else:
        events[date_col] = events[date_col].dt.tz_convert("Asia/Kolkata")
    events[date_col] = events[date_col].dt.normalize()
    if "severity" in events.columns:
        severity = pd.to_numeric(events["severity"], errors="coerce").fillna(1.0)
    else:
        severity = pd.Series(1.0, index=events.index, dtype=float)
    grouped = (
        pd.DataFrame({"event_date": events[date_col], "severity": severity})
        .groupby("event_date")
        .agg(event_severity=("severity", "max"))
        .sort_index()
    )
    bar_dates = target_index.tz_convert("Asia/Kolkata").normalize()
    unique_bar_dates = pd.DatetimeIndex(pd.Index(bar_dates.unique()), tz="Asia/Kolkata")
    daily = pd.DataFrame(index=unique_bar_dates)
    daily["event_severity"] = grouped["event_severity"].reindex(unique_bar_dates).fillna(0.0)
    daily["has_known_event"] = daily["event_severity"].gt(0).astype(float)
    event_mask = daily["has_known_event"].astype(bool).to_numpy()
    daily["days_to_known_event"] = _distance_to_event(event_mask)
    daily["days_since_known_event"] = _distance_since_event(event_mask)
    daily["known_event_window"] = (
        daily["has_known_event"].astype(bool)
        | daily["days_to_known_event"].between(0, 1, inclusive="both")
        | daily["days_since_known_event"].between(0, 1, inclusive="both")
    ).astype(float)

    aligned = daily.loc[bar_dates].copy()
    aligned.index = target_index

    meta.update(
        {
            "available": True,
            "record_count": int(len(events)),
            "matched_days": int(daily["has_known_event"].sum()),
        }
    )
    return aligned, meta


def _fetch_signal_frame(
    spec: PublicSignalSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    candle_minutes: int,
) -> pd.DataFrame:
    raw = fetch_yfinance(
        spec.symbol,
        start.to_pydatetime(),
        end.to_pydatetime(),
        auto_adjust=spec.auto_adjust,
        source_name=f"exo_{spec.alias}",
    )
    if raw.empty:
        return pd.DataFrame()
    if int(candle_minutes) != 60:
        raw = _aggregate_ohlcv(raw, int(candle_minutes), spec.symbol)
    return _normalize_ohlcv_frame(raw)


def _featurize_signal_frame(frame: pd.DataFrame, alias: str, candle_minutes: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    close = frame["close"].astype(float)
    safe_close = close.where(close > 0)
    log_close = np.log(safe_close)
    out = pd.DataFrame(index=frame.index)
    ret_col = f"exo_{alias}_ret_1c"
    out[ret_col] = log_close.diff(1)
    out[f"exo_{alias}_ret_4h"] = log_close.diff(_bars_for_hours(4, candle_minutes))
    out[f"exo_{alias}_ret_24h"] = log_close.diff(_bars_for_hours(24, candle_minutes))

    roll_window = max(3, _bars_for_hours(24, candle_minutes))
    roll_std = out[ret_col].rolling(roll_window, min_periods=min(roll_window, 4)).std(ddof=0)
    ema = close.ewm(span=roll_window, adjust=False, min_periods=min(roll_window, 4)).mean()
    out[f"exo_{alias}_z_24"] = out[ret_col] / (roll_std + 1e-9)
    out[f"exo_{alias}_ema_dist_24"] = (close - ema) / (ema.abs() + 1e-9)

    if "volume" in frame.columns:
        volume = frame["volume"].astype(float)
        vol_mean = volume.rolling(roll_window, min_periods=min(roll_window, 4)).mean()
        vol_std = volume.rolling(roll_window, min_periods=min(roll_window, 4)).std(ddof=0)
        out[f"exo_{alias}_vol_z_24"] = (volume - vol_mean) / (vol_std + 1e-9)

    if alias == "vix":
        median = close.rolling(roll_window, min_periods=min(roll_window, 4)).median()
        out["exo_vix_high_regime"] = (close > median).astype(float)

    return out.replace([np.inf, -np.inf], np.nan)


def _cache_paths(alias: str, candle_minutes: int) -> Tuple[Path, Path]:
    cache_dir = _app_data_dir() / "exogenous"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / f"{alias}_{int(candle_minutes)}m"
    return base.with_suffix(".csv"), base.with_suffix(".meta.json")


def _load_cached_signal_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return _normalize_ohlcv_frame(df)


def _write_cached_signal_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reset_index().rename(columns={"index": "timestamp_utc"})
    out.to_csv(path, index=False)


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "source"])
    out = frame.copy()
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc")
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.loc[~out.index.isna()]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "source" not in out.columns:
        out["source"] = "yfinance"
    required = ["open", "high", "low", "close", "volume", "source"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        return pd.DataFrame(columns=required)
    out = out[required].sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    return out


def _signal_freshness(frame: pd.DataFrame, candle_minutes: int, nse_mode: bool) -> Dict[str, object]:
    if frame.empty:
        return {
            "available": False,
            "latest_timestamp": None,
            "expected_latest_timestamp": None,
            "stale": True,
            "lag_slots": None,
        }
    holidays = load_nse_holidays(_app_data_dir()) if nse_mode else set()
    freshness = assess_freshness(frame, candle_minutes, nse_mode=nse_mode, holidays=holidays)
    freshness["available"] = True
    return freshness


def _read_event_calendar(path: Path) -> pd.DataFrame:
    """Read the flat event-calendar contract from JSON or CSV.

    Supported row keys:
    - required: ``event_date`` (aliases ``date``, ``timestamp``, ``event_ts``)
    - optional: ``symbol``, ``severity``
    """
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows = payload.get("events", [])
            else:
                rows = payload
            df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df
    rename_map = {}
    for candidate in ["event_date", "date", "timestamp", "event_ts"]:
        if candidate in df.columns:
            rename_map[candidate] = "event_date"
            break
    if rename_map:
        df = df.rename(columns=rename_map)
    if "event_date" not in df.columns:
        return pd.DataFrame()
    return df


def _distance_to_event(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[0], np.nan, dtype=float)
    next_event = None
    for idx in range(mask.shape[0] - 1, -1, -1):
        if mask[idx]:
            next_event = idx
        if next_event is not None:
            out[idx] = float(next_event - idx)
    return out


def _distance_since_event(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[0], np.nan, dtype=float)
    last_event = None
    for idx in range(mask.shape[0]):
        if mask[idx]:
            last_event = idx
        if last_event is not None:
            out[idx] = float(idx - last_event)
    return out


def _bars_for_hours(hours: int, candle_minutes: int) -> int:
    if candle_minutes <= 0:
        return max(1, hours)
    return max(1, int(round((hours * 60) / candle_minutes)))


def _coerce_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _app_data_dir() -> Path:
    return Path(os.getenv("APP_DATA_DIR", "data"))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
