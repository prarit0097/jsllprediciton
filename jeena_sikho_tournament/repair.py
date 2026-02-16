from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import TournamentConfig
from .data_sources import fetch_and_stitch
from .market_calendar import load_nse_holidays
from .storage import Storage
from .validator import validate_ohlcv_quality


def _parse_timeframes(value: str) -> List[str]:
    tokens: List[str] = []
    for p in value.replace(";", ",").replace("|", ",").split(","):
        t = p.strip()
        if t:
            tokens.append(t)
    out: List[str] = []
    seen = set()
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _resolve_timeframes(config: TournamentConfig) -> List[str]:
    env_list = os.getenv("MARKET_TIMEFRAMES") or os.getenv("TIMEFRAMES")
    if env_list:
        parsed = _parse_timeframes(env_list)
        if parsed:
            return parsed
    return [config.timeframe]


def _timeframe_to_minutes(timeframe: str, fallback: int) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("d") and tf[:-1].isdigit():
        return int(tf[:-1]) * 24 * 60
    return fallback


def _config_for_timeframe(base: TournamentConfig, timeframe: str) -> TournamentConfig:
    cfg = TournamentConfig()
    cfg.__dict__.update(base.__dict__)
    cfg.timeframe = timeframe
    cfg.candle_minutes = _timeframe_to_minutes(timeframe, base.candle_minutes)
    cfg.ohlcv_table = "ohlcv" if cfg.candle_minutes == 60 else f"ohlcv_{cfg.candle_minutes}m"
    cfg.registry_path = base.data_dir / f"registry_{cfg.candle_minutes}m.json"
    cfg.log_path = base.data_dir / f"tournament_{cfg.candle_minutes}m.log"
    return cfg


def _expected_index(df: pd.DataFrame, candle_minutes: int, config: TournamentConfig) -> pd.DatetimeIndex:
    from .data_sources import _expected_nse_slots

    start = df.index.min()
    end = df.index.max()
    if start is None or end is None:
        return pd.DatetimeIndex([], tz="UTC")
    if (config.yfinance_symbol or "").upper().endswith((".NS", ".BO")):
        holidays = load_nse_holidays(config.data_dir)
        return _expected_nse_slots(start, end, candle_minutes, holidays)
    return pd.date_range(start=start, end=end, freq=f"{max(1, candle_minutes)}min", tz="UTC")


def _missing_slots(storage: Storage, lookback_days: int, config: TournamentConfig) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    df = storage.load().sort_index()
    if df.empty:
        return pd.DatetimeIndex([], tz="UTC"), df
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
    recent = df.loc[df.index >= cutoff]
    if recent.empty:
        recent = df
    expected = _expected_index(recent, config.candle_minutes, config)
    missing = expected.difference(recent.index)
    return missing, recent


def repair_timeframe_data(config: TournamentConfig, lookback_days: int = 120) -> Dict[str, Any]:
    storage = Storage(config.db_path, config.ohlcv_table)
    storage.init_db()
    missing, recent = _missing_slots(storage, lookback_days, config)
    before_missing = int(len(missing))
    inserted = 0
    fetched_rows = 0
    if before_missing > 0:
        # Fetch only covering missing span.
        start = (missing.min() - timedelta(minutes=max(1, config.candle_minutes) * 2)).to_pydatetime()
        merged, _ = fetch_and_stitch(config.symbol, config.yfinance_symbol, start, config.timeframe, config.candle_minutes)
        if not merged.empty:
            fetched_rows = int(len(merged))
            m = merged.copy()
            m["timestamp_utc"] = pd.to_datetime(m["timestamp_utc"], utc=True)
            m = m.set_index("timestamp_utc")
            only_missing = m.loc[m.index.isin(missing)]
            inserted = int(len(only_missing))
            if not only_missing.empty:
                storage.upsert(only_missing)
    refreshed = storage.load().sort_index()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
    validate_df = refreshed.loc[refreshed.index >= cutoff] if not refreshed.empty else refreshed
    after_missing, _ = _missing_slots(storage, lookback_days, config)
    holidays = load_nse_holidays(config.data_dir)
    dq = validate_ohlcv_quality(
        validate_df,
        config.candle_minutes,
        nse_mode=(config.yfinance_symbol or "").upper().endswith((".NS", ".BO")),
        holidays=holidays,
        max_missing_ratio=float(os.getenv("MAX_MISSING_RATIO", "0.15")),
    )
    return {
        "timeframe": config.timeframe,
        "table": config.ohlcv_table,
        "before_missing": before_missing,
        "after_missing": int(len(after_missing)),
        "inserted": inserted,
        "fetched_rows": fetched_rows,
        "dq_ok": dq.ok,
        "dq_errors": dq.errors,
        "dq_warnings": dq.warnings,
        "dq_stats": dq.stats,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def run_nightly_repair(base_config: TournamentConfig) -> Dict[str, Any]:
    lookback_days = int(os.getenv("REPAIR_LOOKBACK_DAYS", "120"))
    frames = _resolve_timeframes(base_config)
    reports: List[Dict[str, Any]] = []
    for tf in frames:
        cfg = _config_for_timeframe(base_config, tf)
        reports.append(repair_timeframe_data(cfg, lookback_days=lookback_days))
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "reports": reports,
    }
    data_dir = base_config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    latest = data_dir / "repair_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    hist = data_dir / "repair_reports.jsonl"
    with hist.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return payload
