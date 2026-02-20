import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

from .market_calendar import IST, is_nse_trading_day, load_nse_completeness_exclusions, load_nse_holidays
from .kite_client import fetch_kite_ohlcv, is_kite_enabled

LOGGER = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    earliest: Optional[pd.Timestamp]
    total_candles: int
    missing_intervals: int
    interval_minutes: int


def _to_utc(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True)
    return ts


def _as_ohlcv(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.rename_axis("timestamp_utc").reset_index()
    out["timestamp_utc"] = _to_utc(out["timestamp_utc"])
    out["source"] = source
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    out = out.dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])
    return out[["timestamp_utc", "open", "high", "low", "close", "volume", "source"]]


def fetch_binance(symbol: str, timeframe: str, start: dt.datetime) -> pd.DataFrame:
    try:
        import ccxt  # type: ignore
    except Exception:
        return pd.DataFrame()

    supported = {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    }
    if timeframe not in supported:
        return pd.DataFrame()

    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int(start.timestamp() * 1000)
    all_rows = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        except Exception:
            return pd.DataFrame()
        if not candles:
            break
        all_rows.extend(candles)
        last_ts = candles[-1][0]
        since_ms = last_ts + 1
        if len(candles) < 1000:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).set_index("timestamp_utc")
    return _as_ohlcv(df, "binance")


def _expected_nse_slots(start_utc: pd.Timestamp, end_utc: pd.Timestamp, interval_minutes: int, holidays: Set[dt.date]) -> pd.DatetimeIndex:
    step = max(1, int(interval_minutes))
    start_ist = start_utc.tz_convert(IST)
    end_ist = end_utc.tz_convert(IST)
    day = start_ist.date()
    excluded = load_nse_completeness_exclusions()
    slots: List[pd.Timestamp] = []
    while day <= end_ist.date():
        cur = dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST)
        if day in excluded:
            day = day + dt.timedelta(days=1)
            continue
        if is_nse_trading_day(cur, holidays):
            if step >= 1440:
                ts = dt.datetime.combine(day, dt.time(15, 30), tzinfo=IST)
                slots.append(pd.Timestamp(ts).tz_convert("UTC"))
            else:
                minute = 9 * 60 + 15
                while minute <= (15 * 60 + 30):
                    hh, mm = divmod(minute, 60)
                    ts = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=IST)
                    slots.append(pd.Timestamp(ts).tz_convert("UTC"))
                    minute += step
        day = day + dt.timedelta(days=1)
    if not slots:
        return pd.DatetimeIndex([], tz="UTC")
    idx = pd.DatetimeIndex(slots, tz="UTC")
    idx = idx[(idx >= start_utc) & (idx <= end_utc)]
    if step < 1440:
        return idx

    # For multi-day horizons (2d/3d...), expected slots are every N trading-day close.
    n_trading_days = max(1, int(np.ceil(step / 1440.0)))
    close_mask = (idx.tz_convert(IST).hour == 15) & (idx.tz_convert(IST).minute == 30)
    close_slots = idx[close_mask]
    if len(close_slots) == 0:
        return close_slots
    return close_slots[::n_trading_days]


def _enforce_nse_slots(df: pd.DataFrame, interval_minutes: int, symbol: str) -> pd.DataFrame:
    sym = (symbol or "").strip().upper()
    if not (sym.endswith(".NS") or sym.endswith(".BO")):
        return df
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    holidays = load_nse_holidays(Path(os.getenv("APP_DATA_DIR", "data")))
    idx_local = out.index.tz_convert(IST)
    minute_of_day = idx_local.hour * 60 + idx_local.minute
    valid_weekday = idx_local.weekday < 5
    valid_holiday = ~pd.Series(idx_local.date, index=out.index).isin(holidays).to_numpy()
    valid_session = (minute_of_day >= (9 * 60 + 15)) & (minute_of_day <= (15 * 60 + 30))
    step = max(1, int(interval_minutes))
    if step >= 1440:
        valid_align = minute_of_day == (15 * 60 + 30)
    else:
        valid_align = ((minute_of_day - (9 * 60 + 15)) % step) == 0
    mask = valid_weekday & valid_holiday & valid_session & valid_align
    out = out.loc[mask]
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_yfinance(symbol: str, start: dt.datetime, end: dt.datetime, *, auto_adjust: bool = False, source_name: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame()

    for name in ["yfinance", "yfinance.base", "yfinance.shared", "yfinance.ticker"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Yahoo 1h data is limited (~730 days), so chunk requests.
    all_parts = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + dt.timedelta(days=730), end)
        try:
            part = yf.download(
                symbol,
                start=chunk_start,
                end=chunk_end,
                interval="60m",
                auto_adjust=auto_adjust,
                progress=False,
            )
        except Exception:
            part = pd.DataFrame()
        if not part.empty:
            all_parts.append(part)
        chunk_start = chunk_end
    if not all_parts:
        return pd.DataFrame()
    data = pd.concat(all_parts)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(c[0]) for c in data.columns]
    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[["open", "high", "low", "close", "volume"]]
    data = data[~data.index.duplicated(keep="last")]
    data = _filter_nse_session(data, symbol)
    data = _enforce_nse_slots(data, 60, symbol)
    src = source_name or ("yfinance_adj" if auto_adjust else "yfinance")
    return _as_ohlcv(data, src)


def fetch_cryptocompare(symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    fsym, tsym = _split_symbol_for_cryptocompare(symbol)
    all_rows = []
    to_ts = int(end.timestamp())
    limit = 2000
    while True:
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "toTs": to_ts,
            "limit": limit,
        }
        api_key = _cryptocompare_key()
        if api_key:
            params["api_key"] = api_key
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break
        data = payload.get("Data", {}).get("Data", [])
        if not data:
            break
        all_rows.extend(data)
        oldest = data[0]["time"]
        to_ts = oldest - 1
        if dt.datetime.fromtimestamp(oldest, tz=dt.timezone.utc) <= start:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volumefrom": "volume"})
    df = df.set_index("timestamp_utc")
    df = df[["open", "high", "low", "close", "volume"]]
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    df = df.loc[df.index >= start_ts]
    return _as_ohlcv(df, "cryptocompare")


def fetch_cryptocompare_minute(symbol: str, start: dt.datetime, end: dt.datetime, aggregate: int) -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    fsym, tsym = _split_symbol_for_cryptocompare(symbol)
    all_rows = []
    to_ts = int(end.timestamp())
    limit = 2000
    while True:
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "toTs": to_ts,
            "limit": limit,
            "aggregate": aggregate,
        }
        api_key = _cryptocompare_key()
        if api_key:
            params["api_key"] = api_key
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break
        data = payload.get("Data", {}).get("Data", [])
        if not data:
            break
        all_rows.extend(data)
        oldest = data[0]["time"]
        to_ts = oldest - 1
        if dt.datetime.fromtimestamp(oldest, tz=dt.timezone.utc) <= start:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volumefrom": "volume"})
    df = df.set_index("timestamp_utc")
    df = df[["open", "high", "low", "close", "volume"]]
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    df = df.loc[df.index >= start_ts]
    return _as_ohlcv(df, "cryptocompare")


def stitch_sources(sources: List[pd.DataFrame], interval_minutes: int, symbol: str = "") -> Tuple[pd.DataFrame, CoverageReport]:
    if not sources:
        return pd.DataFrame(), CoverageReport(None, 0, 0, interval_minutes)
    non_empty = [s for s in sources if not s.empty]
    if not non_empty:
        return pd.DataFrame(), CoverageReport(None, 0, 0, interval_minutes)
    merged = pd.concat(non_empty, ignore_index=True)
    if merged.empty:
        return pd.DataFrame(), CoverageReport(None, 0, 0, interval_minutes)

    merged = merged.dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])
    priority = {"kite": -2, "binance": 2, "cryptocompare": 1, "yfinance": 0, "yfinance_adj": 3, "yfinance_fallback": 4}
    latest_by_source = merged.groupby("source")["timestamp_utc"].max().to_dict()
    freshest_src = max(latest_by_source, key=latest_by_source.get) if latest_by_source else None
    if freshest_src is not None:
        priority[freshest_src] = -1
    merged["priority"] = merged["source"].map(priority).fillna(9)
    merged = merged.sort_values(["timestamp_utc", "priority"])
    merged = merged.drop_duplicates(subset=["timestamp_utc"], keep="first")
    merged = merged.drop(columns=["priority"]).sort_values("timestamp_utc")

    merged = merged.set_index("timestamp_utc")
    earliest = merged.index.min()
    total = len(merged)

    freq = f"{interval_minutes}min"
    is_nse_symbol = (symbol or "").strip().upper().endswith(".NS") or (symbol or "").strip().upper().endswith(".BO")
    if is_nse_symbol:
        holidays = load_nse_holidays(Path(os.getenv("APP_DATA_DIR", "data")))
        full_range = _expected_nse_slots(earliest, merged.index.max(), interval_minutes, holidays)
    else:
        full_range = pd.date_range(start=earliest, end=merged.index.max(), freq=freq, tz="UTC")
    missing = len(full_range.difference(merged.index))

    return merged.reset_index(), CoverageReport(earliest, total, missing, interval_minutes)


def fetch_and_stitch(
    symbol: str,
    yfinance_symbol: str,
    start: dt.datetime,
    timeframe: str,
    interval_minutes: int,
) -> Tuple[pd.DataFrame, CoverageReport]:
    end = dt.datetime.now(dt.timezone.utc)
    is_nse_symbol = (yfinance_symbol or "").strip().upper().endswith(".NS") or (yfinance_symbol or "").strip().upper().endswith(".BO")
    if is_nse_symbol:
        nse_sources: List[pd.DataFrame] = []
        allow_yf_fallback = os.getenv("KITE_ALLOW_YFINANCE_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}

        if is_kite_enabled():
            try:
                kdf = fetch_kite_ohlcv(start=start, end=end, interval="60minute")
                if not kdf.empty and interval_minutes != 60:
                    kdf = _aggregate_ohlcv(kdf, interval_minutes, yfinance_symbol)
                if not kdf.empty:
                    nse_sources.append(kdf)
            except Exception as exc:
                LOGGER.warning("Kite OHLCV fetch failed; using fallback sources: %s", exc)

        if allow_yf_fallback or not nse_sources:
            ydf_primary = fetch_yfinance(yfinance_symbol, start, end, auto_adjust=False, source_name="yfinance")
            ydf_fallback = fetch_yfinance(yfinance_symbol, start, end, auto_adjust=True, source_name="yfinance_adj")
            if not ydf_primary.empty and interval_minutes != 60:
                ydf_primary = _aggregate_ohlcv(ydf_primary, interval_minutes, yfinance_symbol)
            if not ydf_fallback.empty and interval_minutes != 60:
                ydf_fallback = _aggregate_ohlcv(ydf_fallback, interval_minutes, yfinance_symbol)
            nse_sources.extend([ydf_primary, ydf_fallback])

        merged, report = stitch_sources(nse_sources, interval_minutes, yfinance_symbol)
        return merged, report
    sources = []
    sources.append(fetch_binance(symbol, timeframe, start))
    if interval_minutes == 60:
        sources.append(fetch_cryptocompare(symbol, start, end))
        sources.append(fetch_yfinance(yfinance_symbol, start, end))
    else:
        sources.append(fetch_cryptocompare_minute(symbol, start, end, interval_minutes))
    merged, report = stitch_sources(sources, interval_minutes, yfinance_symbol)
    return merged, report


def _cryptocompare_key() -> Optional[str]:
    return (
        os.getenv("CRYPTOCOMPARE_API_KEY")
        or os.getenv("CRYPTOCOMPARE_KEY")
        or os.getenv("CRYPTOCOMPARE_TOKEN")
    )


def _split_symbol_for_cryptocompare(symbol: str) -> Tuple[str, str]:
    cleaned = (symbol or "").strip().upper().replace("-", "/")
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        if base and quote:
            return base, quote
    return "BTC", "USD"


def _timeframe_to_minutes(timeframe: str, fallback: int) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    return fallback


def _filter_nse_session(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    sym = (symbol or "").strip().upper()
    if not (sym.endswith(".NS") or sym.endswith(".BO")):
        return data
    if data.empty:
        return data
    holidays = load_nse_holidays(Path(os.getenv("APP_DATA_DIR", "data")))
    idx_local = data.index.tz_convert(IST)
    minutes = (idx_local.hour * 60) + idx_local.minute
    is_holiday = pd.Series(idx_local.date).isin(holidays).to_numpy()
    mask = (idx_local.weekday < 5) & (~is_holiday) & (minutes >= (9 * 60 + 15)) & (minutes <= (15 * 60 + 30))
    return data.loc[mask]


def _aggregate_ohlcv(df: pd.DataFrame, interval_minutes: int, symbol: str = "") -> pd.DataFrame:
    if df.empty or interval_minutes <= 60:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.set_index("timestamp_utc").sort_index()
    idx_local = out.index.tz_convert(IST)
    out.index = idx_local

    if interval_minutes >= 1440:
        # First, build strict 1D trading-day bars, then roll them into N-day bars.
        day_g = out.groupby(out.index.date)
        daily = day_g.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source=("source", "last"),
        )
        if daily.empty:
            return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "volume", "source"])
        daily.index = pd.to_datetime(daily.index).tz_localize(IST) + pd.Timedelta(hours=15, minutes=30)
        n_days = max(1, int(np.ceil(interval_minutes / 1440.0)))
        seq = np.arange(len(daily))
        daily["chunk_id"] = seq // n_days
        chunk = daily.groupby("chunk_id")
        agg = chunk.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source=("source", "last"),
            _slot_count=("close", "size"),
            _end_ts=("close", lambda s: s.index.max()),
        )
        # Keep only full N-day bars; partial tail bars should not be published.
        agg = agg[agg["_slot_count"] == n_days]
        if agg.empty:
            return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "volume", "source"])
        agg.index = pd.DatetimeIndex(agg["_end_ts"])
        agg = agg.drop(columns=["_slot_count", "_end_ts"])
    else:
        rule = f"{int(interval_minutes)}min"
        agg = out.resample(
            rule,
            origin="start_day",
            offset="9h15min",
            closed="right",
            label="right",
        ).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "source": "last",
            }
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"])
        mins = agg.index.hour * 60 + agg.index.minute
        agg = agg[(agg.index.weekday < 5) & (mins >= (9 * 60 + 15)) & (mins <= (15 * 60 + 30))]

    agg.index = agg.index.tz_convert("UTC")
    agg = _enforce_nse_slots(agg, interval_minutes, symbol)
    agg.index.name = "timestamp_utc"
    agg = agg.reset_index()
    return agg
