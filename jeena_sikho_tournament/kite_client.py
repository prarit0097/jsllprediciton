from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def is_kite_enabled() -> bool:
    return _env_flag("KITE_ENABLE", "0")


def _data_dir() -> Path:
    return Path(os.getenv("APP_DATA_DIR", "data"))


def _symbol_key() -> str:
    exchange = (os.getenv("KITE_EXCHANGE", "NSE") or "NSE").strip().upper()
    tradingsymbol = (os.getenv("KITE_TRADINGSYMBOL", "") or "").strip().upper()
    if not tradingsymbol:
        yf = (os.getenv("MARKET_YFINANCE_SYMBOL", "") or "").strip().upper()
        if yf.endswith(".NS") or yf.endswith(".BO"):
            tradingsymbol = yf.split(".", 1)[0]
    if not tradingsymbol:
        market = (os.getenv("MARKET_SYMBOL", "") or "").strip().upper().replace("-", "/")
        tradingsymbol = market.split("/", 1)[0] if "/" in market else market
    return f"{exchange}:{tradingsymbol}"


def _instrument_cache_path() -> Path:
    return _data_dir() / "kite_instruments_cache.json"


def _auth_state_path() -> Path:
    custom = (os.getenv("KITE_AUTH_FILE", "") or "").strip()
    if custom:
        return Path(custom)
    return _data_dir() / "kite_auth.json"


def load_kite_auth_state() -> Dict[str, Any]:
    path = _auth_state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_kite_auth_state(state: Dict[str, Any]) -> None:
    path = _auth_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def current_kite_access_token() -> str:
    state = load_kite_auth_state()
    token = (state.get("access_token", "") or "").strip()
    if token:
        return token
    return (os.getenv("KITE_ACCESS_TOKEN", "") or "").strip()


def save_kite_access_token(access_token: str, session_meta: Optional[Dict[str, Any]] = None) -> None:
    token = (access_token or "").strip()
    if not token:
        raise RuntimeError("empty access token")
    state = load_kite_auth_state()
    state["access_token"] = token
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    if session_meta:
        for key in ("public_token", "user_id", "user_name", "login_time"):
            if key in session_meta and session_meta.get(key) is not None:
                state[key] = session_meta.get(key)
    save_kite_auth_state(state)
    os.environ["KITE_ACCESS_TOKEN"] = token


def kite_login_url() -> str:
    api_key = (os.getenv("KITE_API_KEY", "") or "").strip()
    if len(api_key) < 6:
        raise RuntimeError("KITE_API_KEY missing/invalid")
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def exchange_request_token(request_token: str) -> Dict[str, Any]:
    req = (request_token or "").strip()
    if not req:
        raise RuntimeError("request_token missing")
    try:
        from kiteconnect import KiteConnect  # type: ignore
    except Exception as exc:
        raise RuntimeError("kiteconnect package is not installed") from exc

    api_key = (os.getenv("KITE_API_KEY", "") or "").strip()
    api_secret = (os.getenv("KITE_API_SECRET", "") or "").strip()
    if len(api_key) < 6:
        raise RuntimeError("KITE_API_KEY missing/invalid")
    if not api_secret:
        raise RuntimeError("KITE_API_SECRET missing")

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(req, api_secret=api_secret) or {}
    access_token = (data.get("access_token") or "").strip()
    if not access_token:
        raise RuntimeError("Kite session generated without access token")
    save_kite_access_token(access_token, session_meta=data)
    return data


def _load_instrument_cache() -> Dict[str, Any]:
    path = _instrument_cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_instrument_cache(data: Dict[str, Any]) -> None:
    path = _instrument_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _kite_client():
    try:
        from kiteconnect import KiteConnect  # type: ignore
    except Exception as exc:
        raise RuntimeError("kiteconnect package is not installed") from exc

    api_key = (os.getenv("KITE_API_KEY", "") or "").strip()
    access_token = current_kite_access_token()
    if len(api_key) < 6:
        raise RuntimeError("KITE_API_KEY missing/invalid")
    if not access_token:
        raise RuntimeError("KITE_ACCESS_TOKEN missing")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def resolve_kite_instrument_token(force_refresh: bool = False) -> Optional[int]:
    token_raw = (os.getenv("KITE_INSTRUMENT_TOKEN", "") or "").strip()
    if token_raw.isdigit():
        return int(token_raw)

    key = _symbol_key()
    exchange, tradingsymbol = key.split(":", 1)
    cache = _load_instrument_cache()
    if not force_refresh:
        entry = cache.get(key)
        if isinstance(entry, dict):
            cached_token = entry.get("instrument_token")
            if isinstance(cached_token, int) and cached_token > 0:
                return cached_token

    kite = _kite_client()
    instruments = kite.instruments(exchange)
    selected: Optional[int] = None
    for row in instruments:
        if (row.get("tradingsymbol", "") or "").strip().upper() == tradingsymbol:
            try:
                selected = int(row.get("instrument_token"))
            except Exception:
                selected = None
            if selected:
                break
    if not selected:
        return None

    cache[key] = {
        "instrument_token": selected,
        "exchange": exchange,
        "tradingsymbol": tradingsymbol,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_instrument_cache(cache)
    return selected


def fetch_kite_ltp() -> float:
    kite = _kite_client()
    key = _symbol_key()
    data = kite.ltp([key]) or {}
    row = data.get(key, {})
    price = row.get("last_price")
    if price is None:
        quote = kite.quote([key]) or {}
        q = quote.get(key, {})
        price = q.get("last_price")
    if price is None:
        raise RuntimeError(f"Kite LTP unavailable for {key}")
    return float(price)


def fetch_kite_ohlcv(start: datetime, end: Optional[datetime] = None, interval: str = "60minute") -> pd.DataFrame:
    kite = _kite_client()
    token = resolve_kite_instrument_token()
    if not token:
        raise RuntimeError("Unable to resolve Kite instrument token; set KITE_INSTRUMENT_TOKEN")

    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end and end.tzinfo else (end.replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc))
    if start_utc >= end_utc:
        return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "volume", "source"])

    # Kite intraday history can reject very old from-date ranges.
    # Clamp lookback for minute/hour intervals and let fallback sources provide older backfill.
    interval_l = (interval or "").strip().lower()
    is_intraday = interval_l.endswith("minute") or interval_l.endswith("min")
    if is_intraday:
        max_intraday_days = max(5, int(os.getenv("KITE_MAX_INTRADAY_DAYS", "60")))
        min_start = end_utc - timedelta(days=max_intraday_days)
        if start_utc < min_start:
            start_utc = min_start

    # Keep chunks small to avoid API-side rejections for long minute series.
    chunk_days = max(1, int(os.getenv("KITE_HIST_CHUNK_DAYS", "60")))
    cur = start_utc
    rows = []
    while cur < end_utc:
        chunk_end = min(cur + timedelta(days=chunk_days), end_utc)
        payload = kite.historical_data(token, cur, chunk_end, interval, oi=False)
        if payload:
            rows.extend(payload)
        cur = chunk_end + timedelta(seconds=1)

    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "volume", "source"])
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "volume", "source"])
    df["timestamp_utc"] = pd.to_datetime(df["date"], utc=True)
    out = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    out["source"] = "kite"
    out = out[["timestamp_utc", "open", "high", "low", "close", "volume", "source"]]
    out = out.dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])
    out = out.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")
    return out
