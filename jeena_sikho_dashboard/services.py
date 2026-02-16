import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from jeena_sikho_tournament.config import TournamentConfig, _timeframe_to_minutes
from jeena_sikho_tournament.data_sources import fetch_and_stitch
from jeena_sikho_tournament.features import make_supervised, resolve_feature_windows_for_horizon
from jeena_sikho_tournament.market_calendar import (
    IST,
    align_to_nse_interval_floor,
    load_nse_holidays,
    next_nse_slot_at_or_after,
    nse_market_state,
)
from jeena_sikho_tournament.storage import Storage

from .db import (
    ensure_tables,
    get_champions,
    get_latest_prediction_for_timeframe,
    get_latest_ready_prediction_for_timeframe,
    get_latest_run,
    get_ohlcv_close_at,
    get_recent_ready_predictions,
    get_recent_runs,
    get_scores,
    insert_prediction,
    list_pending_predictions,
    update_prediction,
)

LOGGER = logging.getLogger(__name__)

MARKET_SYMBOL = os.getenv("MARKET_SYMBOL", "BTC/USDT")
MARKET_YFINANCE_SYMBOL = os.getenv("MARKET_YFINANCE_SYMBOL", "BTC-USD")
PRICE_SOURCE = os.getenv("PRICE_SOURCE", "auto").strip().lower()
_default_binance_symbol = MARKET_SYMBOL.replace("/", "").replace("-", "").upper()
_default_coinbase_pair = MARKET_YFINANCE_SYMBOL if "-" in MARKET_YFINANCE_SYMBOL else MARKET_SYMBOL.replace("/", "-").upper()

BINANCE_TICKER_SYMBOL = os.getenv("BINANCE_TICKER_SYMBOL", _default_binance_symbol)
BINANCE_TICKER_URL = f"https://api.binance.com/api/v3/ticker/price?symbol={BINANCE_TICKER_SYMBOL}"
COINBASE_FX_URL = "https://api.coinbase.com/v2/exchange-rates?currency=USD"
FX_CACHE_SECONDS = int(os.getenv("FX_CACHE_SECONDS", "60"))
COINBASE_SPOT_PAIR = os.getenv("COINBASE_SPOT_PAIR", _default_coinbase_pair)
COINBASE_SPOT_URL = f"https://api.coinbase.com/v2/prices/{COINBASE_SPOT_PAIR}/spot"
KRAKEN_PAIR = os.getenv("KRAKEN_PAIR", "XXBTZUSD")
KRAKEN_TICKER_URL = f"https://api.kraken.com/0/public/Ticker?pair={KRAKEN_PAIR}"
LEGACY_PREDICTION_HORIZON_MINUTES = 60
DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h"]
MATCH_EPS = 1e-6
MATCH_MAX_NONZERO = 99.9999

_RUN_LOCK = threading.Lock()
_RUN_STATE = {"running": False, "last_started_at": None}
_RUN_STATE_PATH = Path(os.getenv("APP_DATA_DIR", "data")) / "run_state.json"
_CALIB_STATE_PATH = Path(os.getenv("APP_DATA_DIR", "data")) / "calibration_state.json"
_PRICE_CACHE: Dict[str, Any] = {}
_FX_CACHE: Dict[str, Any] = {}
_TOURNAMENT_INTERVAL_MIN = int(os.getenv("TOURNAMENT_INTERVAL_MINUTES", "120"))
_TOURNAMENT_START_MIN = int(os.getenv("TOURNAMENT_SCHEDULE_MINUTE", "1"))
_NSE_HOLIDAYS = load_nse_holidays(Path(os.getenv("APP_DATA_DIR", "data")))


def _load_calib_state() -> Dict[str, Any]:
    if not _CALIB_STATE_PATH.exists():
        return {}
    try:
        raw = json.loads(_CALIB_STATE_PATH.read_text(encoding="utf8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_calib_state(state: Dict[str, Any]) -> None:
    try:
        _CALIB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CALIB_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf8")
    except Exception:
        pass


def _quote_currency() -> str:
    sym = (MARKET_SYMBOL or "").strip().upper().replace("-", "/")
    if "/" in sym:
        quote = sym.split("/", 1)[1].strip()
        if quote:
            return quote
    yf_sym = (MARKET_YFINANCE_SYMBOL or "").strip().upper()
    if yf_sym.endswith(".NS") or yf_sym.endswith(".BO"):
        return "INR"
    if "-" in yf_sym:
        parts = yf_sym.split("-")
        if len(parts) > 1 and parts[1]:
            return parts[1]
    return "USD"


def _is_indian_equity() -> bool:
    yf_sym = (MARKET_YFINANCE_SYMBOL or "").strip().upper()
    if yf_sym.endswith(".NS") or yf_sym.endswith(".BO"):
        return True
    return _quote_currency() == "INR"


def _market_state(now_utc: datetime) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    if not _is_indian_equity():
        return state
    state.update(nse_market_state(now_utc, _NSE_HOLIDAYS))
    return state


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if value.endswith("Z"):
            try:
                return datetime.fromisoformat(value[:-1])
            except ValueError:
                return None
    return None


def _read_run_state_file() -> Optional[Dict[str, Any]]:
    if not _RUN_STATE_PATH.exists():
        return None
    try:
        with _RUN_STATE_PATH.open("r", encoding="utf8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _next_scheduled_time_local(now_local: datetime) -> datetime:
    interval = max(1, int(_TOURNAMENT_INTERVAL_MIN))
    minute = max(0, min(59, int(_TOURNAMENT_START_MIN)))
    base = now_local.replace(hour=0, minute=minute, second=0, microsecond=0)
    if base > now_local:
        return base
    next_time = base
    while next_time <= now_local:
        next_time = next_time + timedelta(minutes=interval)
    return next_time


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def _fetch_binance_price() -> float:
    resp = requests.get(BINANCE_TICKER_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return float(data.get("price"))


def _fetch_yfinance_price() -> float:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        raise RuntimeError("yfinance not installed") from exc

    ticker = yf.Ticker(MARKET_YFINANCE_SYMBOL)

    # Prefer recent intraday close; fallback to fast_info.
    hist = ticker.history(period="1d", interval="1m")
    if hist is not None and not hist.empty:
        val = hist["Close"].dropna().iloc[-1]
        return float(val)

    fast_info = getattr(ticker, "fast_info", None) or {}
    for key in ("lastPrice", "regularMarketPrice", "previousClose"):
        val = fast_info.get(key)
        if val is not None:
            return float(val)
    raise RuntimeError(f"yfinance price unavailable for {MARKET_YFINANCE_SYMBOL}")


def _fetch_coinbase_price() -> float:
    resp = requests.get(COINBASE_SPOT_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["amount"])


def _fetch_kraken_price() -> float:
    resp = requests.get(KRAKEN_TICKER_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("result", {})
    pair_key = next(iter(result.keys()), None)
    if not pair_key:
        raise RuntimeError("kraken pair data missing")
    price = result[pair_key]["c"][0]
    return float(price)


def _fetch_fx_coinbase() -> float:
    resp = requests.get(COINBASE_FX_URL, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["rates"]["INR"])


def _get_fx_rate() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    cached = _FX_CACHE.copy()
    if cached:
        expires_at = cached.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at > now:
            return {
                "rate": cached.get("rate"),
                "updated_at": cached.get("updated_at"),
                "source": cached.get("source"),
                "stale": False,
            }

    try:
        rate = _fetch_fx_coinbase()
        updated_at = now.isoformat()
        _FX_CACHE.update(
            {
                "rate": rate,
                "updated_at": updated_at,
                "source": "coinbase",
                "expires_at": now + timedelta(seconds=FX_CACHE_SECONDS),
            }
        )
        return {"rate": rate, "updated_at": updated_at, "source": "coinbase", "stale": False}
    except Exception:
        if cached.get("rate") is not None:
            return {
                "rate": cached.get("rate"),
                "updated_at": cached.get("updated_at"),
                "source": cached.get("source"),
                "stale": True,
            }
        raise


def get_live_price() -> Dict[str, Any]:
    if PRICE_SOURCE == "yfinance":
        sources = (("yfinance", _fetch_yfinance_price),)
    else:
        sources = (
            ("yfinance", _fetch_yfinance_price),
            ("binance", _fetch_binance_price),
            ("coinbase", _fetch_coinbase_price),
            ("kraken", _fetch_kraken_price),
        )
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    quote_ccy = _quote_currency()
    use_fx = quote_ccy == "USD"
    market_state = _market_state(now)
    for name, fetch in sources:
        try:
            price = float(fetch())
            _PRICE_CACHE.update({"price": price, "updated_at": now_iso, "source": name, "quote_currency": quote_ccy})
            result: Dict[str, Any] = {"price": price, "updated_at": now_iso, "source": name, "quote_currency": quote_ccy}
            result.update(market_state)
            if name == "yfinance" and market_state.get("market_open") is False:
                result["price_mode"] = "last_traded"
            else:
                result["price_mode"] = "live"
            if use_fx:
                try:
                    fx = _get_fx_rate()
                    if fx.get("rate"):
                        result["price_inr"] = price * float(fx["rate"])
                        result["fx_rate"] = fx["rate"]
                        result["fx_updated_at"] = fx["updated_at"]
                        result["fx_source"] = fx["source"]
                        result["fx_stale"] = fx["stale"]
                except Exception:
                    pass
            return result
        except Exception:
            continue

    cached = _PRICE_CACHE.copy()
    if cached:
        cached["stale"] = True
        cached.update(market_state)
        if "price_mode" not in cached:
            if cached.get("source") == "yfinance" and market_state.get("market_open") is False:
                cached["price_mode"] = "last_traded"
            else:
                cached["price_mode"] = "live"
        if cached.get("quote_currency", quote_ccy) == "USD":
            try:
                fx = _get_fx_rate()
                if fx.get("rate"):
                    cached["price_inr"] = cached["price"] * float(fx["rate"])
                    cached["fx_rate"] = fx["rate"]
                    cached["fx_updated_at"] = fx["updated_at"]
                    cached["fx_source"] = fx["source"]
                    cached["fx_stale"] = fx["stale"]
            except Exception:
                pass
        return cached
    raise RuntimeError("Price source unavailable")


def _parse_iso_utc(value: str) -> datetime:
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        if value.endswith("Z"):
            ts = datetime.fromisoformat(value[:-1])
        else:
            raise
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _parse_user_timestamp(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    if raw.isdigit():
        epoch = int(raw)
        if epoch > 10_000_000_000:
            epoch = int(epoch / 1000)
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d %b %Y, %I:%M:%S %p",
            "%d %b %Y, %I:%M %p",
            "%d %b %Y %I:%M:%S %p",
            "%d %b %Y %I:%M %p",
        )
        for fmt in formats:
            try:
                ts = datetime.strptime(raw, fmt)
                break
            except ValueError:
                ts = None
        if ts is None:
            raise
    if ts.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        ts = ts.replace(tzinfo=local_tz)
    return ts.astimezone(timezone.utc)


def _align_to_interval(ts: datetime, minutes: int) -> datetime:
    if minutes <= 0:
        return ts
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def _prediction_target_timestamp(pred_at: datetime, horizon_min: int, tf_minutes: int) -> datetime:
    step = max(1, int(tf_minutes))
    horizon = max(1, int(horizon_min))
    if not _is_indian_equity():
        anchor = _align_to_interval(pred_at, step)
        return anchor + timedelta(minutes=horizon)

    if horizon >= 1440:
        # NSE daily label is standardized to next trading-day close (15:30 IST).
        steps = int(np.ceil(horizon / 1440.0))
        cur_local = pred_at.astimezone(IST)
        day = cur_local.date()
        advanced = 0
        while advanced < steps:
            day = day + timedelta(days=1)
            probe = datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST)
            if probe.weekday() < 5 and day not in _NSE_HOLIDAYS:
                advanced += 1
        return datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST).astimezone(timezone.utc)

    anchor = align_to_nse_interval_floor(pred_at, step, _NSE_HOLIDAYS)
    full_steps = max(1, int(np.ceil(horizon / step)))
    target = anchor
    for _ in range(full_steps):
        target = next_nse_slot_at_or_after(target + timedelta(minutes=step), step, _NSE_HOLIDAYS)
    return target


def _market_regime(now_utc: Optional[datetime] = None) -> str:
    ts = now_utc or datetime.now(timezone.utc)
    if not _is_indian_equity():
        return "always"
    ts_ist = ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
    mins = ts_ist.hour * 60 + ts_ist.minute
    open_min = 9 * 60 + 15
    close_min = 15 * 60 + 30
    if mins < open_min or mins > close_min:
        return "off_session"
    if mins < (open_min + 60):
        return "opening"
    if mins >= (close_min - 60):
        return "closing"
    return "mid_session"


def _horizon_target_label(candle_minutes: int) -> str:
    minutes = max(1, int(candle_minutes))
    if minutes % 1440 == 0:
        days = max(1, minutes // 1440)
        return f"y_ret_{days}d"
    if minutes % 60 == 0:
        hours = max(1, minutes // 60)
        return f"y_ret_{hours}h"
    return f"y_ret_{minutes}m"


def _return_clip_bounds(horizon_min: int) -> tuple[float, float]:
    # Keep log-return predictions in realistic bounds to avoid unstable bands.
    hm = max(1, int(horizon_min))
    if hm <= 120:
        lim = float(os.getenv("MAX_ABS_RET_SHORT", "0.12"))
    elif hm >= 1440:
        lim = float(os.getenv("MAX_ABS_RET_LONG", "0.35"))
    else:
        lim = float(os.getenv("MAX_ABS_RET_MID", "0.2"))
    lim = max(0.02, min(1.5, lim))
    return (-lim, lim)


def _clip_return(predicted_return: float, horizon_min: int) -> float:
    lo, hi = _return_clip_bounds(horizon_min)
    return float(max(lo, min(hi, float(predicted_return))))


def _price_from_log_return(base_price: float, log_ret: float) -> float:
    lr = float(max(-20.0, min(20.0, log_ret)))
    return float(base_price) * float(np.exp(lr))


def _resolve_feature_cols(model, fallback_cols: List[str]) -> List[str]:
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_cols)


def _load_latest_dataset(config: TournamentConfig) -> pd.DataFrame:
    storage = Storage(config.db_path, config.ohlcv_table)
    storage.init_db()
    df = storage.load()
    return df


def _ensure_recent_data(config: TournamentConfig, days: int = 14) -> pd.DataFrame:
    if _is_indian_equity():
        # Session-filtered NSE data has fewer bars/day; keep larger warmup.
        days = max(days, 120)
    df = _load_latest_dataset(config)
    now_utc = datetime.now(timezone.utc)
    need_fetch = df.empty
    if not need_fetch:
        latest = df.index.max()
        if latest is None:
            need_fetch = True
        elif _is_indian_equity():
            if latest < (now_utc - timedelta(days=2)):
                need_fetch = True
            if len(df) < 200:
                need_fetch = True
            if "source" in df.columns:
                uniq = set(str(v).lower() for v in df["source"].dropna().unique())
                if uniq and uniq.issubset({"doctor"}):
                    need_fetch = True
        else:
            if latest < (now_utc - timedelta(days=2)):
                need_fetch = True
    if not need_fetch:
        return df
    try:
        end = now_utc
        start = end - timedelta(days=days)
        fetched, _ = fetch_and_stitch(
            config.symbol,
            config.yfinance_symbol,
            start,
            config.timeframe,
            config.candle_minutes,
        )
        if not fetched.empty:
            fetched = fetched.set_index("timestamp_utc")
            Storage(config.db_path, config.ohlcv_table).upsert(fetched)
        return _load_latest_dataset(config)
    except Exception:
        return df


def _parse_timeframes(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_TIMEFRAMES)
    tokens: List[str] = []
    for part in value.replace("|", ",").replace(";", ",").split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    if not tokens:
        return list(DEFAULT_TIMEFRAMES)
    seen = set()
    ordered: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def get_timeframes(config: TournamentConfig) -> List[str]:
    env_list = os.getenv("MARKET_TIMEFRAMES") or os.getenv("TIMEFRAMES")
    if env_list:
        return _parse_timeframes(env_list)
    return list(DEFAULT_TIMEFRAMES)


def get_primary_timeframe(config: TournamentConfig) -> str:
    frames = get_timeframes(config)
    return frames[0] if frames else config.timeframe


def _config_for_timeframe(base: TournamentConfig, timeframe: str) -> TournamentConfig:
    cfg = TournamentConfig()
    cfg.__dict__.update(base.__dict__)
    cfg.timeframe = timeframe
    cfg.candle_minutes = _timeframe_to_minutes(timeframe, base.candle_minutes)
    cfg.feature_windows = resolve_feature_windows_for_horizon(cfg.candle_minutes, base.feature_windows)
    if cfg.candle_minutes == 60:
        cfg.ohlcv_table = "ohlcv"
    else:
        cfg.ohlcv_table = f"ohlcv_{cfg.candle_minutes}m"
    cfg.registry_path = base.data_dir / f"registry_{cfg.candle_minutes}m.json"
    cfg.log_path = base.data_dir / f"tournament_{cfg.candle_minutes}m.log"
    return cfg


def _expected_points_in_window(start_utc: datetime, end_utc: datetime, tf_minutes: int, nse_mode: bool) -> int:
    if end_utc <= start_utc:
        return 0
    step = max(1, int(tf_minutes))
    if not nse_mode:
        rng = pd.date_range(start=start_utc, end=end_utc, freq=f"{step}min", tz="UTC")
        return int(len(rng))
    cur = start_utc.astimezone(IST)
    end_local = end_utc.astimezone(IST)
    count = 0
    while cur.date() <= end_local.date():
        day = cur.date()
        day_start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=cur.tzinfo)
        day_end = datetime(day.year, day.month, day.day, 15, 30, tzinfo=cur.tzinfo)
        if day_start.weekday() < 5 and day not in _NSE_HOLIDAYS:
            minute = 9 * 60 + 15
            while minute <= (15 * 60 + 30):
                hh, mm = divmod(minute, 60)
                slot = datetime(day.year, day.month, day.day, hh, mm, tzinfo=cur.tzinfo)
                slot_utc = slot.astimezone(timezone.utc)
                if start_utc <= slot_utc <= end_utc:
                    count += 1
                minute += step
        cur = cur + timedelta(days=1)
    return int(count)


def _completeness_by_horizon(config: TournamentConfig) -> List[Dict[str, Any]]:
    lookback_days = max(5, int(os.getenv("COMPLETENESS_LOOKBACK_DAYS", "30")))
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    nse_mode = _is_indian_equity()
    rows: List[Dict[str, Any]] = []
    for tf in get_timeframes(config):
        cfg = _config_for_timeframe(config, tf)
        storage = Storage(cfg.db_path, cfg.ohlcv_table)
        df = storage.load()
        if df.empty:
            rows.append({"timeframe": tf, "lookback_days": lookback_days, "expected": 0, "actual": 0, "completeness_pct": 0.0})
            continue
        recent = df.loc[df.index >= start]
        expected = _expected_points_in_window(start, now, cfg.candle_minutes, nse_mode)
        actual = int(len(recent))
        pct = (actual / expected * 100.0) if expected > 0 else 0.0
        rows.append(
            {
                "timeframe": tf,
                "lookback_days": lookback_days,
                "expected": expected,
                "actual": actual,
                "completeness_pct": round(float(pct), 2),
            }
        )
    return rows


def _champion_confidence(final_score: float, stability: float = 0.0, trend_delta: float = 0.0) -> float:
    score = float(final_score)
    stability_penalty = max(0.0, float(stability))
    trend = float(trend_delta)
    raw = (score * 100.0) - (stability_penalty * 25.0) + (trend * 20.0)
    return max(1.0, min(99.0, raw))


def _champion_detail_from_registry(reg: Dict[str, Any], task: str) -> Dict[str, Any]:
    champ = (reg.get("champions") or {}).get(task) or {}
    hist = (reg.get("history") or {}).get(task) or []
    latest_score = float(champ.get("final_score") or 0.0)
    prev_score = None
    if len(hist) >= 2:
        try:
            prev_score = float(hist[-2].get("best", {}).get("final_score"))
        except Exception:
            prev_score = None
    trend_delta = (latest_score - prev_score) if prev_score is not None else 0.0
    stability = 0.0
    fam = champ.get("family")
    if fam:
        model_hist = (reg.get("model_history") or {}).get(fam) or []
        if len(model_hist) >= 3:
            vals = [float(x.get("score", 0.0)) for x in model_hist[-10:]]
            if vals:
                stability = float(np.std(vals))
    confidence = _champion_confidence(latest_score, stability, trend_delta)
    fi_rows = (((reg.get("feature_importance") or {}).get(task)) or [])
    latest_fi = fi_rows[-1] if fi_rows else {}
    top_feats = [str(x.get("feature")) for x in (latest_fi.get("top_features") or [])[:5]]
    return {
        "model_id": champ.get("model_id"),
        "feature_set_id": champ.get("feature_set_id"),
        "final_score": latest_score if champ else None,
        "trend_delta": trend_delta,
        "stability": stability,
        "confidence_pct": confidence if champ else None,
        "top_features": top_feats,
    }


def get_price_at_timestamp(config: TournamentConfig, value: str) -> Dict[str, Any]:
    ts_utc = _parse_user_timestamp(value)
    primary_tf = get_primary_timeframe(config)
    tf_cfg = _config_for_timeframe(config, primary_tf)
    tf_minutes = max(1, int(tf_cfg.candle_minutes))
    if _is_indian_equity():
        anchor = next_nse_slot_at_or_after(ts_utc, tf_minutes, _NSE_HOLIDAYS)
    else:
        anchor = _align_to_interval(ts_utc, tf_minutes)
    target_iso = anchor.isoformat()
    price = get_ohlcv_close_at(target_iso, table=tf_cfg.ohlcv_table)
    if price is None:
        raise LookupError("price not found for timestamp")
    result: Dict[str, Any] = {
        "requested_at": value,
        "timestamp_utc": ts_utc.isoformat(),
        "aligned_at": target_iso,
        "price": float(price),
        "quote_currency": _quote_currency(),
        "timeframe": tf_cfg.timeframe,
        "table": tf_cfg.ohlcv_table,
        "aligned": True,
    }
    if result["quote_currency"] == "USD":
        try:
            fx = _get_fx_rate()
            if fx.get("rate"):
                result["price_inr"] = float(price) * float(fx["rate"])
                result["fx_rate"] = fx["rate"]
                result["fx_updated_at"] = fx["updated_at"]
                result["fx_source"] = fx["source"]
                result["fx_stale"] = fx["stale"]
        except Exception:
            pass
    return result


def get_tournament_summary(config: TournamentConfig) -> Dict[str, Any]:
    primary_tf = get_primary_timeframe(config)
    tf_cfg = _config_for_timeframe(config, primary_tf)
    latest_run = get_latest_run()
    run_at = latest_run["run_at"] if latest_run else None
    run_started_at = latest_run.get("run_started_at") if latest_run else None
    run_finished_at = latest_run.get("run_finished_at") if latest_run else None
    if not run_started_at:
        run_started_at = run_at
    if not run_finished_at:
        run_finished_at = run_at
    file_state = _read_run_state_file()
    if file_state:
        running = bool(file_state.get("running"))
        file_started = file_state.get("last_started_at")
        file_finished = file_state.get("last_finished_at")
        if running:
            if file_started:
                file_start_dt = _parse_iso(file_started)
                run_start_dt = _parse_iso(run_started_at)
                if file_start_dt and (not run_start_dt or file_start_dt > run_start_dt):
                    run_started_at = file_started
        else:
            if file_finished:
                file_finish_dt = _parse_iso(file_finished)
                run_finish_dt = _parse_iso(run_finished_at)
                if file_finish_dt and (not run_finish_dt or file_finish_dt > run_finish_dt):
                    run_finished_at = file_finished
                    if file_started:
                        file_start_dt = _parse_iso(file_started)
                        if file_start_dt and file_finish_dt and file_start_dt <= file_finish_dt:
                            run_started_at = file_started
        if not run_at:
            run_at = run_finished_at or run_started_at
    candidate_count = latest_run["candidate_count"] if latest_run else 0
    eta_seconds = _estimate_eta_seconds(tf_cfg, candidate_count)
    now_local = datetime.now().astimezone()
    next_run_local = _next_scheduled_time_local(now_local)
    champions: Dict[str, Any] = {}
    if latest_run:
        champions = get_champions(latest_run["id"])
    if not champions:
        reg = _load_registry(tf_cfg.registry_path)
        champions = reg.get("champions", {}) if isinstance(reg, dict) else {}
    champions_by_horizon: Dict[str, Any] = {}
    for tf in get_timeframes(config):
        cfg_tf = _config_for_timeframe(config, tf)
        reg_tf = _load_registry(cfg_tf.registry_path)
        if not isinstance(reg_tf, dict):
            reg_tf = {}
        champs_tf = {
            "direction": _champion_detail_from_registry(reg_tf, "direction"),
            "return": _champion_detail_from_registry(reg_tf, "return"),
            "range": _champion_detail_from_registry(reg_tf, "range"),
        }
        champions_by_horizon[tf] = champs_tf

    drift_status = _compute_drift_status(config)
    backtest_report = _build_backtest_report(config)
    auto_retrain = bool(drift_status.get("alert"))
    completeness_rows = _completeness_by_horizon(config)
    return {
        "last_run_at": run_at,
        "last_run_started_at": run_started_at,
        "last_run_finished_at": run_finished_at,
        "run_mode": latest_run["run_mode"] if latest_run else None,
        "candidate_count": candidate_count,
        "champions": champions,
        "champions_by_horizon": champions_by_horizon,
        "eta_seconds": eta_seconds,
        "next_run_at": next_run_local.isoformat(),
        "drift_status": drift_status,
        "backtest_report": backtest_report,
        "completeness_by_horizon": completeness_rows,
        "auto_retrain_recommended": auto_retrain,
    }


def get_scoreboard(limit: int = 500) -> List[Dict[str, Any]]:
    latest = get_latest_run()
    if not latest:
        return []
    return get_scores(latest["id"], limit)


def update_pending_predictions(config: TournamentConfig) -> None:
    ensure_tables()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
    cutoff_iso = cutoff.isoformat()
    pending = list_pending_predictions(cutoff_iso)
    if not pending:
        return

    for p in pending:
        pred_at = _parse_iso_utc(p["predicted_at"])
        horizon_min = p.get("prediction_horizon_min") or p.get("timeframe_minutes") or LEGACY_PREDICTION_HORIZON_MINUTES
        tf_minutes = p.get("timeframe_minutes") or horizon_min
        target_ts = _prediction_target_timestamp(pred_at, int(horizon_min), int(tf_minutes))
        if datetime.now(timezone.utc) < target_ts:
            continue
        target_iso = target_ts.isoformat()

        table = "ohlcv" if int(tf_minutes) == 60 else f"ohlcv_{int(tf_minutes)}m"
        actual = get_ohlcv_close_at(target_iso, table=table)
        if actual is None:
            try:
                actual = get_live_price()["price"]
            except Exception:
                continue
        metrics = _compute_match_metrics(p.get("predicted_price"), actual)
        update_prediction(p["id"], actual, metrics["match_percent"], "ready")


def _target_mode() -> str:
    return os.getenv("RETURN_TARGET_MODE", "volnorm_logret").strip().lower()


def _target_scale_from_latest(latest_row: pd.DataFrame) -> float:
    if latest_row is None or latest_row.empty:
        return 1.0
    floor = float(os.getenv("TARGET_VOL_FLOOR", "0.001"))
    cap = float(os.getenv("TARGET_VOL_CAP", "0.08"))
    try:
        raw = float(latest_row.iloc[-1].get("vol_24", 1.0))
    except Exception:
        raw = 1.0
    if not np.isfinite(raw):
        raw = 1.0
    return float(max(max(1e-6, floor), min(max(floor, cap), raw)))


def _denorm_return(pred_model: float, latest_row: pd.DataFrame, mode: Optional[str]) -> float:
    m = (mode or _target_mode()).strip().lower()
    if m in {"volnorm", "volnorm_logret", "normalized"}:
        return float(pred_model) * _target_scale_from_latest(latest_row)
    return float(pred_model)


def _predict_return_from_champion(config: TournamentConfig, latest_row: pd.DataFrame) -> Optional[Dict[str, Any]]:
    reg = _load_registry(config.registry_path)
    champ = reg.get("champions", {}).get("return")
    if not champ:
        return None

    model_path = champ.get("model_path")
    if not model_path:
        return None

    import joblib

    model = joblib.load(model_path)
    feature_cols = _resolve_feature_cols(model, champ.get("feature_cols", []))
    X = latest_row.reindex(columns=feature_cols, fill_value=0.0)
    pred_model = float(model.predict(X)[0])
    pred = _denorm_return(pred_model, latest_row, champ.get("target_mode"))
    score = float(champ.get("final_score") or 0.0)
    confidence_pct = _champion_confidence(score)
    return {
        "predicted_return": pred,
        "model_name": champ.get("model_id", "return_champion"),
        "feature_set": champ.get("feature_set_id"),
        "confidence_pct": confidence_pct,
        "uncertainty_return_std": None,
    }


def _predict_return_from_ensemble(config: TournamentConfig, latest_row: pd.DataFrame, regime: Optional[str] = None) -> Optional[Dict[str, Any]]:
    reg = _load_registry(config.registry_path)
    ensemble = reg.get("ensembles", {}).get("return")
    if not ensemble:
        return None
    members = ensemble.get("members") or []
    if not members:
        return None

    import joblib

    candidates: List[Dict[str, Any]] = []
    pred_map: Dict[str, float] = {}
    regime_name = regime or _market_regime()
    regime_weights = (ensemble.get("regime_weights") or {}).get(regime_name, {})
    regime_top_members = (ensemble.get("regime_top_members") or {}).get(regime_name) or []
    error_weights = ensemble.get("error_weights") or {}
    regime_pref = {
        "opening": {"micro_momentum", "momentum", "session", "vwap_flow"},
        "mid_session": {"trend", "trend_longer", "signal", "base"},
        "closing": {"trend_longer", "volatility", "session", "trend"},
        "off_session": {"trend_longer", "long", "base"},
        "always": {"base", "trend", "signal"},
    }
    pref = regime_pref.get(regime_name, set())
    for member in members:
        model_path = member.get("model_path")
        if not model_path:
            continue
        if not Path(model_path).exists():
            continue
        model = joblib.load(model_path)
        feature_cols = _resolve_feature_cols(model, member.get("feature_cols", []))
        X = latest_row.reindex(columns=feature_cols, fill_value=0.0)
        pred_model = float(model.predict(X)[0])
        pred_val = _denorm_return(pred_model, latest_row, member.get("target_mode"))
        if not np.isfinite(pred_val):
            continue
        weight = member.get("final_score")
        try:
            weight_val = float(weight) if weight is not None else 1.0
        except (TypeError, ValueError):
            weight_val = 1.0
        fs_id = str(member.get("feature_set_id") or "")
        model_id = member.get("model_id", "unknown")
        err_boost = float(error_weights.get(model_id, 0.0))
        regime_gate = float(regime_weights.get(model_id, 0.0))
        regime_bonus = 0.2 if fs_id in pref else 0.0
        routed_weight = max(0.0, (0.6 * weight_val) + (0.25 * err_boost) + (0.15 * regime_gate) + regime_bonus)
        pred_map[model_id] = pred_val
        candidates.append(
            {
                "model_id": model_id,
                "pred": pred_val,
                "feature_set_id": fs_id,
                "weight": routed_weight,
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["weight"], reverse=True)
    top_n = max(1, int(os.getenv("ROUTING_TOP_MEMBERS", "3")))
    selected = candidates[:top_n]
    if regime_top_members:
        preferred = [c for c in candidates if str(c.get("model_id")) in set(regime_top_members)]
        if preferred:
            selected = preferred[: min(top_n, len(preferred))]
    preds = [float(c["pred"]) for c in selected]
    weights = [float(c["weight"]) for c in selected]
    used_members = [str(c["model_id"]) for c in selected]

    predicted_return: Optional[float] = None
    used_stacking = False
    stacking = ensemble.get("stacking") or {}
    stacking_path = stacking.get("model_path")
    stacking_ids = stacking.get("member_ids") or []
    if stacking_path and stacking_ids and Path(stacking_path).exists():
        try:
            if all(mid in pred_map for mid in stacking_ids):
                meta = joblib.load(stacking_path)
                X_meta = np.array([pred_map[mid] for mid in stacking_ids], dtype=float).reshape(1, -1)
                meta_pred = float(meta.predict(X_meta)[0])
                if np.isfinite(meta_pred):
                    predicted_return = meta_pred
                    used_stacking = True
        except Exception:
            predicted_return = None

    if predicted_return is None:
        if sum(weights) > 0:
            predicted_return = float(np.average(preds, weights=weights))
        else:
            predicted_return = float(np.mean(preds))

    pred_std = float(np.std(preds)) if len(preds) >= 2 else 0.0
    base_mag = float(np.mean(np.abs(preds))) + 1e-6
    rel_disp = pred_std / base_mag
    conf_disp = 1.0 / (1.0 + (4.0 * rel_disp))
    conf_size = min(1.0, len(preds) / 3.0)
    confidence_pct = float(max(1.0, min(99.0, 100.0 * conf_disp * conf_size)))

    return {
        "predicted_return": predicted_return,
        "model_name": "stacked_ridge" if used_stacking else f"gated_ensemble_top{len(preds)}",
        "feature_set": "ensemble",
        "ensemble_members": used_members,
        "ensemble_size": len(preds),
        "regime": regime_name,
        "uncertainty_return_std": pred_std,
        "confidence_pct": confidence_pct,
    }


def _predict_return_from_direction(config: TournamentConfig, latest_row: pd.DataFrame, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    reg = _load_registry(config.registry_path)
    champ = reg.get("champions", {}).get("direction")
    if not champ:
        return None

    model_path = champ.get("model_path")
    if not model_path:
        return None

    import joblib

    model = joblib.load(model_path)
    feature_cols = _resolve_feature_cols(model, champ.get("feature_cols", []))
    X = latest_row.reindex(columns=feature_cols, fill_value=0.0)
    direction = int(model.predict(X)[0])

    recent = df["close"].pct_change().dropna().tail(24)
    avg_move = float(np.abs(recent).mean()) if not recent.empty else 0.002
    sign = 1.0 if direction == 1 else -1.0
    predicted_return = np.log(1 + avg_move * sign)
    score = float(champ.get("final_score") or 0.0)
    confidence_pct = _champion_confidence(score)

    return {
        "predicted_return": float(predicted_return),
        "model_name": champ.get("model_id", "direction_champion"),
        "feature_set": champ.get("feature_set_id"),
        "confidence_pct": confidence_pct,
        "uncertainty_return_std": float(abs(predicted_return) * 0.6),
    }


def _cooldown_minutes(horizon_min: int) -> int:
    return max(1, min(15, int(horizon_min / 2)))


def _compute_match_metrics(predicted: Optional[float], actual: Optional[float]) -> Dict[str, Optional[float]]:
    if predicted is None or actual is None:
        return {"abs_diff": None, "pct_error": None, "match_percent": None}
    try:
        predicted_val = float(predicted)
        actual_val = float(actual)
    except (TypeError, ValueError):
        return {"abs_diff": None, "pct_error": None, "match_percent": None}
    abs_diff = abs(predicted_val - actual_val)
    if actual_val == 0:
        return {"abs_diff": abs_diff, "pct_error": None, "match_percent": None}
    pct_error = (abs_diff / abs(actual_val)) * 100.0
    match = 100.0 - pct_error
    match = max(0.0, min(100.0, match))
    if abs_diff > MATCH_EPS:
        match = min(match, MATCH_MAX_NONZERO)
    return {"abs_diff": abs_diff, "pct_error": pct_error, "match_percent": match}


def _get_recent_bias(config: TournamentConfig, timeframe: str) -> float:
    limit = max(1, int(config.bias_window))
    rows = get_recent_ready_predictions(timeframe, limit)
    if not rows:
        return 0.0
    errors: List[float] = []
    for row in rows:
        predicted_ret = row.get("predicted_return")
        actual_price = row.get("actual_price_1h")
        current_price = row.get("current_price")
        if predicted_ret is None or actual_price is None or current_price is None:
            continue
        try:
            actual_val = float(actual_price)
            current_val = float(current_price)
            predicted_val = float(predicted_ret)
        except (TypeError, ValueError):
            continue
        if actual_val <= 0 or current_val <= 0:
            continue
        actual_ret = float(np.log(actual_val / current_val))
        err = actual_ret - predicted_val
        if np.isfinite(err):
            errors.append(err)
    if not errors:
        return 0.0
    strategy = os.getenv("BIAS_STRATEGY", "trimmed").strip().lower()
    if strategy == "median":
        bias = float(np.median(errors))
    elif strategy == "mean":
        bias = float(np.mean(errors))
    else:
        # Trim 10% extremes for stability, fallback to median for small samples.
        errors.sort()
        n = len(errors)
        trim = int(n * 0.1)
        if n >= 10 and trim > 0 and (n - 2 * trim) > 0:
            trimmed = errors[trim:-trim]
            bias = float(np.mean(trimmed)) if trimmed else float(np.median(errors))
        else:
            bias = float(np.median(errors))
    max_abs = float(config.bias_max_abs)
    if max_abs > 0:
        bias = max(-max_abs, min(max_abs, bias))
    return bias


def _fit_return_calibrator(config: TournamentConfig, timeframe: str) -> Dict[str, Any]:
    min_samples = max(5, int(os.getenv("CALIBRATION_MIN_SAMPLES", "20")))
    limit = max(min_samples, int(os.getenv("CALIBRATION_LOOKBACK", "200")))
    rows = get_recent_ready_predictions(timeframe, limit)
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        pred_ret = row.get("predicted_return")
        cur = row.get("current_price")
        act = row.get("actual_price_1h")
        if pred_ret is None or cur is None or act is None:
            continue
        try:
            x = float(pred_ret)
            cur_v = float(cur)
            act_v = float(act)
        except (TypeError, ValueError):
            continue
        if cur_v <= 0 or act_v <= 0:
            continue
        y = float(np.log(act_v / cur_v))
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < min_samples:
        return {"method": "none", "samples": len(xs), "active": False}

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    n = len(x_arr)
    holdout = max(5, int(round(n * 0.2)))
    holdout = min(max(5, holdout), n - 2)
    tr_x, va_x = x_arr[:-holdout], x_arr[-holdout:]
    tr_y, va_y = y_arr[:-holdout], y_arr[-holdout:]

    candidates: List[Dict[str, Any]] = []

    # Linear calibrator.
    try:
        X = np.column_stack([np.ones(len(tr_x)), tr_x])
        coeff, *_ = np.linalg.lstsq(X, tr_y, rcond=None)
        alpha = float(max(-0.05, min(0.05, coeff[0])))
        beta = float(max(-2.0, min(2.0, coeff[1])))
        lin_pred = alpha + (beta * va_x)
        lin_rmse = float(np.sqrt(np.mean((va_y - lin_pred) ** 2)))
        candidates.append(
            {
                "method": "linear",
                "alpha": alpha,
                "beta": beta,
                "samples": n,
                "active": True,
                "rmse": lin_rmse,
            }
        )
    except Exception:
        pass

    # Isotonic calibrator.
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(tr_x, tr_y)
        va_pred = iso.predict(va_x)
        iso_rmse = float(np.sqrt(np.mean((va_y - va_pred) ** 2)))
        candidates.append(
            {
                "method": "isotonic",
                "samples": n,
                "active": True,
                "x": tr_x.tolist(),
                "y": iso.predict(tr_x).tolist(),
                "rmse": iso_rmse,
            }
        )
    except Exception:
        pass

    # Quantile-bin calibrator (robust monotonic approximation).
    try:
        q = np.linspace(0.0, 1.0, 8)
        bins = np.quantile(tr_x, q)
        bins = np.unique(bins)
        if bins.size >= 3:
            mids: List[float] = []
            vals: List[float] = []
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                mask = (tr_x >= lo) & (tr_x <= hi if i == len(bins) - 2 else tr_x < hi)
                if not mask.any():
                    continue
                mids.append(float(np.median(tr_x[mask])))
                vals.append(float(np.median(tr_y[mask])))
            if len(mids) >= 3:
                va_pred = np.interp(va_x, mids, vals, left=vals[0], right=vals[-1])
                q_rmse = float(np.sqrt(np.mean((va_y - va_pred) ** 2)))
                candidates.append(
                    {
                        "method": "quantile",
                        "samples": n,
                        "active": True,
                        "x": mids,
                        "y": vals,
                        "rmse": q_rmse,
                    }
                )
    except Exception:
        pass

    if not candidates:
        return {"method": "none", "samples": n, "active": False}

    # Persist per-horizon calibration quality and use sticky selection to avoid noisy method-flips.
    state = _load_calib_state()
    tf_state = state.get(timeframe, {}) if isinstance(state, dict) else {}
    method_scores = tf_state.get("method_scores", {}) if isinstance(tf_state, dict) else {}
    alpha = min(1.0, max(0.05, float(os.getenv("CALIB_SCORE_EMA_ALPHA", "0.35"))))
    switch_gain = min(0.5, max(0.0, float(os.getenv("CALIB_MIN_SWITCH_GAIN", "0.05"))))
    sticky_margin = min(0.5, max(0.0, float(os.getenv("CALIB_STICKINESS_MARGIN", "0.08"))))

    current_rmse = {str(c.get("method")): float(c.get("rmse", 1e9)) for c in candidates}
    rolling_rmse: Dict[str, float] = {}
    for method, rmse_val in current_rmse.items():
        prev = method_scores.get(method)
        try:
            prev_val = float(prev)
        except (TypeError, ValueError):
            prev_val = rmse_val
        if np.isfinite(prev_val):
            rolling_rmse[method] = float((alpha * rmse_val) + ((1.0 - alpha) * prev_val))
        else:
            rolling_rmse[method] = float(rmse_val)

    best_method = min(rolling_rmse.items(), key=lambda kv: kv[1])[0]
    best = next((c for c in candidates if str(c.get("method")) == best_method), candidates[0])

    prev_method = str(tf_state.get("selected_method", "")).strip().lower()
    selected_reason = "best_rolling_rmse"
    if prev_method and prev_method in current_rmse:
        prev_roll = float(rolling_rmse.get(prev_method, 1e9))
        best_roll = float(rolling_rmse.get(best_method, 1e9))
        prev_cur = float(current_rmse.get(prev_method, 1e9))
        best_cur = float(current_rmse.get(best_method, 1e9))
        rel_gain = (prev_roll - best_roll) / max(1e-9, abs(prev_roll))
        if (prev_cur <= best_cur * (1.0 + sticky_margin)) or (rel_gain < switch_gain):
            best = next((c for c in candidates if str(c.get("method")) == prev_method), best)
            best_method = prev_method
            selected_reason = "sticky_previous_method"
        else:
            selected_reason = "switch_on_material_gain"

    # Save rolling calibration scores by horizon.
    try:
        state = state if isinstance(state, dict) else {}
        state[timeframe] = {
            "selected_method": best_method,
            "method_scores": rolling_rmse,
            "samples": int(n),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_calib_state(state)
    except Exception:
        pass

    best["selected_from"] = [c.get("method") for c in candidates]
    best["selection_reason"] = selected_reason
    best["rolling_rmse"] = rolling_rmse
    return best


def _apply_return_calibrator(predicted_return: float, calibrator: Dict[str, Any]) -> float:
    if not calibrator or not calibrator.get("active"):
        return float(predicted_return)
    method = str(calibrator.get("method", "linear")).strip().lower()
    x = float(predicted_return)
    if method == "linear":
        alpha = float(calibrator.get("alpha", 0.0))
        beta = float(calibrator.get("beta", 1.0))
        return float(alpha + (beta * x))
    if method in {"isotonic", "quantile"}:
        xs = np.asarray(calibrator.get("x") or [], dtype=float)
        ys = np.asarray(calibrator.get("y") or [], dtype=float)
        if xs.size >= 2 and ys.size >= 2 and xs.size == ys.size:
            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]
            return float(np.interp(x, xs, ys, left=ys[0], right=ys[-1]))
    return x


def _summarize_ready_metrics(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    abs_errors: List[float] = []
    pct_errors: List[float] = []
    hit_total = 0
    hit_true = 0
    util_vals: List[float] = []
    calib_err_sq: List[float] = []
    for row in rows:
        predicted = row.get("predicted_price")
        actual = row.get("actual_price_1h")
        if predicted is None or actual is None:
            continue
        metrics = _compute_match_metrics(predicted, actual)
        if metrics["abs_diff"] is not None:
            abs_errors.append(float(metrics["abs_diff"]))
        if metrics["pct_error"] is not None:
            pct_errors.append(float(metrics["pct_error"]))

        predicted_ret = row.get("predicted_return")
        current = row.get("current_price")
        if predicted_ret is None or current is None or actual is None:
            continue
        try:
            predicted_ret_val = float(predicted_ret)
            current_val = float(current)
            actual_val = float(actual)
        except (TypeError, ValueError):
            continue
        if current_val <= 0 or actual_val <= 0:
            continue
        actual_ret = float(np.log(actual_val / current_val))
        hit_total += 1
        if (predicted_ret_val >= 0 and actual_ret >= 0) or (predicted_ret_val < 0 and actual_ret < 0):
            hit_true += 1
        util_vals.append(float(np.sign(predicted_ret_val) * actual_ret))
        calib_err_sq.append(float((actual_ret - predicted_ret_val) ** 2))

    sample_size = len(abs_errors)
    if sample_size == 0:
        return None
    mae_val = float(np.mean(abs_errors))
    mape_val = float(np.mean(pct_errors)) if pct_errors else None
    hit_rate = (100.0 * hit_true / hit_total) if hit_total > 0 else None
    out: Dict[str, Any] = {
        "samples": sample_size,
        "mae": mae_val,
        "mape": mape_val,
        "hit_rate": hit_rate,
        "directional_utility": float(np.mean(util_vals)) if util_vals else None,
        "calibration_rmse": float(np.sqrt(np.mean(calib_err_sq))) if calib_err_sq else None,
    }
    if hit_total > 0:
        out["hit_count"] = hit_true
        out["hit_total"] = hit_total
    return out


def _collect_metrics_by_horizon(config: TournamentConfig) -> List[Dict[str, Any]]:
    metrics_limit = max(10, int(os.getenv("PREDICTION_METRICS_LIMIT", "200")))
    rows: List[Dict[str, Any]] = []
    for timeframe in get_timeframes(config):
        horizon_min = _timeframe_to_minutes(timeframe, config.candle_minutes)
        ready_rows = get_recent_ready_predictions(timeframe, metrics_limit)
        summary = _summarize_ready_metrics(ready_rows)
        rows.append(
            {
                "timeframe": timeframe,
                "target": _horizon_target_label(horizon_min),
                "horizon_minutes": horizon_min,
                "metrics": summary,
            }
        )
    return rows


def _build_backtest_report(config: TournamentConfig) -> List[Dict[str, Any]]:
    rows = _collect_metrics_by_horizon(config)
    out: List[Dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        mape = metrics.get("mape")
        hit_rate = metrics.get("hit_rate")
        util = metrics.get("directional_utility")
        calibration_rmse = metrics.get("calibration_rmse")
        production_ready = bool(
            metrics
            and mape is not None
            and hit_rate is not None
            and float(mape) <= float(os.getenv("PROD_MAX_MAPE", "3.0"))
            and float(hit_rate) >= float(os.getenv("PROD_MIN_HIT_RATE", "52.0"))
        )
        out.append(
            {
                "timeframe": row.get("timeframe"),
                "target": row.get("target"),
                "samples": metrics.get("samples") if metrics else 0,
                "mae": metrics.get("mae") if metrics else None,
                "mape": mape,
                "hit_rate": hit_rate,
                "directional_utility": util,
                "calibration_rmse": calibration_rmse,
                "production_ready": production_ready,
            }
        )
    return out


def _production_ready_map(config: TournamentConfig) -> Dict[str, bool]:
    report = _build_backtest_report(config)
    out: Dict[str, bool] = {}
    for row in report:
        tf = str(row.get("timeframe") or "")
        if tf:
            out[tf] = bool(row.get("production_ready"))
    return out


def _compute_drift_status(config: TournamentConfig) -> Dict[str, Any]:
    window = max(20, int(os.getenv("DRIFT_WINDOW", "60")))
    ratio = float(os.getenv("DRIFT_MAPE_RATIO", "1.2"))
    hit_drop = float(os.getenv("DRIFT_HIT_DROP", "5.0"))
    details: List[Dict[str, Any]] = []
    alert = False
    for timeframe in get_timeframes(config):
        rows = get_recent_ready_predictions(timeframe, window)
        if len(rows) < 12:
            details.append({"timeframe": timeframe, "status": "insufficient_samples", "samples": len(rows)})
            continue
        half = len(rows) // 2
        recent = rows[:half]
        prev = rows[half:]
        m_recent = _summarize_ready_metrics(recent) or {}
        m_prev = _summarize_ready_metrics(prev) or {}
        recent_mape = m_recent.get("mape")
        prev_mape = m_prev.get("mape")
        recent_hit = m_recent.get("hit_rate")
        prev_hit = m_prev.get("hit_rate")
        tf_alert = False
        reasons: List[str] = []
        if recent_mape is not None and prev_mape is not None and float(prev_mape) > 0:
            if float(recent_mape) >= float(prev_mape) * ratio:
                tf_alert = True
                reasons.append("mape_up")
        if recent_hit is not None and prev_hit is not None:
            if (float(prev_hit) - float(recent_hit)) >= hit_drop:
                tf_alert = True
                reasons.append("hit_rate_down")
        alert = alert or tf_alert
        details.append(
            {
                "timeframe": timeframe,
                "status": "alert" if tf_alert else "ok",
                "reasons": reasons,
                "recent_mape": recent_mape,
                "prev_mape": prev_mape,
                "recent_hit_rate": recent_hit,
                "prev_hit_rate": prev_hit,
                "samples": len(rows),
            }
        )
    return {"alert": alert, "details": details}


def _estimate_eta_from_runs(
    runs: List[Dict[str, Any]], candidate_count: int, config: TournamentConfig
) -> Optional[int]:
    durations: List[float] = []
    counts: List[int] = []
    workers: List[int] = []
    train_days_list: List[int] = []
    val_hours_list: List[int] = []
    for run in runs:
        dur = run.get("duration_seconds")
        if dur is None:
            continue
        try:
            dur_val = float(dur)
        except (TypeError, ValueError):
            continue
        if dur_val <= 0:
            continue
        durations.append(dur_val)
        cnt = run.get("candidate_count")
        if isinstance(cnt, int) and cnt > 0:
            counts.append(cnt)
        wk = run.get("max_workers")
        if isinstance(wk, int) and wk > 0:
            workers.append(wk)
        td = run.get("train_days")
        if isinstance(td, int) and td > 0:
            train_days_list.append(td)
        vh = run.get("val_hours")
        if isinstance(vh, int) and vh > 0:
            val_hours_list.append(vh)
    if not durations:
        return None
    durations.sort()
    mid = len(durations) // 2
    median_duration = durations[mid] if len(durations) % 2 == 1 else (durations[mid - 1] + durations[mid]) / 2.0
    scale = 1.0
    if counts and candidate_count:
        avg_candidates = float(sum(counts) / len(counts))
        if avg_candidates > 0:
            scale *= float(candidate_count) / avg_candidates
    if workers and config.max_workers:
        avg_workers = float(sum(workers) / len(workers))
        if avg_workers > 0:
            scale *= avg_workers / float(config.max_workers)
    if train_days_list and getattr(config, "train_days", None):
        avg_train_days = float(sum(train_days_list) / len(train_days_list))
        if avg_train_days > 0:
            scale *= float(config.train_days) / avg_train_days
    if val_hours_list and getattr(config, "val_hours", None):
        avg_val_hours = float(sum(val_hours_list) / len(val_hours_list))
        if avg_val_hours > 0:
            scale *= float(config.val_hours) / avg_val_hours
    eta = max(30.0, median_duration * scale)
    return int(round(eta))


def _estimate_eta_seconds(config: TournamentConfig, candidate_count: int, limit: int = 20) -> Optional[int]:
    base_filters: Dict[str, Any] = {
        "run_mode": config.run_mode,
        "timeframe": config.timeframe,
        "candle_minutes": config.candle_minutes,
        "train_days": config.train_days,
        "val_hours": config.val_hours,
        "max_workers": config.max_workers,
        "max_candidates_total": config.max_candidates_total,
        "max_candidates_per_target": config.max_candidates_per_target,
        "enable_dl": config.enable_dl,
    }
    relax_steps = [
        {},
        {"max_candidates_total": None, "max_candidates_per_target": None},
        {"train_days": None, "val_hours": None},
        {"enable_dl": None},
        {"timeframe": None, "candle_minutes": None},
        {"max_workers": None},
        {"run_mode": None},
    ]
    for relax in relax_steps:
        filters = dict(base_filters)
        filters.update(relax)
        runs = get_recent_runs(limit=limit, **filters)
        eta = _estimate_eta_from_runs(runs, candidate_count, config)
        if eta is not None:
            return eta
    return None


def _apply_match_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    predicted = row.get("predicted_price")
    actual = row.get("actual_price_1h") if row.get("actual_price_1h") is not None else row.get("actual_price")
    metrics = _compute_match_metrics(predicted, actual)
    if metrics["abs_diff"] is not None:
        row["abs_diff_usd"] = round(float(metrics["abs_diff"]), 2)
    else:
        row["abs_diff_usd"] = None
    if metrics["pct_error"] is not None:
        row["pct_error"] = round(float(metrics["pct_error"]), 6)
    else:
        row["pct_error"] = None
    if metrics["match_percent"] is not None:
        row["match_percent_precise"] = round(float(metrics["match_percent"]), 4)
        row["match_percent"] = float(metrics["match_percent"])
    else:
        row["match_percent_precise"] = None
    return row


def _decorate_last_ready(last_ready: Optional[Dict[str, Any]], horizon_min: int) -> Optional[Dict[str, Any]]:
    if not last_ready:
        return None
    expected_target = _horizon_target_label(horizon_min)
    raw_target = str(last_ready.get("prediction_target") or "").strip().lower()
    reject_legacy = os.getenv("REJECT_LEGACY_LAST_READY", "1").strip().lower() in {"1", "true", "yes", "on"}
    if reject_legacy:
        if not raw_target:
            return None
        if not (raw_target == expected_target or raw_target.endswith(expected_target)):
            return None

    # Ignore very old matched rows so legacy regime rows don't pollute UI.
    max_age_days = max(1, int(os.getenv("LAST_READY_MAX_AGE_DAYS", "120")))
    try:
        pred_at = _parse_iso_utc(str(last_ready.get("predicted_at")))
        if (datetime.now(timezone.utc) - pred_at).total_seconds() > (max_age_days * 86400):
            return None
    except Exception:
        pass

    # Scale sanity check: if last matched predicted price is wildly off current JSLL scale, hide it.
    try:
        predicted_price = float(last_ready.get("predicted_price")) if last_ready.get("predicted_price") is not None else None
    except (TypeError, ValueError):
        predicted_price = None
    if predicted_price is not None and predicted_price > 0:
        ref_price: Optional[float] = None
        try:
            live = get_live_price()
            ref_price = float(live.get("price") or 0.0)
        except Exception:
            pass
        if ref_price is None or ref_price <= 0:
            try:
                ref_price = float(last_ready.get("current_price") or 0.0)
            except (TypeError, ValueError):
                ref_price = None
        if ref_price is not None and ref_price > 0:
            max_rel_gap = float(os.getenv("LAST_READY_MAX_REL_GAP", "3.0"))
            rel_gap = abs(predicted_price - ref_price) / max(1e-9, abs(ref_price))
            if rel_gap > max_rel_gap:
                return None

    try:
        pred_at = _parse_iso_utc(last_ready["predicted_at"])
        delta_min = int(last_ready.get("prediction_horizon_min") or horizon_min)
        tf_minutes = int(last_ready.get("timeframe_minutes") or delta_min or horizon_min)
        tf_minutes = max(1, tf_minutes)
        target_ts = _prediction_target_timestamp(pred_at, delta_min, tf_minutes)
        last_ready["actual_at"] = target_ts.isoformat()
        last_ready["target_iso"] = target_ts.isoformat()
        last_ready["target_aligned"] = True
    except Exception:
        last_ready["actual_at"] = None
        last_ready["target_iso"] = None
        last_ready["target_aligned"] = False
    last_ready["actual_price"] = last_ready.get("actual_price_1h")
    _apply_match_fields(last_ready)
    return last_ready


def _decorate_pending_target(row: Dict[str, Any], horizon_min: int) -> Dict[str, Any]:
    try:
        pred_at = _parse_iso_utc(row["predicted_at"])
        delta_min = int(row.get("prediction_horizon_min") or horizon_min)
        tf_minutes = int(row.get("timeframe_minutes") or delta_min or horizon_min)
        tf_minutes = max(1, tf_minutes)
        target_ts = _prediction_target_timestamp(pred_at, delta_min, tf_minutes)
        row["target_iso"] = target_ts.isoformat()
        row["target_aligned"] = True
    except Exception:
        row["target_iso"] = None
        row["target_aligned"] = False
    return row


def refresh_prediction(config: TournamentConfig) -> Dict[str, Any]:
    if _RUN_STATE.get("running"):
        return latest_prediction(config, update_pending=False)
    ensure_tables()
    update_pending_predictions(config)
    predictions: List[Dict[str, Any]] = []

    try:
        price_info = get_live_price()
        live_price = price_info["price"]
    except Exception:
        live_price = None

    latest_run = get_latest_run()
    run_id = latest_run["id"] if latest_run else None
    market_open = _market_state(datetime.now(timezone.utc)).get("market_open")

    prod_gate_enable = os.getenv("PRODUCTION_GATE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
    prod_ready = _production_ready_map(config) if prod_gate_enable else {}

    for timeframe in get_timeframes(config):
        tf_cfg = _config_for_timeframe(config, timeframe)
        horizon_min = max(1, tf_cfg.candle_minutes)
        target_label = _horizon_target_label(horizon_min)
        latest = get_latest_prediction_for_timeframe(timeframe)
        if latest:
            pred_at = _parse_iso_utc(latest["predicted_at"])
            cooldown = _cooldown_minutes(horizon_min)
            if datetime.now(timezone.utc) - pred_at < timedelta(minutes=cooldown):
                _apply_match_fields(latest)
                _decorate_pending_target(latest, horizon_min)
                latest["last_ready"] = _decorate_last_ready(
                    get_latest_ready_prediction_for_timeframe(timeframe),
                    horizon_min,
                )
                predictions.append(latest)
                continue

        if _is_indian_equity() and market_open is False and latest:
            _apply_match_fields(latest)
            latest["status"] = latest.get("status") or "market_closed"
            latest["last_ready"] = _decorate_last_ready(
                get_latest_ready_prediction_for_timeframe(timeframe),
                horizon_min,
            )
            predictions.append(latest)
            continue

        if live_price is None:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "status": "no_price",
                }
            )
            continue

        if prod_gate_enable and (prod_ready.get(timeframe) is False):
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "prediction_target": target_label,
                    "status": "blocked_by_backtest_gate",
                }
            )
            continue

        tf_reg = _load_registry(tf_cfg.registry_path)
        has_tf_models = bool((tf_reg.get("champions") if isinstance(tf_reg, dict) else None) or (tf_reg.get("ensembles") if isinstance(tf_reg, dict) else None))
        if not has_tf_models:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "prediction_target": target_label,
                    "status": "no_champion",
                }
            )
            continue

        df = _ensure_recent_data(tf_cfg)
        sup = make_supervised(df, candle_minutes=tf_cfg.candle_minutes, feature_windows_hours=tf_cfg.feature_windows)
        if sup.empty:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "prediction_target": target_label,
                    "status": "no_data",
                }
            )
            continue
        latest_row = sup.iloc[-1:]

        regime = _market_regime()
        pred = _predict_return_from_ensemble(tf_cfg, latest_row, regime=regime)
        prediction_target = target_label
        if pred is None:
            pred = _predict_return_from_champion(tf_cfg, latest_row)
        if pred is None:
            pred = _predict_return_from_direction(tf_cfg, latest_row, df)
            prediction_target = f"direction_{target_label}"
        if pred is None:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "prediction_target": target_label,
                    "status": "no_champion",
                }
            )
            continue

        predicted_return = float(pred["predicted_return"])
        bias = _get_recent_bias(tf_cfg, timeframe)
        predicted_return = float(predicted_return + bias)
        calibrator = _fit_return_calibrator(tf_cfg, timeframe)
        predicted_return = _apply_return_calibrator(predicted_return, calibrator)
        confidence_pct = float(pred.get("confidence_pct") or 55.0)
        low_conf_threshold = float(os.getenv("LOW_CONFIDENCE_PCT", "45"))
        downweight = float(os.getenv("LOW_CONF_DOWNWEIGHT", "0.5"))
        skip_threshold = float(os.getenv("LOW_CONFIDENCE_SKIP_PCT", "0"))
        low_confidence = confidence_pct < low_conf_threshold
        skipped_low_conf = skip_threshold > 0 and confidence_pct < skip_threshold
        if low_confidence:
            factor = max(0.1, min(1.0, downweight * (confidence_pct / max(1.0, low_conf_threshold))))
            predicted_return = float(predicted_return * factor)
        predicted_return = _clip_return(predicted_return, horizon_min)
        ret_std = pred.get("uncertainty_return_std")
        try:
            ret_std_val = float(ret_std) if ret_std is not None else max(0.0005, abs(predicted_return) * 0.5)
        except (TypeError, ValueError):
            ret_std_val = max(0.0005, abs(predicted_return) * 0.5)
        max_band_std = float(os.getenv("MAX_RETURN_STD", "0.12"))
        ret_std_val = max(0.0001, min(max_band_std, ret_std_val))
        predicted_price = _price_from_log_return(float(live_price), predicted_return)
        band_z = float(os.getenv("PRED_BAND_Z", "1.64"))
        low_price = _price_from_log_return(float(live_price), predicted_return - (band_z * ret_std_val))
        high_price = _price_from_log_return(float(live_price), predicted_return + (band_z * ret_std_val))
        band_width_pct = (abs(high_price - low_price) / max(1e-9, float(live_price))) * 100.0
        wide_band_pct = float(os.getenv("NO_SIGNAL_BAND_WIDTH_PCT", "8.0"))
        wide_band = band_width_pct >= wide_band_pct
        guardrail_on = os.getenv("NO_SIGNAL_GUARDRAIL_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
        no_signal = guardrail_on and (skipped_low_conf or (low_confidence and wide_band))

        record = {
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "current_price": live_price,
            "predicted_return": predicted_return,
            "predicted_price": predicted_price,
            "actual_price_1h": None,
            "match_percent": None,
            "status": "no_signal" if no_signal else ("skipped_low_confidence" if skipped_low_conf else "pending"),
            "model_name": pred["model_name"],
            "feature_set": pred["feature_set"],
            "run_id": run_id,
            "prediction_target": prediction_target,
            "prediction_horizon_min": horizon_min,
            "timeframe": timeframe,
            "timeframe_minutes": horizon_min,
            "confidence_pct": confidence_pct,
            "low_confidence": low_confidence,
            "predicted_price_low": low_price,
            "predicted_price_high": high_price,
            "regime": pred.get("regime") or regime,
        }
        pred_id = insert_prediction(record)
        record["id"] = pred_id
        record["bias_correction"] = bias
        if skipped_low_conf:
            record["note"] = f"skipped due to low confidence < {skip_threshold:.1f}%"
        elif no_signal:
            record["note"] = (
                f"no-signal guardrail (conf={confidence_pct:.1f}%, band={band_width_pct:.2f}% >= {wide_band_pct:.2f}%)"
            )
        if calibrator.get("active"):
            record["calibration"] = calibrator
        _decorate_pending_target(record, horizon_min)
        if pred.get("ensemble_members"):
            record["ensemble_members"] = pred.get("ensemble_members")
            record["ensemble_size"] = pred.get("ensemble_size")
        _apply_match_fields(record)
        predictions.append(record)

    return {
        "predictions": predictions,
        "metrics_by_horizon": _collect_metrics_by_horizon(config),
        "backtest_report": _build_backtest_report(config),
        "drift_status": _compute_drift_status(config),
    }


def latest_prediction(config: TournamentConfig, update_pending: bool = True) -> Optional[Dict[str, Any]]:
    if update_pending and not _RUN_STATE.get("running"):
        update_pending_predictions(config)
    predictions: List[Dict[str, Any]] = []
    for timeframe in get_timeframes(config):
        latest = get_latest_prediction_for_timeframe(timeframe)
        horizon_min = _timeframe_to_minutes(timeframe, config.candle_minutes)
        if latest:
            _apply_match_fields(latest)
            _decorate_pending_target(latest, horizon_min)
            latest["last_ready"] = _decorate_last_ready(
                get_latest_ready_prediction_for_timeframe(timeframe),
                horizon_min,
            )
            predictions.append(latest)
        else:
            predictions.append(
                {
                    "timeframe": timeframe,
                    "timeframe_minutes": horizon_min,
                    "prediction_horizon_min": horizon_min,
                    "status": "no_prediction",
                }
            )
    return {
        "predictions": predictions,
        "metrics_by_horizon": _collect_metrics_by_horizon(config),
        "backtest_report": _build_backtest_report(config),
        "drift_status": _compute_drift_status(config),
    }


def run_status() -> Dict[str, Any]:
    state = dict(_RUN_STATE)
    file_state = _read_run_state_file()
    if not file_state:
        return state

    running = bool(state.get("running") or file_state.get("running"))
    state["running"] = running

    file_started = _parse_iso(file_state.get("last_started_at"))
    mem_started = _parse_iso(state.get("last_started_at"))
    if file_started and (not mem_started or file_started > mem_started):
        state["last_started_at"] = file_state.get("last_started_at")

    if file_state.get("last_finished_at"):
        state["last_finished_at"] = file_state.get("last_finished_at")
    if file_state.get("progress"):
        state["progress"] = file_state.get("progress")

    started = _parse_iso(state.get("last_started_at"))
    finished = _parse_iso(state.get("last_finished_at"))
    if started and finished:
        duration = max(0.0, (finished - started).total_seconds())
        state["duration_seconds"] = duration

    return state


def _run_tournament_thread(config: TournamentConfig) -> None:
    try:
        from jeena_sikho_tournament.multi_timeframe import run_multi_timeframe_tournament
        run_multi_timeframe_tournament(config)
    finally:
        with _RUN_LOCK:
            _RUN_STATE["running"] = False


def run_tournament_async(config: TournamentConfig, run_mode: Optional[str]) -> Dict[str, Any]:
    with _RUN_LOCK:
        if _RUN_STATE["running"]:
            return {"status": "already_running", **_RUN_STATE}
        if run_mode:
            config.run_mode = run_mode
        _RUN_STATE["running"] = True
        _RUN_STATE["last_started_at"] = datetime.now(timezone.utc).isoformat()
        t = threading.Thread(target=_run_tournament_thread, args=(config,), daemon=True)
        t.start()
    return {"status": "started", **_RUN_STATE}

