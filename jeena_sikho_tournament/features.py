import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


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


def _is_indian_equity_symbol() -> bool:
    sym = (os.getenv("MARKET_YFINANCE_SYMBOL", "") or "").strip().upper()
    return sym.endswith(".NS") or sym.endswith(".BO")


def _safe_symbol_tag(symbol: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(symbol or "").strip().lower()).strip("_")
    return clean or "sym"


def _parse_symbol_list(raw: str, default: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        text = default
    out: list[str] = []
    for token in text.replace("|", ",").replace(";", ",").split(","):
        t = token.strip()
        if t:
            out.append(t)
    return out


def _load_event_calendar(index_utc: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index_utc)
    out["event_result_day"] = 0
    out["event_dividend_day"] = 0
    out["event_corp_announce_day"] = 0
    out["event_any_day"] = 0
    if index_utc.empty:
        return out

    file_path = os.getenv("EVENT_CALENDAR_FILE", "").strip()
    if not file_path:
        file_path = str(Path(os.getenv("APP_DATA_DIR", "data")) / "jsll_events.csv")
    path = Path(file_path)
    if not path.exists():
        return out
    try:
        ev = pd.read_csv(path)
    except Exception:
        return out
    if ev.empty:
        return out

    # Expected columns: date,event_type (result|dividend|corporate)
    date_col = "date" if "date" in ev.columns else None
    type_col = "event_type" if "event_type" in ev.columns else None
    if not date_col:
        return out
    ev = ev.copy()
    ev["__date"] = pd.to_datetime(ev[date_col], errors="coerce").dt.date
    ev = ev.dropna(subset=["__date"])
    if ev.empty:
        return out
    if not type_col:
        ev["__type"] = "corporate"
    else:
        ev["__type"] = ev[type_col].astype(str).str.strip().str.lower()

    idx_local = index_utc.tz_convert("Asia/Kolkata")
    idx_dates = pd.Series(idx_local.date, index=index_utc)
    result_days = set(ev.loc[ev["__type"].str.contains("result", na=False), "__date"].tolist())
    div_days = set(ev.loc[ev["__type"].str.contains("div", na=False), "__date"].tolist())
    corp_days = set(ev.loc[ev["__type"].str.contains("corp|announce|board", na=False), "__date"].tolist())
    all_days = set(ev["__date"].tolist())

    out["event_result_day"] = idx_dates.isin(result_days).astype(int)
    out["event_dividend_day"] = idx_dates.isin(div_days).astype(int)
    out["event_corp_announce_day"] = idx_dates.isin(corp_days).astype(int)
    out["event_any_day"] = idx_dates.isin(all_days).astype(int)
    return out


def _load_delivery_percent(index_utc: pd.DatetimeIndex, volume: pd.Series) -> pd.Series:
    path_raw = os.getenv("DELIVERY_PCT_FILE", "").strip()
    if path_raw:
        path = Path(path_raw)
        if path.exists():
            try:
                ddf = pd.read_csv(path)
                ts_col = "timestamp_utc" if "timestamp_utc" in ddf.columns else ("date" if "date" in ddf.columns else None)
                val_col = "delivery_pct" if "delivery_pct" in ddf.columns else None
                if ts_col and val_col:
                    ddf = ddf[[ts_col, val_col]].dropna()
                    ddf["timestamp_utc"] = pd.to_datetime(ddf[ts_col], errors="coerce", utc=True)
                    ddf = ddf.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc").sort_index()
                    s = ddf[val_col].astype(float).reindex(index_utc).ffill(limit=12)
                    if not s.dropna().empty:
                        return s.clip(0, 100)
            except Exception:
                pass

    # Proxy when exchange delivery file is unavailable.
    win = max(20, int(_bars_for_hours(24, 60)))
    pct_rank = volume.rolling(win, min_periods=5).rank(pct=True) * 100.0
    return pct_rank.clip(0, 100)


def _load_exogenous_context(index_utc: pd.DatetimeIndex, candle_minutes: int) -> pd.DataFrame:
    out = pd.DataFrame(index=index_utc)
    if index_utc.empty:
        return out
    if os.getenv("EXOGENOUS_ENABLE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return out

    symbols = _parse_symbol_list(
        os.getenv("EXOGENOUS_MARKET_SYMBOLS", ""),
        "^NSEI,^NSEBANK,USDINR=X,^INDIAVIX",
    )
    if not symbols:
        return out
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return out

    start = (index_utc.min() - pd.Timedelta(days=7)).to_pydatetime().astimezone(timezone.utc)
    end = (index_utc.max() + pd.Timedelta(days=2)).to_pydatetime().astimezone(timezone.utc)
    interval = "60m" if candle_minutes <= 120 else "1d"
    if interval == "60m":
        max_intraday_days = max(30, int(os.getenv("EXOGENOUS_MAX_INTRADAY_DAYS", "729")))
        now_utc = datetime.now(timezone.utc)
        intraday_floor = now_utc - pd.Timedelta(days=max_intraday_days).to_pytimedelta()
        if start < intraday_floor:
            start = intraday_floor
    chunk_days = max(30, int(os.getenv("EXOGENOUS_CHUNK_DAYS", "700")))

    def _download_chunked(sym: str) -> pd.DataFrame:
        parts = []
        cur = start
        while cur < end:
            nxt = min(cur + pd.Timedelta(days=chunk_days).to_pytimedelta(), end)
            try:
                part = yf.download(
                    sym,
                    start=cur,
                    end=nxt,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
            except Exception:
                part = pd.DataFrame()
            if part is not None and not part.empty:
                parts.append(part)
            cur = nxt
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts)
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [str(c[0]) for c in out.columns]
        out = out[~out.index.duplicated(keep="last")]
        return out.sort_index()

    for sym in symbols:
        tag = _safe_symbol_tag(sym)
        raw = _download_chunked(sym)
        if raw is None or raw.empty:
            continue
        if "Close" not in raw.columns:
            continue
        s = raw["Close"].astype(float).dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(index_utc).ffill(limit=max(1, int(1440 / max(1, candle_minutes))))
        ret = np.log(s).diff(1)
        vol = ret.rolling(max(5, _bars_for_hours(24, candle_minutes)), min_periods=5).std(ddof=0)
        out[f"exo_{tag}_ret"] = ret
        out[f"exo_{tag}_vol"] = vol
    return out


def _next_nse_day_close_target(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    idx_local = df.index.tz_convert("Asia/Kolkata")
    day = pd.Series(idx_local.date, index=df.index)
    day_close = df.groupby(day)["close"].last()
    next_day_close = day_close.shift(-1)
    mapped_next = day.map(next_day_close)
    return np.log((mapped_next + 1e-12) / (df["close"] + 1e-12))


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
        return {"minimal", "base", "momentum", "micro_momentum", "session", "vwap_flow", "signal", "trend", "volatility", "market_ctx"}
    if minutes >= 1440:
        return {"minimal", "base", "long", "trend_longer", "volatility", "session", "signal", "trend", "market_ctx"}
    return {"minimal", "base", "momentum", "signal", "trend", "volatility", "session", "vwap_flow", "trend_longer", "market_ctx"}


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

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for w in [14, 21]:
        bars = _bars_for_hours(w, candle_minutes)
        avg_gain = gain.rolling(bars, min_periods=bars).mean()
        avg_loss = loss.rolling(bars, min_periods=bars).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))

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
    vol_min_periods = min(max(2, int(vwap_window / 4)), int(vwap_window))
    df["volume_shock_pct"] = (
        volume.rolling(vwap_window, min_periods=vol_min_periods).rank(pct=True) * 100.0
    )
    df["vwap_dev_persist"] = df["vwap_dist"].rolling(max(3, _bars_for_hours(6, candle_minutes)), min_periods=3).mean()
    df["delivery_pct"] = _load_delivery_percent(df.index, volume)

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
    prev_day_high = high.groupby(date_series).transform("max").shift(1)
    prev_day_low = low.groupby(date_series).transform("min").shift(1)
    df["prev_day_high_break"] = (close > prev_day_high).astype(int)
    df["prev_day_low_break"] = (close < prev_day_low).astype(int)

    event_df = _load_event_calendar(df.index)
    if not event_df.empty:
        for col in event_df.columns:
            df[col] = event_df[col]

    exog = _load_exogenous_context(df.index, candle_minutes)
    if not exog.empty:
        for col in exog.columns:
            df[col] = exog[col]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def feature_sets(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    exo_cols = [c for c in cols if c.startswith("exo_")]
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
        "volume_shock_pct",
        "delivery_pct",
        "vwap_dev_persist",
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
        "volume_shock_pct",
        "delivery_pct",
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
        "ret_1h",
        "ret_4h",
        "ret_24h",
        "roll_mean_4",
        "roll_mean_12",
        "roll_mean_24",
        "roll_std_4",
        "roll_std_12",
        "roll_std_24",
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
        "prev_day_high_break",
        "prev_day_low_break",
        "event_any_day",
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
        "ret_1c",
        "ret_1h",
        "vol_24",
        "volume_shock_pct",
        "delivery_pct",
        "vwap_dev_persist",
        "prev_day_high_break",
        "prev_day_low_break",
        "event_any_day",
    ]
    market_ctx = [
        "day_of_week",
        "gap_from_prev_close",
        "event_any_day",
    ] + exo_cols

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
        "market_ctx": market_ctx,
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
    target_label = _target_label_for_minutes(candle_minutes)
    if candle_minutes >= 1440 and _is_indian_equity_symbol():
        df[target_label] = _next_nse_day_close_target(df)
    else:
        df[target_label] = df["ret_1c"].shift(-1)
    if candle_minutes <= 120:
        default_low_q, default_high_q = 0.02, 0.98
    elif candle_minutes >= 1440:
        default_low_q, default_high_q = 0.005, 0.995
    else:
        default_low_q, default_high_q = 0.01, 0.99
    low_q = float(os.getenv("TARGET_WINSOR_LOWER", str(default_low_q)))
    high_q = float(os.getenv("TARGET_WINSOR_UPPER", str(default_high_q)))
    low_q = min(max(low_q, 0.0), 0.49)
    high_q = min(max(high_q, 0.51), 1.0)
    winsor_min_rows = max(10, int(os.getenv("TARGET_WINSOR_MIN_ROWS", "200")))
    if len(df) >= winsor_min_rows:
        lo = float(df[target_label].quantile(low_q))
        hi = float(df[target_label].quantile(high_q))
        df[target_label] = df[target_label].clip(lower=lo, upper=hi)

    if candle_minutes <= 120:
        default_event_gap, default_event_clip = 0.06, 0.035
    elif candle_minutes >= 1440:
        default_event_gap, default_event_clip = 0.10, 0.07
    else:
        default_event_gap, default_event_clip = 0.08, 0.05
    event_gap = float(os.getenv("EVENT_GAP_THRESHOLD", str(default_event_gap)))
    event_clip = float(os.getenv("EVENT_TARGET_CLIP", str(default_event_clip)))
    event_mask = pd.Series(False, index=df.index)
    if "gap_from_prev_close" in df.columns:
        event_mask = df["gap_from_prev_close"].abs() >= event_gap
        if event_mask.any():
            df.loc[event_mask, target_label] = df.loc[event_mask, target_label].clip(-event_clip, event_clip)
    df["is_event_day"] = event_mask.astype(int)

    # Keep raw log-return target for evaluation/trading math.
    df["y_ret_raw"] = df[target_label]
    df["y_ret"] = df["y_ret_raw"]
    df["y_dir"] = (df["y_ret_raw"] > 0).astype(int)

    # Optional modeling target normalization by recent volatility.
    target_mode = os.getenv("RETURN_TARGET_MODE", "volnorm_logret").strip().lower()
    vol_floor = float(os.getenv("TARGET_VOL_FLOOR", "0.001"))
    vol_cap = float(os.getenv("TARGET_VOL_CAP", "0.08"))
    target_scale = df.get("vol_24", pd.Series(1.0, index=df.index))
    target_scale = target_scale.clip(lower=max(1e-6, vol_floor), upper=max(vol_floor, vol_cap))
    df["target_scale"] = target_scale
    if target_mode in {"volnorm", "volnorm_logret", "normalized"}:
        df["y_ret_model"] = df["y_ret_raw"] / (target_scale + 1e-12)
    else:
        df["y_ret_model"] = df["y_ret_raw"]

    drop_event_rows = os.getenv("EVENT_DAY_DROP_FROM_TRAIN", "1").strip().lower() in {"1", "true", "yes", "on"}
    if drop_event_rows and "is_event_day" in df.columns:
        df = df.loc[df["is_event_day"] == 0]

    # Avoid wiping whole horizon datasets when optional context features are sparse.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    protected = {
        target_label,
        "y_ret_raw",
        "y_ret",
        "y_ret_model",
        "y_dir",
        "target_scale",
    }
    fill_cols = [c for c in num_cols if c not in protected]
    fill_sparse = os.getenv("FILL_SPARSE_FEATURES", "1").strip().lower() in {"1", "true", "yes", "on"}
    if fill_sparse and fill_cols:
        df[fill_cols] = df[fill_cols].ffill().bfill()
    required = [c for c in [target_label, "y_ret_raw", "y_ret_model", "y_dir", "target_scale"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)
    else:
        df = df.dropna()
    return df
