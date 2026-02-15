import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import TournamentConfig
from .multi_timeframe import resolve_timeframes


def _load_registry(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _has_any_champion(config: TournamentConfig) -> bool:
    frames = resolve_timeframes(config)
    for tf in frames:
        mins = 60
        if tf.endswith("h") and tf[:-1].isdigit():
            mins = int(tf[:-1]) * 60
        elif tf.endswith("d") and tf[:-1].isdigit():
            mins = int(tf[:-1]) * 24 * 60
        reg_path = config.data_dir / (f"registry_{mins}m.json" if mins != 60 else "registry_60m.json")
        reg = _load_registry(reg_path)
        champs = (reg.get("champions") if isinstance(reg, dict) else None) or {}
        if champs:
            return True
    return False


def _summarize_rows(rows: List[Tuple[float, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    pcts: List[float] = []
    maes: List[float] = []
    hits: List[float] = []
    hit_tol_pct = float(os.getenv("DRIFT_HIT_TOL_PCT", "1.0"))
    for pred, actual in rows:
        try:
            p = float(pred)
            a = float(actual)
        except (TypeError, ValueError):
            continue
        if a == 0:
            continue
        abs_err = abs(p - a)
        maes.append(float(abs_err))
        pct = abs_err / abs(a) * 100.0
        if np.isfinite(pct):
            pcts.append(float(pct))
            hits.append(1.0 if pct <= hit_tol_pct else 0.0)
    if not pcts:
        return {}
    return {
        "mape": float(np.mean(pcts)),
        "mae": float(np.mean(maes)) if maes else float("inf"),
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
    }


def _volatility_state(config: TournamentConfig) -> str:
    if os.getenv("ADAPTIVE_RETRAIN_ENABLE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return "normal"
    table = "ohlcv" if int(config.candle_minutes) == 60 else f"ohlcv_{int(config.candle_minutes)}m"
    db = config.db_path
    if not db.exists():
        return "normal"
    try:
        con = sqlite3.connect(db)
    except Exception:
        return "normal"
    try:
        cur = con.execute(
            f"""
            SELECT close
            FROM {table}
            ORDER BY timestamp_utc DESC
            LIMIT 260
            """
        )
        vals = [float(r[0]) for r in cur.fetchall() if r and r[0] is not None]
    except Exception:
        con.close()
        return "normal"
    finally:
        try:
            con.close()
        except Exception:
            pass
    if len(vals) < 80:
        return "normal"
    arr = np.asarray(list(reversed(vals)), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.diff(np.log(arr))
    ret = ret[np.isfinite(ret)]
    if ret.size < 60:
        return "normal"
    recent = float(np.mean(np.abs(ret[-20:])))
    base = float(np.mean(np.abs(ret[-120:-20]))) if ret.size >= 140 else float(np.mean(np.abs(ret[:-20])))
    if base <= 1e-12:
        return "normal"
    ratio = recent / base
    high_thr = float(os.getenv("VOL_RETRAIN_HIGH_RATIO", "1.35"))
    low_thr = float(os.getenv("VOL_RETRAIN_LOW_RATIO", "0.75"))
    if ratio >= high_thr:
        return "high"
    if ratio <= low_thr:
        return "low"
    return "normal"


def should_retrain_on_drift(config: TournamentConfig) -> Tuple[bool, str]:
    if not _has_any_champion(config):
        return True, "no_champion_yet"
    vol_state = _volatility_state(config)
    if vol_state == "high":
        return True, "volatility_high_force_retrain"
    db = config.db_path
    if not db.exists():
        return True, "no_db"
    window = max(20, int(os.getenv("DRIFT_WINDOW", "60")))
    ratio = float(os.getenv("DRIFT_MAPE_RATIO", "1.2"))
    mae_ratio = float(os.getenv("DRIFT_MAE_RATIO", "1.2"))
    hit_drop = float(os.getenv("DRIFT_HIT_DROP", "0.08"))
    if vol_state == "low":
        ratio = ratio * float(os.getenv("LOW_VOL_MAPE_MULT", "1.15"))
        mae_ratio = mae_ratio * float(os.getenv("LOW_VOL_MAE_MULT", "1.15"))
        hit_drop = hit_drop * float(os.getenv("LOW_VOL_HIT_MULT", "1.2"))
    if hit_drop > 1.0:
        hit_drop = hit_drop / 100.0
    frames = resolve_timeframes(config)
    try:
        con = sqlite3.connect(db)
    except Exception:
        return True, "db_connect_failed"
    try:
        for tf in frames:
            cur = con.execute(
                """
                SELECT predicted_price, actual_price_1h
                FROM btc_predictions
                WHERE status = 'ready' AND timeframe = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (tf, window),
            )
            rows = cur.fetchall()
            if len(rows) < 12:
                return True, f"insufficient_ready_{tf}"
            half = len(rows) // 2
            recent = _summarize_rows(rows[:half])
            prev = _summarize_rows(rows[half:])
            if not recent or not prev:
                return True, f"insufficient_metrics_{tf}"
            if float(recent["mape"]) >= float(prev["mape"]) * ratio:
                return True, f"drift_alert_{tf}"
            if float(recent["mae"]) >= float(prev["mae"]) * mae_ratio:
                return True, f"drift_alert_mae_{tf}"
            if float(recent["hit_rate"]) <= float(prev["hit_rate"]) - hit_drop:
                return True, f"drift_alert_hit_{tf}"
    finally:
        con.close()
    return False, "stable_no_drift"
