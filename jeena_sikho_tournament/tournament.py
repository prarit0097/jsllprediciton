import json
import logging
import os
import sqlite3
import warnings
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import trading_score
from .config import TournamentConfig
from .data_sources import fetch_and_stitch
from .features import allowed_feature_sets_for_horizon, feature_sets, make_supervised, resolve_feature_windows_for_horizon
from .metrics import accuracy, f1_score, mae, pinball_loss, coverage
from .models_zoo import ModelSpec, build_quantile_bundle, get_candidates
from .registry import (
    ChampionDecision,
    load_registry,
    record_model_score,
    save_registry,
    stability_penalty,
    update_champion,
)
from .splits import walk_forward_cv_splits
from .storage import Storage
from .validator import validate_ohlcv_quality
from .market_calendar import load_nse_holidays
from .repair import repair_timeframe_data
from jeena_sikho_dashboard.db import PREDICTIONS_TABLE, insert_run, insert_scores

try:
    from sklearn.exceptions import ConvergenceWarning  # type: ignore
except Exception:  # pragma: no cover
    class ConvergenceWarning(Warning):
        pass

try:
    from scipy.linalg import LinAlgWarning  # type: ignore
except Exception:  # pragma: no cover
    class LinAlgWarning(Warning):
        pass

def _update_predictions_safe(config) -> None:
    try:
        from jeena_sikho_dashboard.services import update_pending_predictions
        update_pending_predictions(config)
    except Exception as exc:
        LOGGER.info("Prediction match update skipped: %s", exc)


LOGGER = logging.getLogger("jeena_sikho_tournament")
_RUN_STATE_PATH = Path(os.getenv("APP_DATA_DIR", "data")) / "run_state.json"


def _write_run_state(
    running: bool,
    started_at: Optional[str],
    finished_at: Optional[str],
    status: str,
    progress: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "running": bool(running),
        "last_started_at": started_at,
        "last_finished_at": finished_at,
        "status": status,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "tournament",
    }
    if progress is not None:
        payload["progress"] = progress
    _RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _RUN_STATE_PATH.with_suffix(".tmp")
    retries = max(1, int(os.getenv("RUN_STATE_WRITE_RETRIES", "5")))
    retry_ms = max(20, int(os.getenv("RUN_STATE_WRITE_RETRY_MS", "120")))
    for attempt in range(1, retries + 1):
        try:
            with tmp.open("w", encoding="utf8") as f:
                json.dump(payload, f)
            tmp.replace(_RUN_STATE_PATH)
            return
        except PermissionError as exc:
            if attempt >= retries:
                LOGGER.warning("Run-state write skipped due to file lock: %s", exc)
                return
            time.sleep(retry_ms / 1000.0)
        except OSError as exc:
            if attempt >= retries:
                LOGGER.warning("Run-state write skipped due to OS error: %s", exc)
                return
            time.sleep(retry_ms / 1000.0)


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    max_mb = max(1.0, float(os.getenv("TOURNAMENT_LOG_MAX_MB", "20")))
    backups = max(2, int(os.getenv("TOURNAMENT_LOG_BACKUPS", "8")))
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(max_mb * 1024 * 1024),
        backupCount=backups,
        encoding="utf8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
        force=True,
    )


def _parse_start(config: TournamentConfig) -> datetime:
    return datetime.fromisoformat(config.start_date_utc).replace(tzinfo=timezone.utc)


def _resolve_run_mode(config: TournamentConfig) -> str:
    env_mode = os.getenv("RUN_MODE")
    if env_mode:
        return env_mode
    return config.run_mode


def _run_mode_for_horizon(config: TournamentConfig) -> str:
    minutes = max(1, int(config.candle_minutes))
    if minutes <= 120:
        key = "RUN_MODE_SHORT"
    elif minutes >= 1440:
        key = "RUN_MODE_LONG"
    else:
        key = "RUN_MODE_MID"
    raw = os.getenv(key, "").strip().lower()
    if raw:
        return raw
    return _resolve_run_mode(config)


def _max_candidates_per_target_for_horizon(config: TournamentConfig) -> int:
    minutes = max(1, int(config.candle_minutes))
    if minutes <= 120:
        key = "MAX_CANDIDATES_PER_TARGET_SHORT"
    elif minutes >= 1440:
        key = "MAX_CANDIDATES_PER_TARGET_LONG"
    else:
        key = "MAX_CANDIDATES_PER_TARGET_MID"
    raw = os.getenv(key, "")
    if raw.isdigit():
        return max(10, int(raw))
    return max(10, int(config.max_candidates_per_target))


def _max_candidates_total_for_horizon(config: TournamentConfig) -> int:
    minutes = max(1, int(config.candle_minutes))
    if minutes <= 120:
        key = "MAX_CANDIDATES_TOTAL_SHORT"
    elif minutes >= 1440:
        key = "MAX_CANDIDATES_TOTAL_LONG"
    else:
        key = "MAX_CANDIDATES_TOTAL_MID"
    raw = os.getenv(key, "")
    if raw.isdigit():
        return max(20, int(raw))
    return max(20, int(config.max_candidates_total))


def _prep_data(config: TournamentConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    storage = Storage(config.db_path, config.ohlcv_table)
    storage.init_db()

    start = _parse_start(config)
    existing = storage.load()
    backfill_days = max(1, int(os.getenv("BACKFILL_LOOKBACK_DAYS", "120")))
    fetch_start = start
    if not existing.empty:
        latest = existing.index.max()
        if latest is not None:
            fetch_start = max(start, latest - timedelta(days=backfill_days))
    stitched, report = fetch_and_stitch(
        config.symbol,
        config.yfinance_symbol,
        fetch_start,
        config.timeframe,
        config.candle_minutes,
    )
    if not stitched.empty:
        stitched = stitched.set_index("timestamp_utc")
        storage.upsert(stitched)
    if os.getenv("BACKFILL_GAP_REPAIR", "1").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            repair_timeframe_data(config, lookback_days=backfill_days)
        except Exception as exc:
            LOGGER.warning("Gap-repair step failed: %s", exc)
    storage.trim(pd.Timestamp(start))
    df = storage.load()
    df = df.loc[df.index >= pd.Timestamp(start)]

    coverage = {
        "earliest": str(report.earliest) if report.earliest is not None else None,
        "total_candles": report.total_candles,
        "missing_intervals": report.missing_intervals,
        "interval_minutes": report.interval_minutes,
    }
    return df, coverage


def _build_dataset(df: pd.DataFrame, config: TournamentConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    horizon_windows = resolve_feature_windows_for_horizon(config.candle_minutes, config.feature_windows)
    sup = make_supervised(df, candle_minutes=config.candle_minutes, feature_windows_hours=horizon_windows)
    feature_sets_map = feature_sets(sup)
    if os.getenv("STRICT_HORIZON_FEATURE_POOL", "1").strip().lower() not in {"0", "false", "no", "off"}:
        allowed = allowed_feature_sets_for_horizon(config.candle_minutes)
        feature_sets_map = {k: v for k, v in feature_sets_map.items() if k in allowed}
    return sup, feature_sets_map


def _is_indian_equity(config: TournamentConfig) -> bool:
    sym = (config.yfinance_symbol or "").strip().upper()
    return sym.endswith(".NS") or sym.endswith(".BO")


def _target_mode(config: TournamentConfig) -> str:
    return os.getenv("RETURN_TARGET_MODE", "volnorm_logret").strip().lower()


def _vol_scale(series: pd.Series) -> np.ndarray:
    floor = float(os.getenv("TARGET_VOL_FLOOR", "0.001"))
    cap = float(os.getenv("TARGET_VOL_CAP", "0.08"))
    arr = pd.Series(series).astype(float).clip(lower=max(1e-6, floor), upper=max(floor, cap)).to_numpy()
    return np.where(np.isfinite(arr), arr, max(1e-6, floor))


def _primary_score_direction(acc: float) -> float:
    return float(acc)


def _primary_score_reg(mae_val: float, y_true: np.ndarray) -> float:
    ref = float(np.mean(np.abs(y_true))) + 1e-9
    return max(0.0, 1.0 - (mae_val / ref))


def _close_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, bps: float) -> float:
    tol = max(1.0, float(bps)) / 10000.0
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if err.size == 0:
        return 0.0
    return float((err <= tol).mean())


def _cv_folds_for_horizon(config: TournamentConfig) -> int:
    minutes = max(1, int(config.candle_minutes))
    if minutes <= 120:
        key = "TOURNAMENT_CV_FOLDS_SHORT"
    elif minutes >= 1440:
        key = "TOURNAMENT_CV_FOLDS_LONG"
    else:
        key = "TOURNAMENT_CV_FOLDS_MID"
    raw = os.getenv(key)
    if raw and raw.isdigit():
        return max(1, int(raw))
    return max(1, int(config.cv_folds))


def _ensemble_top_k_for_horizon(config: TournamentConfig) -> int:
    minutes = max(1, int(config.candle_minutes))
    if minutes <= 120:
        key = "ENSEMBLE_TOP_K_SHORT"
    elif minutes >= 1440:
        key = "ENSEMBLE_TOP_K_LONG"
    else:
        key = "ENSEMBLE_TOP_K_MID"
    raw = os.getenv(key)
    if raw and raw.isdigit():
        return max(1, int(raw))
    return max(1, int(config.ensemble_top_k))


def _train_days_for_horizon(config: TournamentConfig) -> int:
    minutes = max(1, int(config.candle_minutes))
    if minutes <= 60:
        key = "TRAIN_DAYS_SHORT"
    elif minutes >= 1440:
        key = "TRAIN_DAYS_LONG"
    else:
        key = "TRAIN_DAYS_MID"
    raw = os.getenv(key)
    if raw and raw.isdigit():
        return max(30, int(raw))
    return max(30, int(config.train_days))


def _horizon_env_suffix(candle_minutes: int) -> str:
    minutes = max(1, int(candle_minutes))
    if minutes % 1440 == 0:
        return f"{max(1, minutes // 1440)}D"
    if minutes % 60 == 0:
        return f"{max(1, minutes // 60)}H"
    return f"{minutes}M"


def _min_sup_rows_for_horizon(candle_minutes: int) -> int:
    minutes = max(1, int(candle_minutes))
    exact_key = f"MIN_SUP_ROWS_{_horizon_env_suffix(minutes)}"
    raw_exact = os.getenv(exact_key)
    if raw_exact and raw_exact.isdigit():
        return max(10, int(raw_exact))
    if minutes <= 60:
        key = "MIN_SUP_ROWS_SHORT"
        default = "240"
    elif minutes >= 1440:
        key = "MIN_SUP_ROWS_LONG"
        default = "120"
    else:
        key = "MIN_SUP_ROWS_MID"
        default = "180"
    raw = os.getenv(key)
    if raw and raw.isdigit():
        return max(30, int(raw))
    return max(30, int(default))


def _completeness_min_for_horizon(candle_minutes: int) -> float:
    minutes = max(1, int(candle_minutes))
    exact_key = f"COMPLETENESS_MIN_PCT_{_horizon_env_suffix(minutes)}"
    raw_exact = os.getenv(exact_key)
    if raw_exact:
        try:
            return float(raw_exact)
        except ValueError:
            pass
    if minutes <= 60:
        key = "COMPLETENESS_MIN_PCT_SHORT"
    elif minutes >= 1440:
        key = "COMPLETENESS_MIN_PCT_LONG"
    else:
        key = "COMPLETENESS_MIN_PCT_MID"
    raw = os.getenv(key)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(os.getenv("COMPLETENESS_MIN_PCT", "95"))


def _min_val_points_for_horizon(config: TournamentConfig) -> int:
    minutes = max(1, int(config.candle_minutes))
    exact_key = f"MIN_VAL_POINTS_{_horizon_env_suffix(minutes)}"
    raw_exact = os.getenv(exact_key)
    if raw_exact and raw_exact.isdigit():
        return max(1, int(raw_exact))
    if minutes <= 60:
        key = "MIN_VAL_POINTS_SHORT"
    elif minutes >= 1440:
        key = "MIN_VAL_POINTS_LONG"
    else:
        key = "MIN_VAL_POINTS_MID"
    raw = os.getenv(key)
    if raw and raw.isdigit():
        return max(1, int(raw))
    return max(1, int(config.min_val_points))


def _target_label_for_minutes(minutes: int) -> str:
    mins = max(1, int(minutes))
    if mins % 1440 == 0:
        return f"y_ret_{max(1, mins // 1440)}d"
    if mins % 60 == 0:
        return f"y_ret_{max(1, mins // 60)}h"
    return f"y_ret_{mins}m"


def _min_recent_matches_for_horizon(candle_minutes: int) -> int:
    minutes = max(1, int(candle_minutes))
    exact_key = f"MIN_READY_MATCHES_{_horizon_env_suffix(minutes)}"
    raw_exact = os.getenv(exact_key)
    if raw_exact and raw_exact.isdigit():
        return max(0, int(raw_exact))
    if minutes <= 60:
        key = "MIN_READY_MATCHES_SHORT"
        default = "24"
    elif minutes >= 1440:
        key = "MIN_READY_MATCHES_LONG"
        default = "8"
    else:
        key = "MIN_READY_MATCHES_MID"
        default = "16"
    raw = os.getenv(key) or os.getenv("MIN_READY_MATCHES", default)
    try:
        return max(0, int(raw))
    except Exception:
        return int(default)


def _recent_ready_match_count(config: TournamentConfig, lookback_days: int = 180) -> int:
    db = config.db_path
    if not db.exists():
        return 0
    target = _target_label_for_minutes(config.candle_minutes)
    since = (datetime.now(timezone.utc) - timedelta(days=max(7, int(lookback_days)))).isoformat()
    try:
        con = sqlite3.connect(db)
        cur = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {PREDICTIONS_TABLE}
            WHERE status = 'ready'
              AND timeframe = ?
              AND predicted_at >= ?
              AND COALESCE(prediction_target, '') IN (?, ?)
            """,
            (str(config.timeframe), since, target, f"direction_{target}"),
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0
    finally:
        try:
            con.close()
        except Exception:
            pass


def _regime_code_from_val(val_df: pd.DataFrame) -> np.ndarray:
    if val_df.empty:
        return np.array([], dtype=int)
    open_f = val_df.get("is_opening_hour", pd.Series(0, index=val_df.index)).astype(int).to_numpy()
    close_f = val_df.get("is_closing_hour", pd.Series(0, index=val_df.index)).astype(int).to_numpy()
    mins = val_df.get("minutes_from_open", pd.Series(0, index=val_df.index)).astype(float).to_numpy()
    out = np.full(len(val_df), 1, dtype=int)  # mid-session default
    out[open_f == 1] = 0
    out[(close_f == 1) & (open_f == 0)] = 2
    out[mins < 0] = 3
    return out


def _inv_err_weight(v: float, floor: float = 1e-8) -> float:
    x = max(floor, float(v))
    return float(1.0 / x)


def _build_member_error_weights(task_results: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    vals: List[float] = []
    keys: List[str] = []
    for r in task_results:
        model_id = _model_id(r["spec"], r["feature_set_id"])
        y_true = np.asarray(r.get("y_true", []), dtype=float)
        y_pred = np.asarray(r.get("y_pred", []), dtype=float)
        if y_true.size == 0 or y_true.size != y_pred.size:
            continue
        mae_val = float(np.mean(np.abs(y_true - y_pred)))
        w = _inv_err_weight(mae_val)
        keys.append(model_id)
        vals.append(w)
    if not vals:
        return out
    total = float(sum(vals)) + 1e-12
    for k, v in zip(keys, vals):
        out[k] = float(v / total)
    return out


def _build_regime_weights(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    names = {0: "opening", 1: "mid_session", 2: "closing", 3: "off_session"}
    out: Dict[str, Dict[str, float]] = {}
    for code, name in names.items():
        vals: List[Tuple[str, float]] = []
        for r in task_results:
            model_id = _model_id(r["spec"], r["feature_set_id"])
            y_true = np.asarray(r.get("y_true", []), dtype=float)
            y_pred = np.asarray(r.get("y_pred", []), dtype=float)
            regimes = np.asarray(r.get("regime_codes", []), dtype=int)
            if y_true.size == 0 or y_true.size != y_pred.size or y_true.size != regimes.size:
                continue
            mask = regimes == code
            if not mask.any():
                continue
            mae_val = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            vals.append((model_id, _inv_err_weight(mae_val)))
        if not vals:
            continue
        s = float(sum(v for _, v in vals)) + 1e-12
        out[name] = {k: float(v / s) for k, v in vals}
    return out


def _regime_top_members(regime_weights: Dict[str, Dict[str, float]], top_k: int = 3) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    k = max(1, int(top_k))
    for regime, weights in regime_weights.items():
        ordered = sorted((weights or {}).items(), key=lambda x: float(x[1]), reverse=True)
        out[regime] = [mid for mid, _ in ordered[:k]]
    return out


def _unstable_features(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    if os.getenv("FEATURE_INSTABILITY_DROP_ENABLE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return set()
    min_rows = max(30, int(os.getenv("FEATURE_INSTABILITY_MIN_ROWS", "120")))
    if len(df) < min_rows:
        return set()
    ratio_thr = float(os.getenv("FEATURE_INSTABILITY_NAN_RATIO", "0.02"))
    std_thr = float(os.getenv("FEATURE_INSTABILITY_STD_MIN", "1e-9"))
    unstable = set()
    for col in df.columns:
        if col.startswith("y_") or col in {"target_scale"}:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if float(s.isna().mean()) > ratio_thr:
            unstable.add(col)
            continue
        if s.dropna().empty:
            unstable.add(col)
            continue
        std = float(s.std(skipna=True))
        if (not np.isfinite(std)) or (std <= std_thr):
            unstable.add(col)
    return unstable


def _primary_score_reg_weighted(
    mae_val: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dir_acc: float,
    close_hit: float,
    candle_minutes: int,
) -> float:
    mae_component = _primary_score_reg(mae_val, y_true)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Calibration proxy: prediction dispersion should be close to true dispersion.
    true_scale = float(np.std(y_true_arr)) + 1e-9
    pred_scale = float(np.std(y_pred_arr)) + 1e-9
    cal_component = max(0.0, 1.0 - min(1.0, abs(pred_scale - true_scale) / true_scale))
    minutes = max(1, int(candle_minutes))
    if minutes <= 60:
        w_mae, w_dir, w_hit, w_cal = 0.45, 0.20, 0.15, 0.20
    elif minutes >= 1440:
        w_mae, w_dir, w_hit, w_cal = 0.25, 0.45, 0.25, 0.05
    else:
        w_mae, w_dir, w_hit, w_cal = 0.45, 0.30, 0.20, 0.05
    return float((w_mae * mae_component) + (w_dir * float(dir_acc)) + (w_hit * float(close_hit)) + (w_cal * cal_component))


def _primary_score_range(pinball: float, y_true: np.ndarray, cov: float) -> float:
    ref = float(np.mean(np.abs(y_true))) + 1e-9
    pin_score = max(0.0, 1.0 - (pinball / ref))
    cov_score = 1.0 - abs(cov - 0.8)
    return 0.5 * pin_score + 0.5 * cov_score


def _final_score(trading: float, primary: float, stability: float, config: TournamentConfig) -> float:
    return 0.5 * trading + 0.3 * primary + config.stability_weight * (1 - stability)


def _fit_predict_direction(spec: ModelSpec, X_train, y_train, X_val):
    model = spec.model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def _fit_predict_reg(spec: ModelSpec, X_train, y_train, X_val):
    model = spec.model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def _fit_predict_range(spec: ModelSpec, X_train, y_train, X_val):
    quantiles = (0.1, 0.5, 0.9)
    model = build_quantile_bundle(spec, quantiles)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def _impute_feature_frames(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Strict model-fit guard: sanitize non-finite values before estimator training.
    Xt = X_train.copy().replace([np.inf, -np.inf], np.nan)
    Xv = X_val.copy().replace([np.inf, -np.inf], np.nan)

    medians = Xt.median(numeric_only=True)
    for col in Xt.columns:
        med = medians.get(col, np.nan)
        if not np.isfinite(med):
            med = 0.0
        Xt[col] = Xt[col].fillna(float(med))
        Xv[col] = Xv[col].fillna(float(med))

    Xt = Xt.fillna(0.0)
    Xv = Xv.fillna(0.0)
    return Xt, Xv


def _clone_model(spec: ModelSpec):
    try:
        return deepcopy(spec.model)
    except Exception:
        return spec.model


def _weak_families(registry: Dict[str, Any]) -> set:
    if os.getenv("WEAK_FAMILY_PRUNE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return set()
    model_hist = registry.get("model_history", {}) or {}
    if not model_hist:
        return set()
    lookback = max(3, int(os.getenv("WEAK_FAMILY_LOOKBACK", "5")))
    bottom_pct = min(0.5, max(0.05, float(os.getenv("WEAK_FAMILY_BOTTOM_PCT", "0.2"))))
    score_max = float(os.getenv("WEAK_FAMILY_SCORE_MAX", "0.45"))
    std_max = float(os.getenv("WEAK_FAMILY_STD_MAX", "0.06"))
    protected = {"naive", "zero", "ema", "bias"}

    stats: List[Tuple[str, float, float]] = []
    for family, rows in model_hist.items():
        if family in protected:
            continue
        if not isinstance(rows, list) or len(rows) < lookback:
            continue
        recent = rows[-lookback:]
        scores = [float(r.get("score", 0.0)) for r in recent]
        if not scores:
            continue
        stats.append((family, float(np.mean(scores)), float(np.std(scores))))

    if not stats:
        return set()
    stats.sort(key=lambda x: x[1])
    k = max(1, int(round(len(stats) * bottom_pct)))
    weak = {fam for fam, mean_s, std_s in stats[:k] if mean_s <= score_max and std_s <= std_max}
    return weak


def _extract_feature_importance(model: Any, feature_cols: List[str], top_k: int = 10) -> List[Dict[str, float]]:
    vals = None
    if hasattr(model, "feature_importances_"):
        try:
            vals = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            vals = None
    elif hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_, dtype=float)
            vals = np.abs(coef.ravel())
        except Exception:
            vals = None
    if vals is None or vals.size == 0:
        return []
    n = min(len(feature_cols), len(vals))
    pairs = [(feature_cols[i], float(vals[i])) for i in range(n)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    out = [{"feature": f, "importance": v} for f, v in pairs[: max(1, int(top_k))]]
    return out


def _feature_drop_set(registry: Dict[str, Any], task: str) -> set:
    if os.getenv("FEATURE_DROP_ENABLE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return set()
    lookback = max(5, int(os.getenv("FEATURE_DROP_LOOKBACK", "10")))
    min_hits = max(1, int(os.getenv("FEATURE_DROP_MIN_HITS", "1")))
    protected = {"ret_1c", "ret_1h", "ret_4h", "ret_24h"}
    hist = (((registry.get("feature_importance") or {}).get(task)) or [])[-lookback:]
    if len(hist) < 5:
        return set(registry.get("feature_drop", {}).get(task, []))
    counts: Dict[str, int] = {}
    for row in hist:
        for item in (row.get("top_features") or []):
            feat = str(item.get("feature"))
            counts[feat] = counts.get(feat, 0) + 1
    existing = set(registry.get("feature_drop", {}).get(task, []))
    if not counts:
        return existing
    to_drop = {f for f, c in counts.items() if c <= min_hits and f not in protected}
    cooldown_runs = max(0, int(os.getenv("FEATURE_DROP_COOLDOWN_RUNS", "2")))
    if cooldown_runs > 0:
        run_count = len(((registry.get("feature_importance") or {}).get(task)) or [])
        if run_count < cooldown_runs:
            to_drop = set()
    merged = existing.union(to_drop)
    max_drop = max(0, int(os.getenv("FEATURE_DROP_MAX", "20")))
    if max_drop and len(merged) > max_drop:
        merged = set(sorted(merged)[:max_drop])
    return merged


def _evaluate_candidate(args: Tuple[ModelSpec, str, str, List[Tuple[pd.DataFrame, pd.DataFrame]], List[str], TournamentConfig]):
    spec, task, feature_set_id, split_pairs, feature_cols, config = args
    fold_metrics: List[Dict[str, float]] = []
    fold_primary: List[float] = []
    fold_trading: List[float] = []
    preds_collect: List[np.ndarray] = []
    truth_collect: List[np.ndarray] = []
    regime_collect: List[np.ndarray] = []
    last_model: Any = None
    ret_mode = _target_mode(config)
    use_volnorm = ret_mode in {"volnorm", "volnorm_logret", "normalized"}

    for train_df, val_df in split_pairs:
        if train_df.empty or val_df.empty:
            continue
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_train, X_val = _impute_feature_frames(X_train, X_val)
        spec_local = ModelSpec(spec.name, _clone_model(spec), spec.task, spec.meta)
        regime_codes = _regime_code_from_val(val_df)

        if task == "direction":
            y_train = train_df["y_dir"].values
            y_val = val_df["y_dir"].values
            model, y_pred = _fit_predict_direction(spec_local, X_train, y_train, X_val)
            acc = accuracy(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            log_ret = val_df["y_ret"].values
            positions = (y_pred > 0).astype(int)
            net, mdd, trading = trading_score(positions, log_ret, config.fee_slippage)
            primary = _primary_score_direction(acc)
            fold_metrics.append({"accuracy": acc, "f1": f1, "net": net, "mdd": mdd})
            fold_primary.append(primary)
            fold_trading.append(trading)
            preds_collect.append(np.asarray(y_pred))
            truth_collect.append(np.asarray(y_val))
            regime_collect.append(regime_codes)
            last_model = model
            continue

        if task == "return":
            y_train_raw = train_df["y_ret"].values
            y_val_raw = val_df["y_ret"].values
            y_train_model = train_df["y_ret_model"].values if use_volnorm and "y_ret_model" in train_df.columns else y_train_raw
            model, y_pred_model = _fit_predict_reg(spec_local, X_train, y_train_model, X_val)
            if use_volnorm and "target_scale" in val_df.columns:
                y_pred = y_pred_model * _vol_scale(val_df["target_scale"])
            else:
                y_pred = y_pred_model
            y_val = y_val_raw
            mae_val = mae(y_val, y_pred)
            rmse_val = float(np.sqrt(np.mean((y_val - y_pred) ** 2))) if len(y_val) else float("inf")
            dir_acc = accuracy((y_val > 0).astype(int), (y_pred > 0).astype(int))
            close_hit = _close_hit_rate(y_val, y_pred, config.close_hit_bps)
            log_ret = y_val
            positions = (y_pred > 0).astype(int)
            net, mdd, trading = trading_score(positions, log_ret, config.fee_slippage)
            primary = _primary_score_reg_weighted(mae_val, y_val, y_pred, dir_acc, close_hit, config.candle_minutes)
            fold_metrics.append(
                {"mae": mae_val, "rmse": rmse_val, "dir_acc": dir_acc, "close_hit": close_hit, "net": net, "mdd": mdd}
            )
            fold_primary.append(primary)
            fold_trading.append(trading)
            preds_collect.append(np.asarray(y_pred))
            truth_collect.append(np.asarray(y_val))
            regime_collect.append(regime_codes)
            last_model = model
            continue

        y_train = train_df["y_ret"].values
        y_val = val_df["y_ret"].values
        y_train_model = train_df["y_ret_model"].values if use_volnorm and "y_ret_model" in train_df.columns else y_train
        model, y_pred_model = _fit_predict_range(spec_local, X_train, y_train_model, X_val)
        if use_volnorm and "target_scale" in val_df.columns:
            scale = _vol_scale(val_df["target_scale"]).reshape(-1, 1)
            y_pred = y_pred_model * scale
        else:
            y_pred = y_pred_model
        p10, p50, p90 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        cov = coverage(y_val, p10, p90)
        pin = (
            pinball_loss(y_val, p10, 0.1)
            + pinball_loss(y_val, p50, 0.5)
            + pinball_loss(y_val, p90, 0.9)
        ) / 3.0
        positions = (p50 > 0).astype(int)
        net, mdd, trading = trading_score(positions, y_val, config.fee_slippage)
        primary = _primary_score_range(pin, y_val, cov)
        fold_metrics.append({"coverage": cov, "pinball": pin, "net": net, "mdd": mdd})
        fold_primary.append(primary)
        fold_trading.append(trading)
        preds_collect.append(np.asarray(p50))
        truth_collect.append(np.asarray(y_val))
        regime_collect.append(regime_codes)
        last_model = model

    if not fold_metrics or last_model is None:
        raise RuntimeError("No valid folds")

    metric_keys = list(fold_metrics[0].keys())
    avg_metrics: Dict[str, float] = {}
    for key in metric_keys:
        avg_metrics[key] = float(np.mean([m[key] for m in fold_metrics]))

    y_pred_full = np.concatenate(preds_collect) if preds_collect else np.array([])
    y_true_full = np.concatenate(truth_collect) if truth_collect else np.array([])
    regime_full = np.concatenate(regime_collect) if regime_collect else np.array([], dtype=int)
    return {
        "spec": spec,
        "feature_set_id": feature_set_id,
        "metrics": avg_metrics,
        "primary": float(np.mean(fold_primary)) if fold_primary else 0.0,
        "trading": float(np.mean(fold_trading)) if fold_trading else 0.0,
        "y_pred": y_pred_full,
        "y_true": y_true_full,
        "regime_codes": regime_full,
        "model": last_model,
        "target_mode": ret_mode if task in {"return", "range"} else "raw_logret",
    }


def _model_id(spec: ModelSpec, feature_set_id: str) -> str:
    return f"{spec.name}__{feature_set_id}"


def _family_label(family: str) -> str:
    mapping = {
        "logreg": "linear",
        "sgd": "linear",
        "ridge": "linear",
        "lasso": "linear",
        "enet": "linear",
        "svr": "svm",
        "knn": "instance",
        "ada": "boosting",
        "rf": "forest",
        "et": "forest",
        "gb": "boosting",
        "hgb": "boosting",
        "xgb": "boosting",
        "lgb": "boosting",
        "cat": "boosting",
        "gbr_q": "boosting",
        "hgb_q": "boosting",
        "lgb_q": "boosting",
        "naive": "baseline",
        "ema": "baseline",
        "bias": "baseline",
        "zero": "baseline",
        "dl": "deep",
    }
    return mapping.get(family, family)


def _save_model_artifacts(
    artifacts_dir: Path,
    model_id: str,
    model: Any,
    task: str,
    feature_set_id: str,
    feature_cols: List[str],
    metrics: Dict[str, Any],
    final_score: float,
    ts: str,
    rank: int,
) -> Tuple[str, str]:
    import joblib

    safe_rank = max(1, rank)
    model_path = artifacts_dir / f"{model_id}_{ts}_r{safe_rank}.pkl"
    meta_path = artifacts_dir / f"{model_id}_{ts}_r{safe_rank}.json"
    joblib.dump(model, model_path)
    meta = {
        "model_id": model_id,
        "task": task,
        "timestamp": ts,
        "rank": safe_rank,
        "feature_set_id": feature_set_id,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "final_score": final_score,
    }
    with meta_path.open("w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return str(model_path), str(meta_path)


def _filter_by_run_mode(candidates: List[Tuple[ModelSpec, str, List[str]]], run_mode: str) -> List[Tuple[ModelSpec, str, List[str]]]:
    groups = {
        "hourly": {"fast"},
        "six_hourly": {"fast", "medium"},
        "daily": {"fast", "medium", "heavy"},
        "all": {"fast", "medium", "heavy"},
    }
    allowed = groups.get(run_mode, {"fast"})
    filtered = []
    for spec, fs_id, cols in candidates:
        group = spec.meta.get("group", "fast")
        if group in allowed:
            filtered.append((spec, fs_id, cols))
    return filtered


def _cap_candidates(
    candidates: List[Tuple[ModelSpec, str, List[str]]],
    max_total: int,
    seed: int,
) -> List[Tuple[ModelSpec, str, List[str]]]:
    rng = np.random.default_rng(seed)
    if len(candidates) <= max_total:
        return candidates
    idx = rng.choice(len(candidates), size=max_total, replace=False)
    return [candidates[i] for i in idx]


def run_tournament(config: TournamentConfig) -> Dict[str, Any]:
    _setup_logging(config.log_path)
    LOGGER.info("Starting tournament")
    run_started_at = datetime.now(timezone.utc)
    _write_run_state(True, run_started_at.isoformat(), None, "running")
    error: Optional[Exception] = None
    progress_total: Optional[int] = None
    progress_done = 0
    progress_failed = 0
    progress_failed_nan = 0
    progress_failed_other = 0
    progress_task: Optional[str] = None
    progress_last_model: Optional[str] = None
    last_progress_write: Optional[datetime] = None

    try:
        df, coverage = _prep_data(config)
        if df.empty:
            raise RuntimeError("No data available")
        holidays = load_nse_holidays(config.data_dir)
        dq_lookback_days = max(5, int(os.getenv("DQ_LOOKBACK_DAYS", "120")))
        dq_cutoff = datetime.now(timezone.utc) - timedelta(days=dq_lookback_days)
        dq_df = df.loc[df.index >= dq_cutoff] if not df.empty else df
        dq = validate_ohlcv_quality(
            dq_df,
            config.candle_minutes,
            nse_mode=_is_indian_equity(config),
            holidays=holidays,
            max_missing_ratio=float(os.getenv("MAX_MISSING_RATIO", "0.15")),
        )
        min_comp_pct = _completeness_min_for_horizon(config.candle_minutes)
        missing_ratio = float((dq.stats or {}).get("missing_ratio", 1.0))
        completeness_pct = max(0.0, min(100.0, (1.0 - missing_ratio) * 100.0))
        if completeness_pct < min_comp_pct:
            dq.errors.append(f"completeness_below_threshold:{completeness_pct:.2f}%<{min_comp_pct:.2f}%")
        if not dq.ok and os.getenv("AUTO_REPAIR_ON_DQ_FAIL", "1").strip().lower() in {"1", "true", "yes", "on"}:
            repair = repair_timeframe_data(config, lookback_days=int(os.getenv("REPAIR_LOOKBACK_DAYS", "120")))
            LOGGER.warning("DQ failed; auto-repair attempted: %s", repair)
            df, coverage = _prep_data(config)
            dq_df = df.loc[df.index >= dq_cutoff] if not df.empty else df
            dq = validate_ohlcv_quality(
                dq_df,
                config.candle_minutes,
                nse_mode=_is_indian_equity(config),
                holidays=holidays,
                max_missing_ratio=float(os.getenv("MAX_MISSING_RATIO", "0.15")),
            )
            missing_ratio = float((dq.stats or {}).get("missing_ratio", 1.0))
            completeness_pct = max(0.0, min(100.0, (1.0 - missing_ratio) * 100.0))
            if completeness_pct < min_comp_pct:
                dq.errors.append(f"completeness_below_threshold:{completeness_pct:.2f}%<{min_comp_pct:.2f}%")
        if completeness_pct < min_comp_pct and os.getenv("AUTO_REPAIR_ON_COMPLETENESS_FAIL", "1").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                repair = repair_timeframe_data(config, lookback_days=int(os.getenv("REPAIR_LOOKBACK_DAYS", "120")))
                LOGGER.warning("Completeness gate failed; auto-repair attempted: %s", repair)
                df, coverage = _prep_data(config)
                dq_df = df.loc[df.index >= dq_cutoff] if not df.empty else df
                dq = validate_ohlcv_quality(
                    dq_df,
                    config.candle_minutes,
                    nse_mode=_is_indian_equity(config),
                    holidays=holidays,
                    max_missing_ratio=float(os.getenv("MAX_MISSING_RATIO", "0.15")),
                )
                missing_ratio = float((dq.stats or {}).get("missing_ratio", 1.0))
                completeness_pct = max(0.0, min(100.0, (1.0 - missing_ratio) * 100.0))
            except Exception as exc:
                LOGGER.warning("Completeness auto-repair failed: %s", exc)
        coverage["dq_errors"] = dq.errors
        coverage["dq_warnings"] = dq.warnings
        coverage["dq_stats"] = dq.stats
        coverage["completeness_pct"] = completeness_pct
        strict_dq = os.getenv("STRICT_DATA_QUALITY", "1").strip().lower() not in {"0", "false", "no", "off"}
        if strict_dq and not dq.ok:
            raise RuntimeError(f"Data quality check failed: {'; '.join(dq.errors)}")
        if completeness_pct < min_comp_pct:
            LOGGER.warning(
                "Skipping run due to completeness gate %.2f%% < %.2f%% (timeframe=%s)",
                completeness_pct,
                min_comp_pct,
                config.timeframe,
            )
            return {
                "coverage": coverage,
                "status": "skipped_completeness_gate",
                "timeframe": config.timeframe,
                "completeness_pct": completeness_pct,
                "min_completeness_pct": min_comp_pct,
            }
        if dq.warnings:
            LOGGER.warning("Data quality warnings: %s", "; ".join(dq.warnings))

        sup, feature_sets_map = _build_dataset(df, config)
        min_sup_rows = _min_sup_rows_for_horizon(config.candle_minutes)
        if len(sup) < min_sup_rows:
            LOGGER.warning(
                "Skipping run due to low supervised rows %d < %d (timeframe=%s)",
                len(sup),
                min_sup_rows,
                config.timeframe,
            )
            return {
                "coverage": coverage,
                "status": "skipped_low_samples",
                "timeframe": config.timeframe,
                "supervised_rows": int(len(sup)),
                "min_supervised_rows": int(min_sup_rows),
            }
        registry = load_registry(config.registry_path)
        train_days_eff = _train_days_for_horizon(config)
        cv_splits = walk_forward_cv_splits(
            sup,
            train_days_eff,
            config.val_hours,
            config.test_hours,
            config.use_test,
            folds=_cv_folds_for_horizon(config),
        )
        if not cv_splits:
            raise RuntimeError("No valid CV splits")
        split_pairs = [(s.train, s.val) for s in cv_splits]
        val_points_total = int(sum(len(s.val) for s in cv_splits))
        weak_fams = _weak_families(registry)

        results: Dict[str, Any] = {"coverage": coverage}
        run_at = run_started_at.isoformat()

        run_mode = _run_mode_for_horizon(config)
        max_candidates_per_target_eff = _max_candidates_per_target_for_horizon(config)
        max_candidates_total_eff = _max_candidates_total_for_horizon(config)

        per_task_candidates: Dict[str, List[Tuple[ModelSpec, str, List[str]]]] = {}
        per_task_feature_cols: Dict[str, Dict[str, List[str]]] = {}
        strict_horizon_pool = os.getenv("STRICT_HORIZON_MODEL_POOL", "1").strip().lower() not in {"0", "false", "no", "off"}
        unstable_feats = _unstable_features(sup)
        for task in ["direction", "return", "range"]:
            max_cands_for_task = max_candidates_per_target_eff
            low_sample_enable = os.getenv("LOW_SAMPLE_CANDIDATE_SHRINK_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
            if low_sample_enable:
                nrows = len(sup)
                if config.candle_minutes >= 1440:
                    low_n = max(50, int(os.getenv("LOW_SAMPLE_ROWS_LONG", "180")))
                elif config.candle_minutes >= 120:
                    low_n = max(120, int(os.getenv("LOW_SAMPLE_ROWS_MID", "360")))
                else:
                    low_n = 0
                if low_n and nrows < low_n:
                    scale = min(1.0, max(0.2, float(os.getenv("LOW_SAMPLE_CANDIDATE_SCALE", "0.5"))))
                    max_cands_for_task = max(20, int(round(max_candidates_per_target_eff * scale)))
            specs = get_candidates(
                task,
                max_cands_for_task,
                config.enable_dl,
                candle_minutes=config.candle_minutes,
                strict_horizon_pool=strict_horizon_pool,
            )
            dropped_features = _feature_drop_set(registry, task)
            dropped_features = dropped_features.union(unstable_feats)
            fs_cols_map: Dict[str, List[str]] = {}
            candidates = []
            for fs_id, cols in feature_sets_map.items():
                cols_use = [c for c in cols if c not in dropped_features]
                if not cols_use:
                    continue
                fs_cols_map[fs_id] = cols_use
                for spec in specs:
                    candidates.append((spec, fs_id, cols_use))
            candidates = _filter_by_run_mode(candidates, run_mode)
            if weak_fams:
                candidates = [
                    (spec, fs_id, cols)
                    for spec, fs_id, cols in candidates
                    if spec.meta.get("family", spec.name) not in weak_fams
                ]
            # Drop candidates whose required features are missing in feature set
            filtered = []
            for spec, fs_id, cols in candidates:
                req = spec.meta.get("required_features", [])
                if all(r in cols for r in req):
                    filtered.append((spec, fs_id, cols))
            candidates = filtered
            if len(candidates) > max_cands_for_task:
                candidates = _cap_candidates(candidates, max_cands_for_task, config.random_seed)
            per_task_candidates[task] = candidates
            per_task_feature_cols[task] = fs_cols_map

        all_candidates = [c for task in per_task_candidates for c in per_task_candidates[task]]
        if len(all_candidates) > max_candidates_total_eff:
            capped = _cap_candidates(all_candidates, max_candidates_total_eff, config.random_seed)
            per_task_candidates = {"direction": [], "return": [], "range": []}
            for spec, fs_id, cols in capped:
                per_task_candidates[spec.task].append((spec, fs_id, cols))

        candidate_count_total = sum(len(per_task_candidates[t]) for t in per_task_candidates)
        progress_total = candidate_count_total

        def _emit_progress(force: bool = False) -> None:
            nonlocal last_progress_write
            if progress_total is None:
                return
            now = datetime.now(timezone.utc)
            if not force and last_progress_write is not None:
                if (now - last_progress_write).total_seconds() < 2.0 and (progress_done % 25) != 0:
                    return
            last_progress_write = now
            _write_run_state(
                True,
                run_started_at.isoformat(),
                None,
                "running",
                progress={
                    "total": progress_total,
                    "done": progress_done,
                    "failed": progress_failed,
                    "task": progress_task,
                    "last_model": progress_last_model,
                    "updated_at": now.isoformat(),
                },
            )

        _emit_progress(force=True)
        scoreboard_rows = []

        for task in ["direction", "return", "range"]:
            candidates = per_task_candidates.get(task, [])
            task_feature_cols = per_task_feature_cols.get(task, {})
            LOGGER.info("Task %s candidates %d", task, len(candidates))
            progress_task = task
            _emit_progress(force=True)

            jobs = [(spec, task, fs_id, split_pairs, cols, config) for spec, fs_id, cols in candidates]
            task_results = []

            use_workers = config.max_workers
            if any(spec.name.startswith("lgb_") for spec, _, _ in candidates):
                use_workers = 1

            if use_workers > 1:
                with ProcessPoolExecutor(max_workers=use_workers) as ex:
                    futures = {ex.submit(_evaluate_candidate, j): j for j in jobs}
                    for fut in futures:
                        job = futures[fut]
                        spec = job[0]
                        fs_id = job[2]
                        try:
                            res = fut.result(timeout=config.model_timeout_sec)
                            task_results.append(res)
                        except TimeoutError:
                            LOGGER.info("Timeout: %s", spec.name)
                            progress_failed += 1
                            progress_failed_other += 1
                        except Exception as exc:
                            LOGGER.warning("Failed: %s %s", spec.name, exc)
                            progress_failed += 1
                            msg = str(exc).lower()
                            if "nan" in msg or "inf" in msg or "input x contains" in msg:
                                progress_failed_nan += 1
                            else:
                                progress_failed_other += 1
                        progress_done += 1
                        progress_last_model = _model_id(spec, fs_id)
                        _emit_progress()
            else:
                for j in jobs:
                    spec = j[0]
                    fs_id = j[2]
                    try:
                        res = _evaluate_candidate(j)
                        task_results.append(res)
                    except Exception as exc:
                        LOGGER.warning("Failed: %s %s", spec.name, exc)
                        progress_failed += 1
                        msg = str(exc).lower()
                        if "nan" in msg or "inf" in msg or "input x contains" in msg:
                            progress_failed_nan += 1
                        else:
                            progress_failed_other += 1
                    progress_done += 1
                    progress_last_model = _model_id(spec, fs_id)
                    _emit_progress()

            if not task_results:
                LOGGER.warning("No valid models for %s", task)
                continue

            for r in task_results:
                spec = r["spec"]
                family = spec.meta.get("family", spec.name)
                stab = stability_penalty(registry, family)
                final = _final_score(r["trading"], r["primary"], stab, config)
                r["final_score"] = final
                r["stability"] = stab
                r["family"] = family

            task_results.sort(key=lambda x: x["final_score"], reverse=True)
            artifacts_dir = config.data_dir / "models" / task
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            top_k = max(1, min(_ensemble_top_k_for_horizon(config), len(task_results)))
            ensemble_members: List[Dict[str, Any]] = []
            best_member: Optional[Dict[str, Any]] = None

            for rank, r in enumerate(task_results[:top_k], start=1):
                spec = r["spec"]
                fs_id = r["feature_set_id"]
                model_id = _model_id(spec, fs_id)
                model_path = None
                meta_path = None
                try:
                    model_path, meta_path = _save_model_artifacts(
                        artifacts_dir,
                        model_id,
                        r["model"],
                        task,
                        fs_id,
                        task_feature_cols.get(fs_id, []),
                        r["metrics"],
                        r["final_score"],
                        ts,
                        rank,
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to save model artifacts: %s", exc)

                member = {
                    "rank": rank,
                    "model_id": model_id,
                    "model_path": model_path,
                    "meta_path": meta_path,
                    "feature_set_id": fs_id,
                    "feature_cols": task_feature_cols.get(fs_id, []),
                    "final_score": r["final_score"],
                    "family": r["family"],
                    "target_mode": r.get("target_mode"),
                }
                if model_path:
                    ensemble_members.append(member)
                if rank == 1:
                    best_member = member

            stacking_info = None
            if task == "return" and len(ensemble_members) >= 2:
                try:
                    import joblib
                    from sklearn.linear_model import Ridge

                    member_ids = [m["model_id"] for m in ensemble_members]
                    preds_map = {
                        _model_id(r["spec"], r["feature_set_id"]): r["y_pred"]
                        for r in task_results
                    }
                    if all(mid in preds_map for mid in member_ids):
                        X_meta = np.column_stack([preds_map[mid] for mid in member_ids])
                        y_true = task_results[0].get("y_true")
                        if isinstance(y_true, np.ndarray) and len(y_true) == X_meta.shape[0]:
                            meta = Ridge(alpha=1.0)
                            meta.fit(X_meta, y_true)
                            meta_path = artifacts_dir / f"stacking_{ts}.pkl"
                            joblib.dump(meta, meta_path)
                            stacking_info = {
                                "model_path": str(meta_path),
                                "member_ids": member_ids,
                                "model": "ridge",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                except Exception as exc:
                    LOGGER.warning("Failed to build stacking model: %s", exc)
            dynamic_error_weights = {}
            regime_weights = {}
            if task == "return" and ensemble_members:
                id_set = {m.get("model_id") for m in ensemble_members}
                filtered_for_weights = [
                    r for r in task_results
                    if _model_id(r["spec"], r["feature_set_id"]) in id_set
                ]
                dynamic_error_weights = _build_member_error_weights(filtered_for_weights)
                regime_weights = _build_regime_weights(filtered_for_weights)

            best = task_results[0]
            record_model_score(registry, best["family"], best["final_score"], config.history_keep)
            best_model_id = _model_id(best["spec"], best["feature_set_id"])
            best_fs_id = best["feature_set_id"]
            best_model_path = best_member["model_path"] if best_member else None

            challenger = {
                "model_id": best_model_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "final_score": best["final_score"],
                "metrics": best["metrics"],
                "val_points": val_points_total,
                "model_path": best_model_path,
                "feature_cols": task_feature_cols.get(best_fs_id, []),
                "feature_set_id": best_fs_id,
                "family": best["family"],
                "target_mode": best.get("target_mode"),
            }
            top_features = _extract_feature_importance(best.get("model"), task_feature_cols.get(best_fs_id, []), top_k=10)
            registry.setdefault("feature_importance", {}).setdefault(task, []).append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_id": best_model_id,
                    "feature_set_id": best_fs_id,
                    "top_features": top_features,
                }
            )
            registry["feature_importance"][task] = registry["feature_importance"][task][-config.history_keep:]
            registry.setdefault("feature_drop", {})[task] = sorted(_feature_drop_set(registry, task))

            ensemble_record = {
                "k": top_k,
                "members": ensemble_members,
                "run_at": run_at,
            }
            if dynamic_error_weights:
                ensemble_record["error_weights"] = dynamic_error_weights
            if regime_weights:
                ensemble_record["regime_weights"] = regime_weights
                regime_k = max(1, int(os.getenv("ROUTING_TOP_MEMBERS", "4")))
                ensemble_record["regime_top_members"] = _regime_top_members(regime_weights, top_k=regime_k)
            if stacking_info:
                ensemble_record["stacking"] = stacking_info
            registry.setdefault("ensembles", {})[task] = ensemble_record

            min_matches = _min_recent_matches_for_horizon(config.candle_minutes) if task == "return" else 0
            matched_cnt = _recent_ready_match_count(config, lookback_days=int(os.getenv("READY_MATCH_LOOKBACK_DAYS", "180"))) if task == "return" else 0
            if task == "return" and matched_cnt < min_matches:
                decision = ChampionDecision(False, "insufficient_recent_matches")
            else:
                decision = update_champion(
                    registry,
                    task,
                    challenger,
                    _min_val_points_for_horizon(config),
                    config.champion_margin,
                    config.champion_margin_override,
                    config.champion_cooldown_hours,
                )

            registry["history"][task].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "best": challenger,
                "decision": decision.reason,
                "coverage": coverage,
                "run_mode": run_mode,
            })
            registry["history"][task] = registry["history"][task][-config.history_keep:]

            results[task] = {
                "best": challenger,
                "decision": decision.reason,
                "top_scores": [
                    {"model_id": _model_id(r["spec"], r["feature_set_id"]), "final_score": r["final_score"]}
                    for r in task_results[:5]
                ],
            }

            for idx, r in enumerate(task_results, start=1):
                metric_name = "accuracy" if task == "direction" else "weighted_obj" if task == "return" else "pinball"
                metric_value = r["metrics"].get("accuracy") if task == "direction" else r.get("primary") if task == "return" else r["metrics"].get("pinball")
                model_name = r["spec"].model.__class__.__name__ if hasattr(r["spec"], "model") else r["spec"].name
                scoreboard_rows.append(
                    {
                        "rank": idx,
                        "target": task,
                        "feature_set": r.get("feature_set_id"),
                        "model_name": model_name,
                        "family": _family_label(r.get("family", "")),
                        "final_score": r.get("final_score"),
                        "primary_metric_name": metric_name,
                        "primary_metric_value": metric_value,
                        "trading_score": r.get("trading"),
                        "stability_penalty": r.get("stability"),
                        "is_champion": idx == 1,
                        "run_at": run_at,
                    }
                )

        run_finished_at = datetime.now(timezone.utc)
        run_at = run_finished_at.isoformat()
        _emit_progress(force=True)
        for task_key, record in registry.get("ensembles", {}).items():
            if isinstance(record, dict):
                record["run_at"] = run_at
        for row in scoreboard_rows:
            row["run_at"] = run_at
        save_registry(config.registry_path, registry)
        try:
            duration_seconds = (run_finished_at - run_started_at).total_seconds()
            run_id = insert_run(
                run_at,
                run_mode,
                candidate_count_total,
                run_started_at=run_started_at.isoformat(),
                run_finished_at=run_finished_at.isoformat(),
                duration_seconds=duration_seconds,
                max_workers=config.max_workers,
                timeframe=config.timeframe,
                candle_minutes=config.candle_minutes,
                train_days=train_days_eff,
                val_hours=config.val_hours,
                max_candidates_total=config.max_candidates_total,
                max_candidates_per_target=config.max_candidates_per_target,
                enable_dl=config.enable_dl,
                ensemble_top_k=config.ensemble_top_k,
            )
            insert_scores(run_id, scoreboard_rows)
        except Exception as exc:
            LOGGER.warning("Failed to store scoreboard: %s", exc)
        _update_predictions_safe(config)
        results["run_audit"] = {
            "failed_models": int(progress_failed),
            "failed_nan_like": int(progress_failed_nan),
            "failed_other": int(progress_failed_other),
            "train_days_effective": int(train_days_eff),
            "supervised_rows": int(len(sup)),
            "min_supervised_rows": int(min_sup_rows),
        }
        LOGGER.info("Tournament complete")
        return results
    except Exception as exc:
        error = exc
        raise
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        status = "error" if error else "ok"
        progress_payload = None
        if progress_total is not None:
            progress_payload = {
                "total": progress_total,
                "done": progress_done,
                "failed": progress_failed,
                "task": progress_task,
                "last_model": progress_last_model,
                "updated_at": finished_at,
            }
        _write_run_state(
            False,
            run_started_at.isoformat(),
            finished_at,
            status,
            progress=progress_payload,
        )

