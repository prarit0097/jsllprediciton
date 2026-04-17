import json
import importlib.util
import logging
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import trading_score
from .config import TournamentConfig
from .data_sources import fetch_and_stitch
from .features import allowed_feature_sets_for_horizon, feature_sets, make_supervised, resolve_feature_windows_for_horizon
from .forecast_metrics import summarize_price_forecast
from .metrics import accuracy, f1_score, mae, pinball_loss, coverage
from .models_zoo import ModelSpec, build_quantile_bundle, get_candidates
from .registry import (
    get_promotion_state,
    load_registry,
    record_model_score,
    record_promotion_state,
    save_registry,
    stability_penalty,
    update_champion,
)
from .splits import walk_forward_cv_splits, walk_forward_split
from .storage import Storage
from .validator import validate_ohlcv_quality
from .market_calendar import load_nse_holidays
from .repair import repair_timeframe_data
from jeena_sikho_dashboard.db import get_recent_ready_predictions, insert_run, insert_scores

def _update_predictions_safe(config) -> None:
    try:
        from jeena_sikho_dashboard.services import refresh_prediction, update_pending_predictions
        update_pending_predictions(config)
        refresh_prediction(config)
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
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "tournament",
    }
    if progress is not None:
        payload["progress"] = progress
    _RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _RUN_STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as f:
        json.dump(payload, f)
    tmp.replace(_RUN_STATE_PATH)


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf8"), logging.StreamHandler()],
    )


def _parse_start(config: TournamentConfig) -> datetime:
    return datetime.fromisoformat(config.start_date_utc).replace(tzinfo=timezone.utc)


def _resolve_run_mode(config: TournamentConfig) -> str:
    env_mode = os.getenv("RUN_MODE")
    if env_mode:
        return env_mode
    return config.run_mode


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


def _target_clip_quantiles(candle_minutes: int) -> tuple[float, float]:
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
    return low_q, high_q


def _event_target_clip(candle_minutes: int) -> float:
    if candle_minutes <= 120:
        default_clip = 0.035
    elif candle_minutes >= 1440:
        default_clip = 0.07
    else:
        default_clip = 0.05
    return float(os.getenv("EVENT_TARGET_CLIP", str(default_clip)))


def _fit_target_clip_bounds(y_train: pd.Series, candle_minutes: int) -> tuple[float, float]:
    clean = pd.Series(y_train).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("-inf"), float("inf")
    low_q, high_q = _target_clip_quantiles(candle_minutes)
    return float(clean.quantile(low_q)), float(clean.quantile(high_q))


def _prepare_training_frame(train_df: pd.DataFrame, config: TournamentConfig) -> pd.DataFrame:
    prepared = train_df.copy()
    drop_event_rows = os.getenv("EVENT_DAY_DROP_FROM_TRAIN", "1").strip().lower() in {"1", "true", "yes", "on"}
    if drop_event_rows and "is_event_day" in prepared.columns:
        prepared = prepared.loc[prepared["is_event_day"] == 0].copy()
    if prepared.empty:
        return prepared

    y_train = pd.Series(prepared["y_ret_raw"], index=prepared.index, dtype=float)
    lo, hi = _fit_target_clip_bounds(y_train, config.candle_minutes)
    y_train = y_train.clip(lower=lo, upper=hi)

    if not drop_event_rows and "is_event_day" in prepared.columns:
        mask = prepared["is_event_day"].astype(bool)
        if mask.any():
            event_clip = _event_target_clip(config.candle_minutes)
            y_train.loc[mask] = y_train.loc[mask].clip(-event_clip, event_clip)

    prepared["y_ret_train"] = y_train
    prepared["y_dir_train"] = (prepared["y_ret_train"] > 0).astype(int)
    ret_mode = _target_mode(config)
    if ret_mode in {"volnorm", "volnorm_logret", "normalized"}:
        prepared["y_ret_model_train"] = prepared["y_ret_train"] / (_vol_scale(prepared["target_scale"]) + 1e-12)
    else:
        prepared["y_ret_model_train"] = prepared["y_ret_train"]
    return prepared


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


def _primary_score_reg_weighted(mae_val: float, y_true: np.ndarray, dir_acc: float, close_hit: float) -> float:
    mae_component = _primary_score_reg(mae_val, y_true)
    # Weighted objective requested for JSLL:
    # 0.5*MAE(accuracy-style) + 0.3*direction_accuracy + 0.2*close_hit_rate
    return float(0.5 * mae_component + 0.3 * float(dir_acc) + 0.2 * float(close_hit))


def _primary_score_range(pinball: float, y_true: np.ndarray, cov: float) -> float:
    ref = float(np.mean(np.abs(y_true))) + 1e-9
    pin_score = max(0.0, 1.0 - (pinball / ref))
    cov_score = 1.0 - abs(cov - 0.8)
    return 0.5 * pin_score + 0.5 * cov_score


def _final_score(trading: float, primary: float, stability: float, config: TournamentConfig) -> float:
    return 0.5 * trading + 0.3 * primary + config.stability_weight * (1 - stability)


def _final_score_for_task(task: str, primary: float, trading: float, stability: float, config: TournamentConfig) -> float:
    if task == "return":
        return float((0.95 * primary) + (0.05 * (1 - stability)))
    return _final_score(trading, primary, stability, config)


def _price_metrics_from_returns(eval_df: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    base_price = eval_df["close"].to_numpy(dtype=float)
    pred_price = base_price * np.exp(np.asarray(y_pred, dtype=float))
    actual_price = base_price * np.exp(np.asarray(y_true, dtype=float))
    abs_diff = np.abs(pred_price - actual_price)
    sq_diff = np.square(pred_price - actual_price)
    pct_err = np.where(actual_price != 0.0, (abs_diff / np.abs(actual_price)) * 100.0, np.nan)
    denom = np.abs(pred_price) + np.abs(actual_price)
    smape_vals = np.where(denom != 0.0, (200.0 * abs_diff) / denom, np.nan)
    direction_hit = (np.sign(y_pred) == np.sign(y_true)).astype(float)
    sample_count = int(len(abs_diff))
    return {
        "price_mae": float(np.mean(abs_diff)) if len(abs_diff) else 0.0,
        "price_rmse": float(np.sqrt(np.mean(sq_diff))) if len(sq_diff) else 0.0,
        "median_abs_error": float(np.median(abs_diff)) if len(abs_diff) else 0.0,
        "p90_abs_error": float(np.quantile(abs_diff, 0.9)) if len(abs_diff) else 0.0,
        "price_mape": float(np.nanmean(pct_err)) if len(pct_err) else 0.0,
        "smape": float(np.nanmean(smape_vals)) if len(smape_vals) else 0.0,
        "return_mae": float(mae(y_true, y_pred)),
        "avg_abs_return": float(np.mean(np.abs(y_true))) if len(y_true) else 0.0,
        "direction_hit_rate": float(np.mean(direction_hit) * 100.0) if len(direction_hit) else 0.0,
        "avg_actual_price": float(np.mean(np.abs(actual_price))) if len(actual_price) else 1.0,
        "signed_bias_rs": float(np.mean(pred_price - actual_price)) if len(actual_price) else 0.0,
        "band_80_coverage": None,
        "sample_count": sample_count,
    }


def _selection_score_return(metrics: Dict[str, float]) -> float:
    mean_price = float(metrics.get("avg_actual_price", 1.0)) + 1e-9
    price_mae = float(metrics.get("price_mae", 0.0))
    price_mape = float(metrics.get("price_mape", 100.0))
    return_mae = float(metrics.get("return_mae", 0.0))
    avg_abs_return = float(metrics.get("avg_abs_return", 0.0))
    hit_rate = float(metrics.get("direction_hit_rate", 0.0)) / 100.0

    price_mae_score = max(0.0, 1.0 - (price_mae / max(1.0, mean_price)))
    price_mape_score = max(0.0, 1.0 - (price_mape / 100.0))
    return_ref = max(1e-6, avg_abs_return)
    return_mae_score = max(0.0, 1.0 - (return_mae / return_ref))
    return float((0.7 * price_mae_score) + (0.15 * price_mape_score) + (0.1 * return_mae_score) + (0.05 * hit_rate))


def _safe_metric_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _result_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get("holdout_metrics") or result.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def _result_price_mae(result: Dict[str, Any]) -> Optional[float]:
    return _safe_metric_float(_result_metrics(result).get("price_mae"))


def _result_sort_key(task: str, result: Dict[str, Any]) -> Tuple[float, float]:
    if task == "return":
        price_mae = _result_price_mae(result)
        return (price_mae if price_mae is not None else float("inf"), -float(result.get("final_score", 0.0)))
    return (-float(result.get("final_score", 0.0)), 0.0)


def _comparison_payload(reference_id: Optional[str], reference_mae: Optional[float], challenger_mae: Optional[float]) -> Dict[str, Any]:
    improvement_ratio = None
    passed = None
    if reference_mae is not None and challenger_mae is not None:
        passed = challenger_mae < reference_mae
        if reference_mae > 0:
            improvement_ratio = float((reference_mae - challenger_mae) / reference_mae)
    return {
        "model_id": reference_id,
        "price_mae": reference_mae,
        "challenger_price_mae": challenger_mae,
        "improvement_ratio": improvement_ratio,
        "passed": passed,
    }


def _served_timeframes(config: TournamentConfig) -> List[str]:
    env_value = os.getenv("MARKET_TIMEFRAMES") or os.getenv("TIMEFRAMES") or os.getenv("BTC_TIMEFRAMES")
    if not env_value:
        return [config.timeframe]
    tokens: List[str] = []
    for part in env_value.replace(";", ",").replace("|", ",").split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens or [config.timeframe]


def _timeframe_minutes(timeframe: str, fallback: int) -> int:
    tf = (timeframe or "").strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("d") and tf[:-1].isdigit():
        return int(tf[:-1]) * 24 * 60
    return fallback


def _artifact_path_for_timeframe(data_dir: Path, timeframe: str, fallback_minutes: int) -> Path:
    minutes = _timeframe_minutes(timeframe, fallback_minutes)
    return data_dir / f"run_artifact_{minutes}m.json"


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _runtime_capability_status() -> Dict[str, Any]:
    modules = {
        "sklearn": _module_available("sklearn"),
        "joblib": _module_available("joblib"),
        "xgboost": _module_available("xgboost"),
        "lightgbm": _module_available("lightgbm"),
        "catboost": _module_available("catboost"),
        "ccxt": _module_available("ccxt"),
    }
    core_ml = modules["sklearn"] and modules["joblib"]
    full_ensemble = core_ml and modules["xgboost"] and modules["lightgbm"] and modules["catboost"] and modules["ccxt"]
    if full_ensemble:
        status = "full-ensemble"
    elif core_ml:
        status = "core-ml"
    else:
        status = "baseline-only"

    expected = (os.getenv("PROD_CAPABILITY_EXPECTATION") or os.getenv("PRODUCTION_CAPABILITY_EXPECTATION") or "baseline-only").strip().lower()
    order = {"baseline-only": 0, "core-ml": 1, "full-ensemble": 2}
    if expected not in order:
        expected = "baseline-only"

    passed = order[status] >= order[expected]
    missing_modules: List[str] = []
    if expected in {"core-ml", "full-ensemble"}:
        for module in ["sklearn", "joblib"]:
            if not modules[module]:
                missing_modules.append(module)
    if expected == "full-ensemble":
        for module in ["xgboost", "lightgbm", "catboost", "ccxt"]:
            if not modules[module]:
                missing_modules.append(module)

    return {
        "status": status,
        "expected": expected,
        "passed": passed,
        "modules": modules,
        "missing_modules": missing_modules,
    }


def _shadow_min_settled(timeframe: str) -> int:
    tf = (timeframe or "").strip().lower()
    if tf == "1d":
        default_value = 10
        suffix = "1D"
    elif tf == "2h":
        default_value = 20
        suffix = "2H"
    else:
        default_value = 20
        suffix = "1H"
    raw = os.getenv(f"SHADOW_PROMOTION_MIN_SETTLED_{suffix}")
    try:
        return max(1, int(raw)) if raw is not None else default_value
    except (TypeError, ValueError):
        return default_value


def _shadow_promotion_report(timeframe: str) -> Dict[str, Any]:
    required = _shadow_min_settled(timeframe)
    rows = get_recent_ready_predictions(timeframe, max(required, 20))
    valid_rows = []
    for row in rows:
        predicted_price = _safe_metric_float(row.get("predicted_price"))
        actual_price = _safe_metric_float(row.get("actual_price_1h"))
        if predicted_price is None or actual_price is None:
            continue
        valid_rows.append((predicted_price, actual_price))
    settled_count = len(valid_rows)
    recent_price_mae = None
    if valid_rows:
        recent_price_mae = float(np.mean([abs(pred - actual) for pred, actual in valid_rows]))
    passed = settled_count >= required
    return {
        "timeframe": timeframe,
        "required_min_settled": required,
        "settled_count": settled_count,
        "passed": passed,
        "recent_price_mae": recent_price_mae,
    }


def _cross_horizon_report(config: TournamentConfig) -> Dict[str, Any]:
    max_regression = float(os.getenv("PROD_MAX_CROSS_HORIZON_REGRESSION", "0.03"))
    entries: List[Dict[str, Any]] = []
    for timeframe in _served_timeframes(config):
        if timeframe == config.timeframe:
            continue
        artifact = _load_json_file(_artifact_path_for_timeframe(config.data_dir, timeframe, config.candle_minutes))
        served = artifact.get("served_artifact") if isinstance(artifact, dict) else None
        snapshot = artifact.get("baseline_accuracy_snapshot") if isinstance(artifact, dict) else None
        if not isinstance(served, dict) or not isinstance(snapshot, dict):
            entries.append({"timeframe": timeframe, "status": "unavailable", "passed": True})
            continue
        current_metrics = served.get("holdout_metrics") if isinstance(served.get("holdout_metrics"), dict) else {}
        baseline_metrics = snapshot.get("holdout_metrics") if isinstance(snapshot.get("holdout_metrics"), dict) else {}
        current_mae = _safe_metric_float(current_metrics.get("price_mae"))
        baseline_mae = _safe_metric_float(baseline_metrics.get("price_mae"))
        if current_mae is None or baseline_mae is None or baseline_mae <= 0:
            entries.append({"timeframe": timeframe, "status": "missing_price_mae", "passed": True})
            continue
        regression_ratio = float((current_mae - baseline_mae) / baseline_mae)
        passed = regression_ratio <= max_regression
        entries.append(
            {
                "timeframe": timeframe,
                "baseline_price_mae": baseline_mae,
                "current_price_mae": current_mae,
                "regression_ratio": regression_ratio,
                "max_allowed_regression": max_regression,
                "passed": passed,
            }
        )
    return {
        "max_allowed_regression": max_regression,
        "entries": entries,
        "passed": all(entry.get("passed") is not False for entry in entries),
    }


def _baseline_comparison(
    task: str,
    best: Dict[str, Any],
    task_results: List[Dict[str, Any]],
    registry: Dict[str, Any],
    config: TournamentConfig,
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {
        "selection_metric": "holdout_price_mae" if best.get("holdout_metrics") is not None else "price_mae",
        "runtime_capability": _runtime_capability_status(),
    }
    if task != "return":
        return comparison
    challenger_mae = _result_price_mae(best)
    baseline_rows = [
        row
        for row in task_results
        if bool(row.get("spec").meta.get("baseline")) and row.get("spec").name in {"naive_zero", "naive_last"}
    ]
    if baseline_rows:
        baseline_rows.sort(
            key=lambda row: (_result_price_mae(row) if _result_price_mae(row) is not None else float("inf"))
        )
        baseline = baseline_rows[0]
        comparison["naive_last_close"] = _comparison_payload(
            _model_id(baseline["spec"], baseline["feature_set_id"]),
            _result_price_mae(baseline),
            challenger_mae,
        )
    incumbent = registry.get("champions", {}).get(task) or {}
    incumbent_metrics = incumbent.get("holdout_metrics") or incumbent.get("metrics") or {}
    incumbent_mae = _safe_metric_float(incumbent_metrics.get("price_mae"))
    if incumbent:
        comparison["incumbent"] = _comparison_payload(
            incumbent.get("model_id"),
            incumbent_mae,
            challenger_mae,
        )
    shadow_report = _shadow_promotion_report(config.timeframe)
    comparison["shadow_promotion_report"] = shadow_report
    comparison["shadow_window_passed"] = shadow_report.get("passed")
    comparison["cross_horizon"] = _cross_horizon_report(config)
    return comparison


def _fit_predict_direction(spec: ModelSpec, X_train, y_train, X_val):
    model = spec.model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def _fit_predict_reg(spec: ModelSpec, X_train, y_train, X_val):
    model = spec.model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def _fit_predict_range(spec: ModelSpec, X_train, y_train, X_val):
    quantiles = (0.1, 0.5, 0.9)
    model = build_quantile_bundle(spec, quantiles)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def _fit_final_model(
    spec: ModelSpec,
    task: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    config: TournamentConfig,
):
    prepared_train = _prepare_training_frame(train_df, config)
    if prepared_train.empty:
        raise RuntimeError("No rows available for final refit")

    X_train = prepared_train[feature_cols]
    spec_local = ModelSpec(spec.name, _clone_model(spec), spec.task, spec.meta)

    if task == "direction":
        model = spec_local.model
        model.fit(X_train, prepared_train["y_dir_train"].values)
        return model

    if task == "return":
        model = spec_local.model
        model.fit(X_train, prepared_train["y_ret_model_train"].values)
        return model

    model = build_quantile_bundle(spec_local, (0.1, 0.5, 0.9))
    y_train = prepared_train["y_ret_model_train"].values if _target_mode(config) in {"volnorm", "volnorm_logret", "normalized"} else prepared_train["y_ret_train"].values
    model.fit(X_train, y_train)
    return model


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
    merged = existing.union(to_drop)
    max_drop = max(0, int(os.getenv("FEATURE_DROP_MAX", "20")))
    if max_drop and len(merged) > max_drop:
        merged = set(sorted(merged)[:max_drop])
    return merged


def _score_candidate_once(
    spec: ModelSpec,
    task: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TournamentConfig,
) -> Dict[str, Any]:
    prepared_train = _prepare_training_frame(train_df, config)
    if prepared_train.empty or eval_df.empty:
        raise RuntimeError("No valid train/eval rows")

    X_train = prepared_train[feature_cols]
    X_eval = eval_df[feature_cols]
    spec_local = ModelSpec(spec.name, _clone_model(spec), spec.task, spec.meta)
    ret_mode = _target_mode(config)
    use_volnorm = ret_mode in {"volnorm", "volnorm_logret", "normalized"}

    if task == "direction":
        y_train = prepared_train["y_dir_train"].values
        y_eval = eval_df["y_dir"].values
        model, y_pred = _fit_predict_direction(spec_local, X_train, y_train, X_eval)
        acc = accuracy(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)
        log_ret = eval_df["y_ret_raw"].values
        positions = (y_pred > 0).astype(int)
        net, mdd, trading = trading_score(positions, log_ret, config.fee_slippage)
        primary = _primary_score_direction(acc)
        return {
            "metrics": {"accuracy": acc, "f1": f1, "net": net, "mdd": mdd},
            "primary": primary,
            "trading": trading,
            "y_pred": np.asarray(y_pred),
            "y_true": np.asarray(y_eval),
            "model": model,
            "target_mode": "raw_logret",
        }

    if task == "return":
        y_train_model = prepared_train["y_ret_model_train"].values
        model, y_pred_model = _fit_predict_reg(spec_local, X_train, y_train_model, X_eval)
        if use_volnorm and "target_scale" in eval_df.columns:
            y_pred = y_pred_model * _vol_scale(eval_df["target_scale"])
        else:
            y_pred = y_pred_model
        y_eval = eval_df["y_ret_raw"].values
        price_metrics = _price_metrics_from_returns(eval_df, y_pred, y_eval)
        mae_val = float(price_metrics["return_mae"])
        dir_acc = float(price_metrics["direction_hit_rate"] / 100.0)
        close_hit = _close_hit_rate(y_eval, y_pred, config.close_hit_bps)
        positions = (y_pred > 0).astype(int)
        net, mdd, trading = trading_score(positions, y_eval, config.fee_slippage)
        primary = _selection_score_return(price_metrics)
        return {
            "metrics": {
                "price_mae": price_metrics["price_mae"],
                "price_rmse": price_metrics["price_rmse"],
                "median_abs_error": price_metrics["median_abs_error"],
                "p90_abs_error": price_metrics["p90_abs_error"],
                "price_mape": price_metrics["price_mape"],
                "smape": price_metrics["smape"],
                "return_mae": mae_val,
                "direction_hit_rate": price_metrics["direction_hit_rate"],
                "signed_bias_rs": price_metrics["signed_bias_rs"],
                "band_80_coverage": price_metrics["band_80_coverage"],
                "sample_count": price_metrics["sample_count"],
                "close_hit": close_hit,
                "net": net,
                "mdd": mdd,
            },
            "primary": primary,
            "trading": trading,
            "y_pred": np.asarray(y_pred),
            "y_true": np.asarray(y_eval),
            "model": model,
            "target_mode": ret_mode,
        }

    y_train_model = prepared_train["y_ret_model_train"].values if use_volnorm else prepared_train["y_ret_train"].values
    model, y_pred_model = _fit_predict_range(spec_local, X_train, y_train_model, X_eval)
    if use_volnorm and "target_scale" in eval_df.columns:
        scale = _vol_scale(eval_df["target_scale"]).reshape(-1, 1)
        y_pred = y_pred_model * scale
    else:
        y_pred = y_pred_model
    y_eval = eval_df["y_ret_raw"].values
    p10, p50, p90 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    cov = coverage(y_eval, p10, p90)
    pin = (
        pinball_loss(y_eval, p10, 0.1)
        + pinball_loss(y_eval, p50, 0.5)
        + pinball_loss(y_eval, p90, 0.9)
    ) / 3.0
    positions = (p50 > 0).astype(int)
    net, mdd, trading = trading_score(positions, y_eval, config.fee_slippage)
    primary = _primary_score_range(pin, y_eval, cov)
    return {
        "metrics": {"coverage": cov, "pinball": pin, "net": net, "mdd": mdd},
        "primary": primary,
        "trading": trading,
        "y_pred": np.asarray(p50),
        "y_true": np.asarray(y_eval),
        "model": model,
        "target_mode": ret_mode,
    }


def _evaluate_candidate(
    args: Tuple[
        ModelSpec,
        str,
        str,
        List[Tuple[pd.DataFrame, pd.DataFrame]],
        List[str],
        TournamentConfig,
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
    ]
):
    spec, task, feature_set_id, split_pairs, feature_cols, config, holdout_train_df, holdout_eval_df = args
    fold_metrics: List[Dict[str, float]] = []
    fold_primary: List[float] = []
    fold_trading: List[float] = []
    preds_collect: List[np.ndarray] = []
    truth_collect: List[np.ndarray] = []
    last_model: Any = None

    for train_df, val_df in split_pairs:
        fold = _score_candidate_once(spec, task, feature_cols, train_df, val_df, config)
        fold_metrics.append(fold["metrics"])
        fold_primary.append(float(fold["primary"]))
        fold_trading.append(float(fold["trading"]))
        preds_collect.append(np.asarray(fold["y_pred"]))
        truth_collect.append(np.asarray(fold["y_true"]))
        last_model = fold["model"]

    if not fold_metrics or last_model is None:
        raise RuntimeError("No valid folds")

    metric_keys = list(fold_metrics[0].keys())
    avg_metrics: Dict[str, float] = {}
    for key in metric_keys:
        values = [_safe_metric_float(m.get(key)) for m in fold_metrics]
        valid_values = [value for value in values if value is not None]
        avg_metrics[key] = float(np.mean(valid_values)) if valid_values else None

    y_pred_full = np.concatenate(preds_collect) if preds_collect else np.array([])
    y_true_full = np.concatenate(truth_collect) if truth_collect else np.array([])
    result = {
        "spec": spec,
        "feature_set_id": feature_set_id,
        "metrics": avg_metrics,
        "primary": float(np.mean(fold_primary)) if fold_primary else 0.0,
        "trading": float(np.mean(fold_trading)) if fold_trading else 0.0,
        "y_pred": y_pred_full,
        "y_true": y_true_full,
        "model": last_model,
        "target_mode": fold.get("target_mode", "raw_logret"),
    }
    if holdout_train_df is not None and holdout_eval_df is not None and not holdout_eval_df.empty:
        holdout = _score_candidate_once(spec, task, feature_cols, holdout_train_df, holdout_eval_df, config)
        result["holdout_metrics"] = holdout["metrics"]
        result["holdout_primary"] = float(holdout["primary"])
        result["holdout_trading"] = float(holdout["trading"])
    return result


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
    if len(candidates) <= max_total:
        return candidates
    tasks = {cand[0].task for cand in candidates}
    bucketed: Dict[Tuple[str, str, str], List[Tuple[ModelSpec, str, List[str]]]] = {}
    for candidate in candidates:
        spec, fs_id, _ = candidate
        family = str(spec.meta.get("family", spec.name))
        task_key = spec.task if len(tasks) > 1 else "_"
        bucketed.setdefault((task_key, family, fs_id), []).append(candidate)

    selected: List[Tuple[ModelSpec, str, List[str]]] = []
    keys = sorted(bucketed.keys())
    while len(selected) < max_total:
        progressed = False
        for key in keys:
            bucket = bucketed[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= max_total:
                break
        if not progressed:
            break
    return selected


def _cap_candidates_by_task_budget(
    candidates: List[Tuple[ModelSpec, str, List[str]]],
    max_total: int,
    seed: int,
) -> List[Tuple[ModelSpec, str, List[str]]]:
    if len(candidates) <= max_total:
        return candidates
    task_order = ["return", "direction", "range"]
    weights = {"return": 0.5, "direction": 0.3, "range": 0.2}
    by_task: Dict[str, List[Tuple[ModelSpec, str, List[str]]]] = {task: [] for task in task_order}
    for candidate in candidates:
        by_task.setdefault(candidate[0].task, []).append(candidate)

    active_tasks = [task for task in task_order if by_task.get(task)]
    if not active_tasks:
        return _cap_candidates(candidates, max_total, seed)

    total_weight = sum(weights[task] for task in active_tasks)
    budgets: Dict[str, int] = {}
    remaining = max_total
    for task in active_tasks:
        raw_budget = int(round(max_total * (weights[task] / total_weight)))
        budget = min(len(by_task[task]), max(1, raw_budget))
        budgets[task] = budget
        remaining -= budget

    while remaining > 0:
        progressed = False
        for task in active_tasks:
            if budgets[task] < len(by_task[task]):
                budgets[task] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    selected: List[Tuple[ModelSpec, str, List[str]]] = []
    for task in active_tasks:
        selected.extend(_cap_candidates(by_task[task], budgets[task], seed))
    return selected[:max_total]


def _activate_served_ensemble(
    registry: Dict[str, Any],
    task: str,
    ensemble_record: Dict[str, Any],
    replaced: bool,
) -> None:
    if replaced:
        registry.setdefault("ensembles", {})[task] = ensemble_record


def _prediction_correlation(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 1.0 if np.allclose(left, right) else 0.0
    corr = np.corrcoef(left, right)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def _select_diverse_ensemble_candidates(task_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 1 or len(task_results) <= 1:
        return task_results[:top_k]
    max_corr = float(os.getenv("ENSEMBLE_MAX_CORR", "0.985"))
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for candidate in task_results:
        preds_raw = candidate.get("y_pred")
        preds = np.asarray(preds_raw if preds_raw is not None else [], dtype=float)
        if not selected:
            selected.append(candidate)
            selected_ids.add(_model_id(candidate["spec"], candidate["feature_set_id"]))
            continue
        too_similar = False
        for existing in selected:
            existing_preds_raw = existing.get("y_pred")
            existing_preds = np.asarray(existing_preds_raw if existing_preds_raw is not None else [], dtype=float)
            corr = abs(_prediction_correlation(preds, existing_preds))
            if corr >= max_corr:
                too_similar = True
                break
        if not too_similar:
            selected.append(candidate)
            selected_ids.add(_model_id(candidate["spec"], candidate["feature_set_id"]))
        if len(selected) >= top_k:
            break
    if not selected:
        return task_results[:top_k]
    if len(selected) < min(top_k, len(task_results)):
        for candidate in task_results:
            model_id = _model_id(candidate["spec"], candidate["feature_set_id"])
            if model_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(model_id)
            if len(selected) >= top_k:
                break
    return selected


def _learn_member_weights(results: List[Dict[str, Any]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    raw_weights: Dict[str, float] = {}
    for result in results:
        metrics = result.get("holdout_metrics") or result.get("metrics") or {}
        error = metrics.get("price_mae")
        if error is None:
            error = metrics.get("return_mae")
        if error is None:
            error = metrics.get("pinball")
        try:
            err_val = max(1e-9, float(error))
        except (TypeError, ValueError):
            err_val = 1.0
        model_id = _model_id(result["spec"], result["feature_set_id"])
        raw_weights[model_id] = 1.0 / err_val
    total = sum(raw_weights.values()) or 1.0
    for model_id, raw in raw_weights.items():
        weights[model_id] = float(raw / total)
    return weights


def _calibration_bucket_summary(timeframe: str, limit: int = 200) -> List[Dict[str, Any]]:
    rows = get_recent_ready_predictions(timeframe, limit)
    buckets: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        predicted = row.get("predicted_return")
        cur = row.get("current_price")
        actual = row.get("actual_price_1h")
        if predicted is None or cur is None or actual is None:
            continue
        try:
            predicted_ret = float(predicted)
            cur_v = float(cur)
            act_v = float(actual)
        except (TypeError, ValueError):
            continue
        if cur_v <= 0 or act_v <= 0:
            continue
        actual_ret = float(np.log(act_v / cur_v))
        regime = str(row.get("regime") or "unknown")
        buckets.setdefault(regime, []).append((predicted_ret, actual_ret))

    out: List[Dict[str, Any]] = []
    for regime, vals in sorted(buckets.items()):
        preds = np.asarray([v[0] for v in vals], dtype=float)
        actuals = np.asarray([v[1] for v in vals], dtype=float)
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2))) if len(vals) else None
        out.append({"regime": regime, "samples": len(vals), "calibration_rmse": rmse})
    return out


def _run_artifact_payload(
    config: TournamentConfig,
    task_results_by_task: Dict[str, Dict[str, Any]],
    *,
    holdout_train_df: Optional[pd.DataFrame],
    holdout_eval_df: Optional[pd.DataFrame],
    run_at: str,
    registry: Dict[str, Any],
    capability_status: Dict[str, Any],
) -> Dict[str, Any]:
    train_start = str(holdout_train_df.index.min()) if holdout_train_df is not None and not holdout_train_df.empty else None
    train_end = str(holdout_train_df.index.max()) if holdout_train_df is not None and not holdout_train_df.empty else None
    holdout_start = str(holdout_eval_df.index.min()) if holdout_eval_df is not None and not holdout_eval_df.empty else None
    holdout_end = str(holdout_eval_df.index.max()) if holdout_eval_df is not None and not holdout_eval_df.empty else None
    served = task_results_by_task.get("return") or {}
    served_artifact = {
        "task": "return",
        "artifact_type": "ensemble" if served.get("ensemble", {}).get("members") else "champion",
        "artifact_id": served.get("served_artifact_id"),
        "holdout_metrics": served.get("best", {}).get("holdout_metrics") or served.get("best", {}).get("metrics"),
        "selection_basis": "holdout",
        "baseline_comparison": served.get("best", {}).get("baseline_comparison"),
        "promotion_state": served.get("promotion_state") or served.get("best", {}).get("promotion_state"),
        "promotion_reason": served.get("decision"),
        "shadow_candidate": served.get("shadow_candidate"),
        "shadow_promotion_report": (registry.get("shadow_promotion_report", {}) or {}).get("return"),
    }
    return {
        "run_at": run_at,
        "timeframe": config.timeframe,
        "candle_minutes": config.candle_minutes,
        "selection_basis": "holdout",
        "train_window": {"start": train_start, "end": train_end},
        "holdout_window": {"start": holdout_start, "end": holdout_end},
        "tasks": task_results_by_task,
        "served_artifact": served_artifact,
        "calibration_buckets": _calibration_bucket_summary(config.timeframe),
        "capability_status": capability_status,
        "promotion_decision_log": registry.get("promotion_decision_log", []),
        "shadow_promotion_report": (registry.get("shadow_promotion_report", {}) or {}).get("return"),
    }


def _save_run_artifact(config: TournamentConfig, payload: Dict[str, Any]) -> None:
    path = config.data_dir / f"run_artifact_{config.candle_minutes}m.json"
    existing = _load_json_file(path)
    baseline_snapshot = existing.get("baseline_accuracy_snapshot") if isinstance(existing, dict) else None
    if not isinstance(baseline_snapshot, dict):
        served_metrics = ((payload.get("served_artifact") or {}).get("holdout_metrics") or {})
        baseline_snapshot = {
            "frozen_at": payload.get("run_at"),
            "comparison_window_id": os.getenv("BASELINE_COMPARISON_WINDOW_ID") or payload.get("run_at"),
            "timeframe": payload.get("timeframe"),
            "holdout_metrics": served_metrics,
        }
    payload["baseline_accuracy_snapshot"] = baseline_snapshot
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def run_tournament(config: TournamentConfig) -> Dict[str, Any]:
    _setup_logging(config.log_path)
    LOGGER.info("Starting tournament")
    run_started_at = datetime.now(timezone.utc)
    _write_run_state(True, run_started_at.isoformat(), None, "running")
    error: Optional[Exception] = None
    progress_total: Optional[int] = None
    progress_done = 0
    progress_failed = 0
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
        min_comp_pct = float(os.getenv("COMPLETENESS_MIN_PCT", "95"))
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
        coverage["dq_errors"] = dq.errors
        coverage["dq_warnings"] = dq.warnings
        coverage["dq_stats"] = dq.stats
        coverage["completeness_pct"] = completeness_pct
        strict_dq = os.getenv("STRICT_DATA_QUALITY", "1").strip().lower() not in {"0", "false", "no", "off"}
        if strict_dq and not dq.ok:
            raise RuntimeError(f"Data quality check failed: {'; '.join(dq.errors)}")
        if dq.warnings:
            LOGGER.warning("Data quality warnings: %s", "; ".join(dq.warnings))

        sup, feature_sets_map = _build_dataset(df, config)
        registry = load_registry(config.registry_path)
        holdout_train_df: Optional[pd.DataFrame] = None
        holdout_eval_df: Optional[pd.DataFrame] = None
        cv_source = sup
        if config.use_test and config.test_hours > 0:
            outer_split = walk_forward_split(sup, config.train_days, config.val_hours, config.test_hours, True)
            if outer_split.test is not None and not outer_split.test.empty:
                holdout_eval_df = outer_split.test.copy()
                holdout_start = holdout_eval_df.index.min()
                holdout_train_df = sup.loc[sup.index < holdout_start].copy()
                if not holdout_train_df.empty:
                    cv_source = holdout_train_df
                else:
                    holdout_eval_df = None
                    holdout_train_df = None

        cv_splits = walk_forward_cv_splits(
            cv_source,
            config.train_days,
            config.val_hours,
            0,
            False,
            folds=config.cv_folds,
        )
        if not cv_splits:
            raise RuntimeError("No valid CV splits")
        split_pairs = [(s.train, s.val) for s in cv_splits]
        val_points_total = int(sum(len(s.val) for s in cv_splits))
        weak_fams = _weak_families(registry)

        results: Dict[str, Any] = {"coverage": coverage}
        run_at = run_started_at.isoformat()

        run_mode = _resolve_run_mode(config)

        per_task_candidates: Dict[str, List[Tuple[ModelSpec, str, List[str]]]] = {}
        per_task_feature_cols: Dict[str, Dict[str, List[str]]] = {}
        strict_horizon_pool = os.getenv("STRICT_HORIZON_MODEL_POOL", "1").strip().lower() not in {"0", "false", "no", "off"}
        for task in ["direction", "return", "range"]:
            specs = get_candidates(
                task,
                config.max_candidates_per_target,
                config.enable_dl,
                candle_minutes=config.candle_minutes,
                strict_horizon_pool=strict_horizon_pool,
            )
            dropped_features = _feature_drop_set(registry, task)
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
            if len(candidates) > config.max_candidates_per_target:
                candidates = _cap_candidates(candidates, config.max_candidates_per_target, config.random_seed)
            per_task_candidates[task] = candidates
            per_task_feature_cols[task] = fs_cols_map

        all_candidates = [c for task in per_task_candidates for c in per_task_candidates[task]]
        if len(all_candidates) > config.max_candidates_total:
            capped = _cap_candidates_by_task_budget(all_candidates, config.max_candidates_total, config.random_seed)
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
        full_fit_source = sup

        for task in ["direction", "return", "range"]:
            candidates = per_task_candidates.get(task, [])
            task_feature_cols = per_task_feature_cols.get(task, {})
            LOGGER.info("Task %s candidates %d", task, len(candidates))
            progress_task = task
            _emit_progress(force=True)

            jobs = [
                (spec, task, fs_id, split_pairs, cols, config, holdout_train_df, holdout_eval_df)
                for spec, fs_id, cols in candidates
            ]
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
                        except Exception as exc:
                            LOGGER.warning("Failed: %s %s", spec.name, exc)
                            progress_failed += 1
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
                selection_primary = float(r.get("holdout_primary", r["primary"]))
                selection_trading = float(r.get("holdout_trading", r["trading"]))
                final = _final_score_for_task(task, selection_primary, selection_trading, stab, config)
                r["final_score"] = final
                r["stability"] = stab
                r["family"] = family
                r["selection_primary"] = selection_primary
                r["selection_trading"] = selection_trading

            task_results.sort(key=lambda x: _result_sort_key(task, x))
            artifacts_dir = config.data_dir / "models" / task
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            top_k = max(1, min(config.ensemble_top_k, len(task_results)))
            selected_for_ensemble = _select_diverse_ensemble_candidates(task_results, top_k) if task == "return" else task_results[:top_k]
            ensemble_weights = _learn_member_weights(selected_for_ensemble) if task == "return" else {}
            ensemble_members: List[Dict[str, Any]] = []
            best_member: Optional[Dict[str, Any]] = None
            best_saved_model: Any = None

            for rank, r in enumerate(selected_for_ensemble, start=1):
                spec = r["spec"]
                fs_id = r["feature_set_id"]
                model_id = _model_id(spec, fs_id)
                model_path = None
                meta_path = None
                model_to_save = r["model"]
                try:
                    model_to_save = _fit_final_model(
                        spec,
                        task,
                        task_feature_cols.get(fs_id, []),
                        full_fit_source,
                        config,
                    )
                    model_path, meta_path = _save_model_artifacts(
                        artifacts_dir,
                        model_id,
                        model_to_save,
                        task,
                        fs_id,
                        task_feature_cols.get(fs_id, []),
                        r.get("holdout_metrics") or r["metrics"],
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
                    "ensemble_weight": ensemble_weights.get(model_id),
                }
                if model_path:
                    ensemble_members.append(member)
                if rank == 1:
                    best_member = member
                    best_saved_model = model_to_save

            stacking_info = None
            if task == "return" and len(ensemble_members) >= 2:
                try:
                    import joblib
                    from sklearn.linear_model import Ridge

                    member_ids = [m["model_id"] for m in ensemble_members]
                    preds_map = {
                        _model_id(r["spec"], r["feature_set_id"]): r["y_pred"]
                        for r in selected_for_ensemble
                    }
                    if all(mid in preds_map for mid in member_ids):
                        X_meta = np.column_stack([preds_map[mid] for mid in member_ids])
                        y_true = selected_for_ensemble[0].get("y_true")
                        if isinstance(y_true, np.ndarray) and len(y_true) == X_meta.shape[0]:
                            meta = Ridge(alpha=1.0, positive=True)
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

            best = task_results[0]
            record_model_score(registry, best["family"], best["final_score"], config.history_keep)
            best_model_id = _model_id(best["spec"], best["feature_set_id"])
            best_fs_id = best["feature_set_id"]
            best_model_path = best_member["model_path"] if best_member else None
            baseline_comparison = _baseline_comparison(task, best, task_results, registry, config)

            challenger = {
                "model_id": best_model_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "final_score": best["final_score"],
                "metrics": best.get("holdout_metrics") or best["metrics"],
                "cv_metrics": best["metrics"],
                "holdout_metrics": best.get("holdout_metrics"),
                "val_points": val_points_total,
                "model_path": best_model_path,
                "feature_cols": task_feature_cols.get(best_fs_id, []),
                "feature_set_id": best_fs_id,
                "family": best["family"],
                "target_mode": best.get("target_mode"),
                "timeframe": config.timeframe,
                "baseline_comparison": baseline_comparison,
                "selection_metric_name": baseline_comparison.get("selection_metric"),
                "selection_metric_value": _result_price_mae(best),
            }
            feature_model = best_saved_model if best_saved_model is not None else best.get("model")
            top_features = _extract_feature_importance(feature_model, task_feature_cols.get(best_fs_id, []), top_k=10)
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
            if stacking_info:
                ensemble_record["stacking"] = stacking_info

            decision = update_champion(
                registry,
                task,
                challenger,
                config.min_val_points,
                config.champion_margin,
                config.champion_margin_override,
                config.champion_cooldown_hours,
                timeframe=config.timeframe,
            )
            _activate_served_ensemble(registry, task, ensemble_record, decision.replaced)

            registry["history"][task].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "best": challenger,
                "decision": decision.reason,
                "promotion_state": decision.promotion_state,
                "baseline_comparison": baseline_comparison,
                "coverage": coverage,
                "run_mode": run_mode,
            })
            registry["history"][task] = registry["history"][task][-config.history_keep:]

            served_artifact_id = None
            current_served_ensemble = registry.get("ensembles", {}).get(task) or {}
            if current_served_ensemble.get("members"):
                served_artifact_id = f"ensemble::{task}::{config.timeframe}"
            else:
                current_champ = registry.get("champions", {}).get(task) or {}
                served_artifact_id = current_champ.get("model_id")

            results[task] = {
                "best": challenger,
                "decision": decision.reason,
                "promotion_state": decision.promotion_state,
                "served_ensemble_updated": decision.replaced,
                "served_artifact_id": served_artifact_id,
                "ensemble": current_served_ensemble,
                "shadow_candidate": (registry.get("shadow", {}) or {}).get(task),
                "top_scores": [
                    {"model_id": _model_id(r["spec"], r["feature_set_id"]), "final_score": r["final_score"]}
                    for r in task_results[:5]
                ],
            }

            for idx, r in enumerate(task_results, start=1):
                metrics_for_board = r.get("holdout_metrics") or r["metrics"]
                metric_name = "accuracy" if task == "direction" else "price_mae" if task == "return" else "pinball"
                if r.get("holdout_metrics") is not None:
                    metric_name = f"holdout_{metric_name}"
                metric_value = (
                    metrics_for_board.get("accuracy")
                    if task == "direction"
                    else metrics_for_board.get("price_mae")
                    if task == "return"
                    else metrics_for_board.get("pinball")
                )
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
                        "trading_score": r.get("selection_trading", r.get("trading")),
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
            artifact_payload = _run_artifact_payload(
                config,
                {k: v for k, v in results.items() if k != "coverage"},
                holdout_train_df=holdout_train_df if holdout_train_df is not None and not holdout_train_df.empty else cv_source,
                holdout_eval_df=holdout_eval_df,
                run_at=run_at,
                registry=registry,
                capability_status=_runtime_capability_status(),
            )
            _save_run_artifact(config, artifact_payload)
        except Exception as exc:
            LOGGER.warning("Failed to save run artifact: %s", exc)
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
                train_days=config.train_days,
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

