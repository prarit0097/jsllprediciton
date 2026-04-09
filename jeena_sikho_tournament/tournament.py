import json
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
from .metrics import accuracy, f1_score, mae, pinball_loss, coverage
from .models_zoo import ModelSpec, build_quantile_bundle, get_candidates
from .registry import (
    load_registry,
    record_model_score,
    save_registry,
    stability_penalty,
    update_champion,
)
from .splits import walk_forward_cv_splits, walk_forward_split
from .storage import Storage
from .validator import validate_ohlcv_quality
from .market_calendar import load_nse_holidays
from .repair import repair_timeframe_data
from jeena_sikho_dashboard.db import insert_run, insert_scores

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
        mae_val = mae(y_eval, y_pred)
        dir_acc = accuracy((y_eval > 0).astype(int), (y_pred > 0).astype(int))
        close_hit = _close_hit_rate(y_eval, y_pred, config.close_hit_bps)
        positions = (y_pred > 0).astype(int)
        net, mdd, trading = trading_score(positions, y_eval, config.fee_slippage)
        primary = _primary_score_reg_weighted(mae_val, y_eval, dir_acc, close_hit)
        return {
            "metrics": {"mae": mae_val, "dir_acc": dir_acc, "close_hit": close_hit, "net": net, "mdd": mdd},
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
        avg_metrics[key] = float(np.mean([m[key] for m in fold_metrics]))

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
                final = _final_score(selection_trading, selection_primary, stab, config)
                r["final_score"] = final
                r["stability"] = stab
                r["family"] = family
                r["selection_primary"] = selection_primary
                r["selection_trading"] = selection_trading

            task_results.sort(key=lambda x: x["final_score"], reverse=True)
            artifacts_dir = config.data_dir / "models" / task
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            top_k = max(1, min(config.ensemble_top_k, len(task_results)))
            ensemble_members: List[Dict[str, Any]] = []
            best_member: Optional[Dict[str, Any]] = None
            best_saved_model: Any = None

            for rank, r in enumerate(task_results[:top_k], start=1):
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

            best = task_results[0]
            record_model_score(registry, best["family"], best["final_score"], config.history_keep)
            best_model_id = _model_id(best["spec"], best["feature_set_id"])
            best_fs_id = best["feature_set_id"]
            best_model_path = best_member["model_path"] if best_member else None

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
            )
            _activate_served_ensemble(registry, task, ensemble_record, decision.replaced)

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
                "served_ensemble_updated": decision.replaced,
                "top_scores": [
                    {"model_id": _model_id(r["spec"], r["feature_set_id"]), "final_score": r["final_score"]}
                    for r in task_results[:5]
                ],
            }

            for idx, r in enumerate(task_results, start=1):
                metrics_for_board = r.get("holdout_metrics") or r["metrics"]
                metric_name = "accuracy" if task == "direction" else "weighted_obj" if task == "return" else "pinball"
                if r.get("holdout_metrics") is not None:
                    metric_name = f"holdout_{metric_name}"
                metric_value = (
                    metrics_for_board.get("accuracy")
                    if task == "direction"
                    else r.get("selection_primary")
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

