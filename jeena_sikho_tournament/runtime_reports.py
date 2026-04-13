from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from jeena_sikho_dashboard.db import get_recent_ready_predictions
from jeena_sikho_dashboard.services import get_timeframes

from .config import TournamentConfig
from .exogenous import event_features_enabled, exogenous_feeds_enabled
from .forecast_metrics import summarize_price_forecast
from .multi_timeframe import config_for_timeframe
from .diagnostics import inspect_runtime_capabilities
from .registry import get_promotion_state, load_registry
from .storage import Storage
from .validator import assess_freshness


def _json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_lineage_summary(config: TournamentConfig) -> Dict[str, Any]:
    storage = Storage(config.db_path, config.ohlcv_table)
    try:
        df = storage.load()
    except Exception:
        return {"available": False, "source_counts": {}, "freshness": None}
    if df.empty:
        return {"available": False, "source_counts": {}, "freshness": None}
    source_counts = {}
    if "source" in df.columns:
        source_counts = {str(k): int(v) for k, v in df["source"].fillna("unknown").value_counts().items()}
    freshness = assess_freshness(
        df,
        config.candle_minutes,
        nse_mode=(config.yfinance_symbol or "").upper().endswith((".NS", ".BO")),
        holidays=set(),
        now_utc=datetime.now(timezone.utc),
    )
    return {
        "available": True,
        "source_counts": source_counts,
        "freshness": freshness,
    }


def build_baseline_accuracy_snapshot(base_config: TournamentConfig) -> Dict[str, Any]:
    capability = inspect_runtime_capabilities()
    snapshot: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability_status": capability["capability_status"],
        "timeframes": {},
    }
    for timeframe in get_timeframes(base_config):
        cfg = config_for_timeframe(base_config, timeframe)
        rows = get_recent_ready_predictions(timeframe, 200)
        metrics = summarize_price_forecast(
            [row.get("actual_price_1h") for row in rows if row.get("actual_price_1h") is not None],
            [row.get("predicted_price") for row in rows if row.get("actual_price_1h") is not None],
            actual_return=[
                None if row.get("current_price") in {None, 0} or row.get("actual_price_1h") in {None, 0}
                else math.log(float(row["actual_price_1h"]) / float(row["current_price"]))
                for row in rows if row.get("actual_price_1h") is not None
            ],
            predicted_return=[row.get("predicted_return") for row in rows if row.get("actual_price_1h") is not None],
            lower_price=[row.get("predicted_price_low") for row in rows if row.get("actual_price_1h") is not None],
            upper_price=[row.get("predicted_price_high") for row in rows if row.get("actual_price_1h") is not None],
        )
        registry = load_registry(cfg.registry_path)
        promotion_state = get_promotion_state(registry, "return")
        run_artifact = _json_read(cfg.data_dir / f"run_artifact_{cfg.candle_minutes}m.json")
        snapshot["timeframes"][timeframe] = {
            "metrics": metrics,
            "sample_count": metrics.get("sample_count", 0),
            "promotion_state": (run_artifact.get("promotion_state") if isinstance(run_artifact, dict) else None) or promotion_state,
            "shadow_status": (run_artifact.get("shadow_status") if isinstance(run_artifact, dict) else None)
            or (run_artifact.get("shadow_promotion_report") or {}).get("status"),
            "source_lineage": _source_lineage_summary(cfg),
        }
    return snapshot


def build_source_lineage_summary(base_config: TournamentConfig) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    for timeframe in get_timeframes(base_config):
        cfg = config_for_timeframe(base_config, timeframe)
        exo_dir = cfg.data_dir / "exogenous"
        exogenous_meta: Dict[str, Any] = {}
        for meta_path in sorted(exo_dir.glob(f"*_{cfg.candle_minutes}m.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            alias = str(meta.get("alias") or meta_path.stem.split("_")[0])
            exogenous_meta[alias] = meta
        source_lineage = _source_lineage_summary(cfg)
        summary["timeframes"][timeframe] = {
            "ohlcv_sources": source_lineage.get("source_counts", {}),
            "data_freshness": source_lineage.get("freshness"),
            "exogenous_signals": exogenous_meta,
        }
    return summary


def build_shadow_promotion_report(base_config: TournamentConfig) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    for timeframe in get_timeframes(base_config):
        cfg = config_for_timeframe(base_config, timeframe)
        registry = load_registry(cfg.registry_path)
        shadow_candidate = ((registry.get("shadow_candidates") or {}).get("return")) if isinstance(registry, dict) else None
        required = _shadow_required(timeframe)
        sample_count = len(get_recent_ready_predictions(timeframe, required))
        report["timeframes"][timeframe] = {
            "shadow_candidate": shadow_candidate,
            "required_settled": required,
            "available_settled": sample_count,
            "status": "ready_for_review" if shadow_candidate and sample_count >= required else "pending_shadow_evidence",
        }
    return report


def build_promotion_decision_log(base_config: TournamentConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "timeframes": {}}
    for timeframe in get_timeframes(base_config):
        cfg = config_for_timeframe(base_config, timeframe)
        registry = load_registry(cfg.registry_path)
        history = ((registry.get("history") or {}).get("return")) if isinstance(registry, dict) else None
        payload["timeframes"][timeframe] = {
            "latest": history[-5:] if history else [],
            "promotion_state": get_promotion_state(registry, "return"),
        }
    return payload


def write_runtime_reports(base_config: TournamentConfig) -> Dict[str, Any]:
    reports = {
        "capability_status": inspect_runtime_capabilities(),
        "baseline_accuracy_snapshot": build_baseline_accuracy_snapshot(base_config),
        "source_lineage_summary": build_source_lineage_summary(base_config),
        "shadow_promotion_report": build_shadow_promotion_report(base_config),
        "promotion_decision_log": build_promotion_decision_log(base_config),
        "matched_history_status": build_matched_history_status(base_config),
        "exogenous_status": {
            "feeds_enabled": exogenous_feeds_enabled(),
            "event_features_enabled": event_features_enabled(),
        },
    }
    for name, payload in reports.items():
        _json_write(base_config.data_dir / f"{name}.json", payload if isinstance(payload, dict) else {"value": payload})
    return reports


def build_matched_history_status(base_config: TournamentConfig) -> Dict[str, Any]:
    thresholds = {"1h": 80, "2h": 80, "1d": 60}
    status: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "timeframes": {}}
    for timeframe in get_timeframes(base_config):
        required = thresholds.get(timeframe, 40)
        count = len(get_recent_ready_predictions(timeframe, required))
        status["timeframes"][timeframe] = {
            "required_settled": required,
            "available_settled": count,
            "sufficient_for_continual_learning": count >= required,
        }
    return status


def load_runtime_report(base_config: TournamentConfig, name: str) -> Dict[str, Any]:
    return _json_read(base_config.data_dir / f"{name}.json")


def _shadow_required(timeframe: str) -> int:
    key = f"SHADOW_PROMOTION_MIN_SETTLED_{timeframe.upper()}"
    try:
        return max(1, int(os.getenv(key, "10")))
    except ValueError:
        return 10


def write_runtime_bootstrap_artifacts(base_config: TournamentConfig, timeframes: List[str] | None = None) -> Dict[str, Any]:
    reports = write_runtime_reports(base_config)
    tf_list = timeframes or get_timeframes(base_config)
    baseline = reports.get("baseline_accuracy_snapshot", {}).get("timeframes", {})
    lineage = reports.get("source_lineage_summary", {}).get("timeframes", {})
    matched = reports.get("matched_history_status", {}).get("timeframes", {})
    shaped = {
        "baseline_accuracy_snapshot": {},
        "source_lineage_summary": {tf: lineage.get(tf, {}) for tf in tf_list},
        "capability_status": reports.get("capability_status"),
        "matched_history_status": reports.get("matched_history_status"),
        "shadow_promotion_report": reports.get("shadow_promotion_report"),
        "promotion_decision_log": reports.get("promotion_decision_log"),
        "exogenous_status": reports.get("exogenous_status"),
    }
    for tf in tf_list:
        base_row = dict(baseline.get(tf, {}))
        match_row = matched.get(tf, {})
        shadow_row = ((reports.get("shadow_promotion_report") or {}).get("timeframes") or {}).get(tf, {})
        source_row = lineage.get(tf, {})
        metrics = base_row.get("metrics") or {}
        base_row["continual_learning_ready"] = bool(match_row.get("sufficient_for_continual_learning"))
        base_row["required_settled"] = match_row.get("required_settled")
        base_row["available_settled"] = match_row.get("available_settled")
        base_row["shadow_status"] = base_row.get("shadow_status") or shadow_row.get("status")
        if base_row.get("promotion_state") is None:
            base_row["promotion_state"] = None
        base_row["frozen_window"] = {
            "sample_count": metrics.get("sample_count", 0),
            "selection_basis": "holdout",
        }
        if source_row:
            base_row["source_lineage"] = source_row
        shaped["baseline_accuracy_snapshot"][tf] = base_row
    return shaped
