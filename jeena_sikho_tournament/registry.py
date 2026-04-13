import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ChampionDecision:
    replaced: bool
    reason: str
    promotion_state: str = "shadow"
    baseline_comparison: Optional[Dict[str, Any]] = None


def _default_registry() -> Dict[str, Any]:
    return {
        "champions": {},
        "shadow": {},
        "ensembles": {},
        "history": {"direction": [], "return": [], "range": []},
        "model_history": {},
        "promotion_state": {},
        "shadow_candidates": {},
        "promotion_decision_log": [],
        "shadow_promotion_report": {},
        "drift_monitor": {},
    }


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_registry()
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return _default_registry()
    default = _default_registry()
    for key, value in default.items():
        data.setdefault(key, value)
    return data


def save_registry(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(data, f, indent=2)


def record_model_score(registry: Dict[str, Any], family: str, score: float, keep: int) -> None:
    hist = registry.setdefault("model_history", {}).setdefault(family, [])
    hist.append({"ts": datetime.now(timezone.utc).isoformat(), "score": score})
    registry["model_history"][family] = hist[-keep:]


def stability_penalty(registry: Dict[str, Any], family: str) -> float:
    hist = registry.get("model_history", {}).get(family, [])
    if len(hist) < 3:
        return 0.0
    scores = [h["score"] for h in hist]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return float(var ** 0.5)


def get_champion(registry: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    return registry.get("champions", {}).get(task)


def record_promotion_state(registry: Dict[str, Any], task: str, state: str, details: Optional[Dict[str, Any]] = None) -> None:
    payload = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat()}
    if details:
        payload.update(details)
    registry.setdefault("promotion_state", {})[task] = payload


def get_promotion_state(registry: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    return registry.get("promotion_state", {}).get(task)


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _active_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    metrics = record.get("holdout_metrics") or record.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def _timeframe_defaults(timeframe: Optional[str]) -> Dict[str, float]:
    tf = (timeframe or "").strip().lower()
    if tf == "1d":
        defaults = {
            "max_price_mae": 3.0,
            "max_p90_abs_err": 7.0,
            "min_dir_hit": 56.0,
            "max_bias_rs": 1.5,
            "min_samples": 60.0,
            "min_improvement_ratio": 0.0,
        }
        suffix = "1D"
    elif tf == "2h":
        defaults = {
            "max_price_mae": 1_000_000.0,
            "max_p90_abs_err": 1_000_000.0,
            "min_dir_hit": 0.0,
            "max_bias_rs": 1_000_000.0,
            "min_samples": 80.0,
            "min_improvement_ratio": 0.10,
        }
        suffix = "2H"
    else:
        defaults = {
            "max_price_mae": 1_000_000.0,
            "max_p90_abs_err": 1_000_000.0,
            "min_dir_hit": 0.0,
            "max_bias_rs": 1_000_000.0,
            "min_samples": 80.0,
            "min_improvement_ratio": 0.10,
        }
        suffix = "1H"
    env_map = {
        "max_price_mae": f"PROD_MAX_PRICE_MAE_{suffix}",
        "max_p90_abs_err": f"PROD_MAX_P90_ABS_ERR_{suffix}",
        "min_dir_hit": f"PROD_MIN_DIR_HIT_{suffix}",
        "max_bias_rs": f"PROD_MAX_BIAS_RS_{suffix}",
        "min_samples": f"PROD_MIN_SAMPLES_{suffix}",
        "min_improvement_ratio": f"PROD_MIN_IMPROVEMENT_RATIO_{suffix}",
    }
    resolved: Dict[str, float] = {}
    for key, default in defaults.items():
        env_value = os.getenv(env_map[key])
        parsed = _safe_float(env_value) if env_value is not None else None
        resolved[key] = parsed if parsed is not None else float(default)
    return resolved


def _required_validation_points(timeframe: Optional[str], min_val_points: int) -> int:
    gates = _timeframe_defaults(timeframe)
    min_samples = int(gates.get("min_samples", 0))
    if min_samples <= 0:
        return int(min_val_points)
    return max(1, min(int(min_val_points), min_samples))


def _passes_holdout_gates(record: Dict[str, Any], timeframe: Optional[str]) -> Tuple[bool, str]:
    metrics = _active_metrics(record)
    gates = _timeframe_defaults(timeframe)
    sample_count = _safe_float(metrics.get("sample_count"))
    if sample_count is None or sample_count < gates["min_samples"]:
        return False, "insufficient_holdout_samples"
    price_mae = _safe_float(metrics.get("price_mae"))
    if price_mae is None or price_mae > gates["max_price_mae"]:
        return False, "holdout_price_mae_gate_failed"
    if gates["max_p90_abs_err"] < 999_999.0:
        p90_abs_err = _safe_float(metrics.get("p90_abs_error"))
        if p90_abs_err is None or p90_abs_err > gates["max_p90_abs_err"]:
            return False, "holdout_p90_abs_err_gate_failed"
    if gates["min_dir_hit"] > 0.0:
        direction_hit_rate = _safe_float(metrics.get("direction_hit_rate"))
        if direction_hit_rate is None or direction_hit_rate < gates["min_dir_hit"]:
            return False, "holdout_direction_gate_failed"
    if gates["max_bias_rs"] < 999_999.0:
        signed_bias = _safe_float(metrics.get("signed_bias_rs"))
        if signed_bias is None or abs(signed_bias) > gates["max_bias_rs"]:
            return False, "holdout_bias_gate_failed"
    return True, "holdout_passed"


def _comparison_improvement(reference_mae: Optional[float], challenger_mae: Optional[float]) -> Optional[float]:
    if reference_mae is None or challenger_mae is None or reference_mae <= 0:
        return None
    return float((reference_mae - challenger_mae) / reference_mae)


def _enrich_comparison_entry(entry: Dict[str, Any], challenger_mae: Optional[float]) -> Dict[str, Any]:
    enriched = dict(entry)
    reference_mae = _safe_float(enriched.get("price_mae"))
    if "challenger_price_mae" not in enriched:
        enriched["challenger_price_mae"] = challenger_mae
    if "improvement_ratio" not in enriched:
        enriched["improvement_ratio"] = _comparison_improvement(reference_mae, challenger_mae)
    if "passed" not in enriched:
        enriched["passed"] = (
            challenger_mae < reference_mae
            if reference_mae is not None and challenger_mae is not None
            else None
        )
    return enriched


def _normalize_baseline_comparison(
    challenger: Dict[str, Any],
    champ: Optional[Dict[str, Any]],
    baseline_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(baseline_comparison)
    challenger_mae = _safe_float(_active_metrics(challenger).get("price_mae"))
    naive_cmp = normalized.get("naive_last_close")
    if isinstance(naive_cmp, dict):
        normalized["naive_last_close"] = _enrich_comparison_entry(naive_cmp, challenger_mae)
    incumbent_cmp = normalized.get("incumbent")
    if isinstance(incumbent_cmp, dict):
        normalized["incumbent"] = _enrich_comparison_entry(incumbent_cmp, challenger_mae)
    elif champ:
        normalized["incumbent"] = _enrich_comparison_entry(
            {
                "model_id": champ.get("model_id"),
                "price_mae": _safe_float(_active_metrics(champ).get("price_mae")),
            },
            challenger_mae,
        )
    return normalized


def _set_shadow_candidate(registry: Dict[str, Any], task: str, challenger: Dict[str, Any]) -> None:
    registry.setdefault("shadow", {})[task] = challenger
    registry.setdefault("shadow_candidates", {})[task] = challenger


def _append_promotion_decision_log(
    registry: Dict[str, Any],
    task: str,
    challenger: Dict[str, Any],
    state: str,
    reason: str,
    baseline_comparison: Dict[str, Any],
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "model_id": challenger.get("model_id"),
        "timeframe": challenger.get("timeframe"),
        "promotion_state": state,
        "reason": reason,
        "baseline_comparison": baseline_comparison,
    }
    log = registry.setdefault("promotion_decision_log", [])
    log.append(entry)
    registry["promotion_decision_log"] = log[-500:]


def _record_shadow_promotion_report(
    registry: Dict[str, Any],
    task: str,
    challenger: Dict[str, Any],
    state: str,
    reason: str,
    baseline_comparison: Dict[str, Any],
) -> None:
    report = baseline_comparison.get("shadow_promotion_report")
    if not isinstance(report, dict):
        return
    payload = dict(report)
    payload.update(
        {
            "task": task,
            "model_id": challenger.get("model_id"),
            "timeframe": challenger.get("timeframe"),
            "promotion_state": state,
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    registry.setdefault("shadow_promotion_report", {})[task] = payload


def _set_decision_state(
    registry: Dict[str, Any],
    task: str,
    challenger: Dict[str, Any],
    state: str,
    reason: str,
    baseline_comparison: Dict[str, Any],
) -> ChampionDecision:
    challenger["promotion_state"] = state
    challenger["promotion_reason"] = reason
    challenger["baseline_comparison"] = baseline_comparison
    record_promotion_state(
        registry,
        task,
        state,
        {"reason": reason, "model_id": challenger.get("model_id")},
    )
    _append_promotion_decision_log(registry, task, challenger, state, reason, baseline_comparison)
    _record_shadow_promotion_report(registry, task, challenger, state, reason, baseline_comparison)
    if state == "active":
        registry.setdefault("champions", {})[task] = challenger
        registry.setdefault("shadow", {}).pop(task, None)
        registry.setdefault("shadow_candidates", {}).pop(task, None)
        return ChampionDecision(True, reason, state, baseline_comparison)
    _set_shadow_candidate(registry, task, challenger)
    return ChampionDecision(False, reason, state, baseline_comparison)


def update_champion(
    registry: Dict[str, Any],
    task: str,
    challenger: Dict[str, Any],
    min_val_points: int,
    margin: float,
    margin_override: float,
    cooldown_hours: int,
    timeframe: Optional[str] = None,
) -> ChampionDecision:
    champ = registry.get("champions", {}).get(task)
    challenger["timeframe"] = timeframe or challenger.get("timeframe")
    baseline_comparison = _normalize_baseline_comparison(
        challenger,
        champ,
        challenger.get("baseline_comparison") or {},
    )
    runtime_capability = baseline_comparison.get("runtime_capability") or {}
    if runtime_capability and runtime_capability.get("passed") is False:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "blocked_by_dq",
            "runtime_capability_insufficient",
            baseline_comparison,
        )

    required_val_points = _required_validation_points(challenger.get("timeframe"), min_val_points)
    if challenger.get("val_points", 0) < required_val_points:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "blocked_by_dq",
            f"insufficient_validation_points:{challenger.get('val_points', 0)}<{required_val_points}",
            baseline_comparison,
        )

    holdout_passed, holdout_reason = _passes_holdout_gates(challenger, challenger.get("timeframe"))
    if not holdout_passed:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "blocked_by_holdout",
            holdout_reason,
            baseline_comparison,
        )

    cross_horizon = baseline_comparison.get("cross_horizon") or {}
    if cross_horizon and cross_horizon.get("passed") is False:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "blocked_by_holdout",
            "cross_horizon_regression_blocked",
            baseline_comparison,
        )

    if baseline_comparison.get("shadow_window_passed") is False:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "blocked_by_shadow",
            "shadow_window_gate_failed",
            baseline_comparison,
        )

    naive_cmp = baseline_comparison.get("naive_last_close") or {}
    if naive_cmp and naive_cmp.get("passed") is False:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "shadow",
            "failed_naive_last_close_baseline",
            baseline_comparison,
        )

    if champ is None:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "active",
            "no_champion",
            baseline_comparison,
        )

    incumbent_cmp = baseline_comparison.get("incumbent") or {}
    improvement_ratio = _safe_float(incumbent_cmp.get("improvement_ratio"))
    if improvement_ratio is None:
        improvement_ratio = _comparison_improvement(
            _safe_float(_active_metrics(champ).get("price_mae")),
            _safe_float(_active_metrics(challenger).get("price_mae")),
        )
    required_improvement = _timeframe_defaults(challenger.get("timeframe")).get("min_improvement_ratio", 0.0)
    if improvement_ratio is None or improvement_ratio < required_improvement:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "shadow",
            "holdout_price_mae_not_improved",
            baseline_comparison,
        )

    champ_score = _safe_float(champ.get("final_score")) or 0.0
    ch_score = _safe_float(challenger.get("final_score")) or 0.0
    delta = ch_score - champ_score
    last_ts_raw = champ.get("timestamp")
    last_ts = _parse_ts(last_ts_raw)
    cooldown = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
    locked = last_ts > cooldown
    if locked and delta < margin_override:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "blocked_by_shadow",
            "cooldown_active",
            baseline_comparison,
        )

    if incumbent_cmp or delta >= margin:
        return _set_decision_state(
            registry,
            task,
            challenger,
            "active",
            f"beat_by_{delta:.4f}",
            baseline_comparison,
        )

    return _set_decision_state(
        registry,
        task,
        challenger,
        "shadow",
        "margin_not_met",
        baseline_comparison,
    )


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        ts = _parse_iso(value)
    else:
        ts = datetime.min
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if value.endswith("Z"):
            return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        raise
