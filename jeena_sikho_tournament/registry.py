import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ChampionDecision:
    replaced: bool
    reason: str


def _default_registry() -> Dict[str, Any]:
    return {
        "champions": {},
        "ensembles": {},
        "history": {"direction": [], "return": [], "range": []},
        "model_history": {},
    }


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_registry()
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


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


def update_champion(
    registry: Dict[str, Any],
    task: str,
    challenger: Dict[str, Any],
    min_val_points: int,
    margin: float,
    margin_override: float,
    cooldown_hours: int,
) -> ChampionDecision:
    champ = registry.get("champions", {}).get(task)
    if challenger.get("val_points", 0) < min_val_points:
        return ChampionDecision(False, "insufficient_validation_points")

    if champ is None:
        registry.setdefault("champions", {})[task] = challenger
        return ChampionDecision(True, "no_champion")

    champ_score = champ.get("final_score", 0)
    ch_score = challenger.get("final_score", 0)
    delta = ch_score - champ_score

    last_ts_raw = champ.get("timestamp")
    last_ts = _parse_ts(last_ts_raw)
    cooldown = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
    locked = last_ts > cooldown

    if locked and delta < margin_override:
        return ChampionDecision(False, "cooldown_active")

    if delta >= margin:
        registry["champions"][task] = challenger
        return ChampionDecision(True, f"beat_by_{delta:.4f}")

    return ChampionDecision(False, "margin_not_met")


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
