import logging
import os
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .config import TournamentConfig, _timeframe_to_minutes
from .tournament import run_tournament

LOGGER = logging.getLogger("jeena_sikho_tournament")

DEFAULT_TIMEFRAMES = ["1h", "2h", "1d"]


def _parse_timeframes(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_TIMEFRAMES)
    tokens: List[str] = []
    for part in value.replace(";", ",").replace("|", ",").split(","):
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


def config_for_timeframe(base: TournamentConfig, timeframe: str) -> TournamentConfig:
    minutes = _timeframe_to_minutes(timeframe, base.candle_minutes)
    cfg = replace(base)
    cfg.timeframe = timeframe
    cfg.candle_minutes = minutes
    if minutes == 60:
        cfg.ohlcv_table = "ohlcv"
    else:
        cfg.ohlcv_table = f"ohlcv_{minutes}m"
    cfg.registry_path = base.data_dir / f"registry_{minutes}m.json"
    cfg.log_path = base.data_dir / f"tournament_{minutes}m.log"
    return cfg


def resolve_timeframes(base: TournamentConfig) -> List[str]:
    env_list = os.getenv("MARKET_TIMEFRAMES") or os.getenv("TIMEFRAMES")
    if env_list:
        return _parse_timeframes(env_list)
    return list(DEFAULT_TIMEFRAMES)


def run_multi_timeframe_tournament(base: TournamentConfig) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    timeframes = resolve_timeframes(base)
    for tf in timeframes:
        cfg = config_for_timeframe(base, tf)
        try:
            LOGGER.info("Running tournament for %s", tf)
            run_tournament(cfg)
            results[tf] = {"status": "ok", "ran_at": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            LOGGER.warning("Tournament failed for %s: %s", tf, exc)
            results[tf] = {"status": "failed", "error": str(exc)}
    return results

