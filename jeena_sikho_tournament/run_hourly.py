from pathlib import Path
import json
import os
from datetime import datetime, timezone

from .config import TournamentConfig
from .drift import should_retrain_on_drift
from .market_calendar import is_nse_run_window, load_nse_holidays
from .multi_timeframe import config_for_timeframe, run_multi_timeframe_tournament
from .tournament import run_tournament
from .env import load_env


def _is_running() -> bool:
    state_path = Path("data") / "run_state.json"
    if not state_path.exists():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("running"))


def _is_indian_equity(config: TournamentConfig) -> bool:
    sym = (config.yfinance_symbol or "").strip().upper()
    return sym.endswith(".NS") or sym.endswith(".BO")


def main():
    load_env()
    if _is_running():
        print("Tournament already running; skipping.")
        return
    config = TournamentConfig()
    config.base_dir = Path(".")
    force_run = os.getenv("FORCE_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
    holidays = load_nse_holidays(config.data_dir)
    if _is_indian_equity(config) and not force_run:
        if not is_nse_run_window(datetime.now(timezone.utc), holidays):
            print("Outside NSE run window; skipping. Set FORCE_RUN=1 to override.")
            return
    auto_retrain = os.getenv("AUTO_RETRAIN_ON_DRIFT", "1").strip().lower() in {"1", "true", "yes", "on"}
    if auto_retrain and not force_run:
        should_run, reason = should_retrain_on_drift(config)
        if not should_run:
            print(f"Skipping run: {reason}. Set FORCE_RUN=1 to override.")
            return
        print(f"Drift retrain trigger: {reason}")
    if os.getenv("MARKET_TIMEFRAMES") or os.getenv("TIMEFRAMES") or os.getenv("BTC_TIMEFRAMES"):
        run_multi_timeframe_tournament(config)
    else:
        tf_cfg = config_for_timeframe(config, config.timeframe)
        run_tournament(tf_cfg)


if __name__ == "__main__":
    main()
