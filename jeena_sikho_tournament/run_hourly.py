from pathlib import Path
import json
import os
from datetime import datetime, timezone

from .config import TournamentConfig
from .drift import should_retrain_on_drift
from .market_calendar import IST, NSE_OPEN_MIN, NSE_RUN_CLOSE_MIN, is_nse_run_window, load_nse_holidays
from .multi_timeframe import config_for_timeframe, run_multi_timeframe_tournament
from .repair import run_nightly_repair
from .run_weekly_reopt import run_weekly_reoptimization
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


def _maintenance_state_path(config: TournamentConfig) -> Path:
    return config.data_dir / "maintenance_state.json"


def _load_maintenance_state(config: TournamentConfig) -> dict:
    path = _maintenance_state_path(config)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _save_maintenance_state(config: TournamentConfig, data: dict) -> None:
    path = _maintenance_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _minute_of_day_ist(now_utc: datetime) -> int:
    now_ist = now_utc.astimezone(IST)
    return now_ist.hour * 60 + now_ist.minute


def _should_run_preopen_refill(now_utc: datetime, state: dict) -> bool:
    if os.getenv("PREOPEN_REFILL_ENABLE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    start_min = int(os.getenv("PREOPEN_REFILL_START_MIN", "525"))  # 08:45 IST
    end_min = int(os.getenv("PREOPEN_REFILL_END_MIN", str(max(start_min, NSE_OPEN_MIN - 1))))  # default 09:14 IST
    minute = _minute_of_day_ist(now_utc)
    now_ist = now_utc.astimezone(IST)
    if now_ist.weekday() >= 5:
        return False
    if minute < start_min or minute > end_min:
        return False
    once_per_day = os.getenv("PREOPEN_REFILL_ONCE_PER_DAY", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not once_per_day:
        return True
    marker = str(now_ist.date())
    return state.get("preopen_refill_done_for") != marker


def _should_run_nightly_repair(now_utc: datetime, state: dict) -> bool:
    if os.getenv("NIGHTLY_REPAIR_AFTER_CLOSE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    now_ist = now_utc.astimezone(IST)
    if now_ist.weekday() >= 5:
        return False
    minute = _minute_of_day_ist(now_utc)
    if minute <= NSE_RUN_CLOSE_MIN:
        return False
    once_per_day = os.getenv("NIGHTLY_REPAIR_ONCE_PER_DAY", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not once_per_day:
        return True
    marker = str(now_ist.date())
    return state.get("nightly_repair_done_for") != marker


def _should_run_scheduled_repair(now_utc: datetime, state: dict) -> bool:
    if os.getenv("SCHEDULED_AUTO_REPAIR_ENABLE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    minute = _minute_of_day_ist(now_utc)
    # avoid overlap with preopen refill and active run window
    if minute < NSE_OPEN_MIN or minute <= NSE_RUN_CLOSE_MIN:
        return False
    cooldown_min = max(15, int(os.getenv("SCHEDULED_AUTO_REPAIR_COOLDOWN_MIN", "120")))
    last_iso = state.get("last_scheduled_repair_at")
    if last_iso:
        try:
            last = datetime.fromisoformat(last_iso)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_min = (now_utc - last.astimezone(timezone.utc)).total_seconds() / 60.0
            if age_min < cooldown_min:
                return False
        except Exception:
            pass
    return True


def _should_run_weekly_reopt(now_utc: datetime, state: dict) -> bool:
    if os.getenv("WEEKLY_REOPT_ENABLE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    now_ist = now_utc.astimezone(IST)
    run_day = int(os.getenv("WEEKLY_REOPT_DAY", "6"))  # Sunday
    if now_ist.weekday() != run_day:
        return False
    minute = _minute_of_day_ist(now_utc)
    start_min = int(os.getenv("WEEKLY_REOPT_START_MIN", "1080"))  # 18:00 IST
    end_min = int(os.getenv("WEEKLY_REOPT_END_MIN", "1380"))      # 23:00 IST
    if minute < start_min or minute > end_min:
        return False
    week_marker = f"{now_ist.isocalendar().year}-W{now_ist.isocalendar().week:02d}"
    return state.get("weekly_reopt_done_for") != week_marker


def _run_repair_with_profile(config: TournamentConfig, lookback_days: int, label: str) -> None:
    prev = os.getenv("REPAIR_LOOKBACK_DAYS")
    os.environ["REPAIR_LOOKBACK_DAYS"] = str(max(1, lookback_days))
    try:
        report = run_nightly_repair(config)
        print(f"{label} repair completed: {len(report.get('reports', []))} horizons.")
    finally:
        if prev is None:
            os.environ.pop("REPAIR_LOOKBACK_DAYS", None)
        else:
            os.environ["REPAIR_LOOKBACK_DAYS"] = prev


def _handle_maintenance_windows(config: TournamentConfig, now_utc: datetime) -> bool:
    # Returns True when a maintenance action ran and tournament should skip this cycle.
    state = _load_maintenance_state(config)
    now_ist = now_utc.astimezone(IST)
    marker = str(now_ist.date())
    changed = False

    if _should_run_preopen_refill(now_utc, state):
        days = int(os.getenv("PREOPEN_REFILL_LOOKBACK_DAYS", "7"))
        _run_repair_with_profile(config, days, "Pre-open refill")
        state["preopen_refill_done_for"] = marker
        state["last_preopen_refill_at"] = now_utc.isoformat()
        changed = True
        _save_maintenance_state(config, state)
        return True

    if _should_run_nightly_repair(now_utc, state):
        days = int(os.getenv("NIGHTLY_REPAIR_LOOKBACK_DAYS", os.getenv("REPAIR_LOOKBACK_DAYS", "120")))
        _run_repair_with_profile(config, days, "Nightly")
        state["nightly_repair_done_for"] = marker
        state["last_nightly_repair_at"] = now_utc.isoformat()
        changed = True
        _save_maintenance_state(config, state)
        return True

    if _should_run_scheduled_repair(now_utc, state):
        days = int(os.getenv("SCHEDULED_AUTO_REPAIR_LOOKBACK_DAYS", "30"))
        _run_repair_with_profile(config, days, "Scheduled")
        state["last_scheduled_repair_at"] = now_utc.isoformat()
        changed = True
        _save_maintenance_state(config, state)
        return True

    if _should_run_weekly_reopt(now_utc, state):
        report = run_weekly_reoptimization(config)
        state["weekly_reopt_done_for"] = f"{now_ist.isocalendar().year}-W{now_ist.isocalendar().week:02d}"
        state["last_weekly_reopt_at"] = now_utc.isoformat()
        state["last_weekly_reopt_diag_ok"] = bool(report.get("diagnostics_ok"))
        changed = True
        _save_maintenance_state(config, state)
        return True

    if changed:
        _save_maintenance_state(config, state)
    return False


def main():
    load_env()
    if _is_running():
        print("Tournament already running; skipping.")
        return
    config = TournamentConfig()
    config.base_dir = Path(".")
    force_run = os.getenv("FORCE_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
    holidays = load_nse_holidays(config.data_dir)
    now_utc = datetime.now(timezone.utc)

    if _is_indian_equity(config) and not force_run:
        if _handle_maintenance_windows(config, now_utc):
            print("Maintenance window action executed; tournament skipped this cycle.")
            return

    if _is_indian_equity(config) and not force_run:
        if not is_nse_run_window(now_utc, holidays):
            print("Outside NSE run window; skipping. Set FORCE_RUN=1 to override.")
            return
    auto_retrain = os.getenv("AUTO_RETRAIN_ON_DRIFT", "1").strip().lower() in {"1", "true", "yes", "on"}
    if auto_retrain and not force_run:
        should_run, reason = should_retrain_on_drift(config)
        if not should_run:
            print(f"Skipping run: {reason}. Set FORCE_RUN=1 to override.")
            return
        print(f"Drift retrain trigger: {reason}")
    if os.getenv("MARKET_TIMEFRAMES") or os.getenv("TIMEFRAMES"):
        run_multi_timeframe_tournament(config)
    else:
        tf_cfg = config_for_timeframe(config, config.timeframe)
        run_tournament(tf_cfg)


if __name__ == "__main__":
    main()
