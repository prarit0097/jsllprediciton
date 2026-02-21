from pathlib import Path
import json
import os
import atexit
from datetime import datetime, timezone

from .config import TournamentConfig
from .drift import should_retrain_on_drift
from .market_calendar import IST, NSE_OPEN_MIN, NSE_RUN_CLOSE_MIN, is_nse_run_window, load_nse_holidays
from .multi_timeframe import config_for_timeframe, run_multi_timeframe_tournament
from .kite_client import is_kite_enabled, kite_login_url, probe_kite_token_health
from .repair import run_nightly_repair
from .run_weekly_reopt import run_weekly_reoptimization
from .tournament import run_tournament
from .env import load_env

_SCHEDULER_LOCK_PATH = Path(os.getenv("APP_DATA_DIR", "data")) / "run_hourly.lock"


def _pid_alive_windows(pid: int) -> bool:
    try:
        import ctypes
    except Exception:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        err = ctypes.GetLastError()
        # Access denied means the process exists but is not queryable.
        return err == 5
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            # If exit code cannot be read, keep scheduler conservative.
            return True
        return int(exit_code.value) == STILL_ACTIVE
    finally:
        try:
            kernel32.CloseHandle(handle)
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but access is restricted.
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False
    except Exception:
        # Windows can raise SystemError (WinError 87) for stale/invalid PIDs.
        return False


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _run_state_is_stale(data: dict) -> bool:
    if not bool(data.get("running")):
        return False
    try:
        pid = int(data.get("pid", 0))
    except Exception:
        pid = 0
    if pid and not _pid_alive(pid):
        return True
    anchor = _parse_iso_ts(
        (data.get("progress") or {}).get("updated_at")
        or data.get("updated_at")
        or data.get("last_started_at")
    )
    if anchor is None:
        return False
    stale_sec = max(600, int(os.getenv("RUN_STATE_STALE_SECONDS", "10800")))
    age = (datetime.now(timezone.utc) - anchor).total_seconds()
    return age > stale_sec


def _reset_stale_run_state(path: Path, data: dict) -> None:
    try:
        data["running"] = False
        data["status"] = "stale_reset"
        data["last_finished_at"] = datetime.now(timezone.utc).isoformat()
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _acquire_scheduler_lock() -> bool:
    _SCHEDULER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _SCHEDULER_LOCK_PATH.exists():
        try:
            data = json.loads(_SCHEDULER_LOCK_PATH.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except Exception:
            pid = 0
        if pid and _pid_alive(pid):
            print(f"Scheduler already running (pid={pid}); skipping duplicate run_hourly process.")
            return False
        try:
            _SCHEDULER_LOCK_PATH.unlink()
        except Exception:
            pass
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(str(_SCHEDULER_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except FileExistsError:
        print("Scheduler lock exists; another run_hourly process is active.")
        return False
    return True


def _release_scheduler_lock() -> None:
    try:
        if _SCHEDULER_LOCK_PATH.exists():
            data = json.loads(_SCHEDULER_LOCK_PATH.read_text(encoding="utf-8"))
            if int(data.get("pid", 0)) != os.getpid():
                return
            _SCHEDULER_LOCK_PATH.unlink()
    except Exception:
        pass


def _is_running() -> bool:
    state_path = Path("data") / "run_state.json"
    if not state_path.exists():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if _run_state_is_stale(data):
        _reset_stale_run_state(state_path, data)
        return False
    return bool(data.get("running"))


def _cooldown_remaining_seconds(min_gap_seconds: int) -> int:
    if min_gap_seconds <= 0:
        return 0
    state_path = Path("data") / "run_state.json"
    if not state_path.exists():
        return 0
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    anchor = _parse_iso_ts(data.get("last_finished_at")) or _parse_iso_ts(data.get("last_started_at"))
    if anchor is None:
        return 0
    age = (datetime.now(timezone.utc) - anchor).total_seconds()
    remaining = int(max(0.0, float(min_gap_seconds) - age))
    return remaining


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


def _should_run_kite_token_healthcheck(now_utc: datetime, state: dict) -> bool:
    if os.getenv("KITE_TOKEN_HEALTHCHECK_ENABLE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if not is_kite_enabled():
        return False
    source = (os.getenv("PRICE_SOURCE", "auto") or "auto").strip().lower()
    if source not in {"kite", "auto"}:
        return False
    start_min = int(os.getenv("KITE_TOKEN_HEALTHCHECK_START_MIN", "525"))  # 08:45 IST
    end_min = int(os.getenv("KITE_TOKEN_HEALTHCHECK_END_MIN", str(max(start_min, NSE_OPEN_MIN - 1))))  # 09:14 IST
    minute = _minute_of_day_ist(now_utc)
    now_ist = now_utc.astimezone(IST)
    if now_ist.weekday() >= 5:
        return False
    if minute < start_min or minute > end_min:
        return False
    once_per_day = os.getenv("KITE_TOKEN_HEALTHCHECK_ONCE_PER_DAY", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not once_per_day:
        return True
    marker = str(now_ist.date())
    return state.get("kite_token_health_done_for") != marker


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

    if _should_run_kite_token_healthcheck(now_utc, state):
        health = probe_kite_token_health()
        state["kite_token_health_done_for"] = marker
        state["last_kite_token_health_at"] = now_utc.isoformat()
        state["last_kite_token_health_ok"] = bool(health.get("ok"))
        state["last_kite_token_health_error"] = health.get("error")
        state["last_kite_token_health_price"] = health.get("price")
        changed = True
        _save_maintenance_state(config, state)
        if health.get("ok"):
            print(f"Kite token health check OK at {health.get('checked_at')} | LTP: {health.get('price')}")
        else:
            try:
                login = kite_login_url()
            except Exception:
                login = "http://127.0.0.1:8000/kite/login"
            print(f"Kite token check FAILED ({health.get('error')}). Refresh via: {login}")
        return True

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
    if not _acquire_scheduler_lock():
        return
    atexit.register(_release_scheduler_lock)
    try:
        if _is_running():
            print("Tournament already running; skipping.")
            return
        config = TournamentConfig()
        config.base_dir = Path(".")
        force_run = os.getenv("FORCE_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
        default_gap = max(60, int(os.getenv("TOURNAMENT_INTERVAL_MINUTES", "60")) * 60)
        min_gap_seconds = max(0, int(os.getenv("RUN_MIN_GAP_SECONDS", str(default_gap))))
        if not force_run:
            remain = _cooldown_remaining_seconds(min_gap_seconds)
            if remain > 0:
                mm, ss = divmod(remain, 60)
                print(f"Cooldown active; skipping. Next run in {mm:02d}:{ss:02d}. Set FORCE_RUN=1 to override.")
                return
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
    finally:
        _release_scheduler_lock()


if __name__ == "__main__":
    main()
