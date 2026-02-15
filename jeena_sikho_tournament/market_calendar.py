from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Set

IST = timezone(timedelta(hours=5, minutes=30))
NSE_OPEN_MIN = 9 * 60 + 15
NSE_CLOSE_MIN = 15 * 60 + 30
NSE_RUN_CLOSE_MIN = 16 * 60


def _parse_day(token: str) -> Optional[date]:
    text = (token or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def load_nse_holidays(data_dir: Optional[Path] = None) -> Set[date]:
    holidays: Set[date] = set()
    env_raw = os.getenv("NSE_HOLIDAYS", "")
    if env_raw:
        for token in env_raw.replace(";", ",").split(","):
            day = _parse_day(token)
            if day:
                holidays.add(day)
    adhoc_raw = os.getenv("NSE_ADHOC_HOLIDAYS", "")
    if adhoc_raw:
        for token in adhoc_raw.replace(";", ",").split(","):
            day = _parse_day(token)
            if day:
                holidays.add(day)

    base_dir = data_dir or Path(os.getenv("APP_DATA_DIR", "data"))
    holiday_file = os.getenv("NSE_HOLIDAY_FILE", "").strip()
    path = Path(holiday_file) if holiday_file else (base_dir / "nse_holidays.txt")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            day = _parse_day(clean)
            if day:
                holidays.add(day)
    # Optional yearly official files, e.g. nse_holidays_2026.txt
    for p in sorted(base_dir.glob("nse_holidays_*.txt")):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                clean = line.split("#", 1)[0].strip()
                if not clean:
                    continue
                day = _parse_day(clean)
                if day:
                    holidays.add(day)
        except Exception:
            continue
    return holidays


def load_nse_completeness_exclusions(data_dir: Optional[Path] = None) -> Set[date]:
    out: Set[date] = set()
    raw = os.getenv("NSE_COMPLETENESS_EXCLUDE_DATES", "")
    if raw:
        for token in raw.replace(";", ",").split(","):
            day = _parse_day(token)
            if day:
                out.add(day)
    base_dir = data_dir or Path(os.getenv("APP_DATA_DIR", "data"))
    file_path = os.getenv("NSE_COMPLETENESS_EXCLUDE_FILE", "").strip()
    path = Path(file_path) if file_path else (base_dir / "nse_completeness_exclude.txt")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            day = _parse_day(clean)
            if day:
                out.add(day)
    return out


def is_nse_trading_day(ts_ist: datetime, holidays: Set[date]) -> bool:
    return ts_ist.weekday() < 5 and ts_ist.date() not in holidays


def is_nse_market_open(now_utc: datetime, holidays: Set[date]) -> bool:
    now_ist = now_utc.astimezone(IST)
    if not is_nse_trading_day(now_ist, holidays):
        return False
    minutes = now_ist.hour * 60 + now_ist.minute
    return NSE_OPEN_MIN <= minutes <= NSE_CLOSE_MIN


def is_nse_run_window(now_utc: datetime, holidays: Set[date]) -> bool:
    now_ist = now_utc.astimezone(IST)
    if not is_nse_trading_day(now_ist, holidays):
        return False
    minutes = now_ist.hour * 60 + now_ist.minute
    return NSE_OPEN_MIN <= minutes <= NSE_RUN_CLOSE_MIN


def nse_market_state(now_utc: datetime, holidays: Set[date]) -> dict:
    now_ist = now_utc.astimezone(IST)
    market_open = is_nse_market_open(now_utc, holidays)
    return {
        "exchange": "NSE",
        "exchange_tz": "Asia/Kolkata",
        "market_open": market_open,
        "market_status": "open" if market_open else "closed",
        "trading_window": "Mon-Fri 09:15-15:30 IST",
        "holiday": now_ist.date() in holidays,
    }


def next_nse_trading_day_start(ts_ist: datetime, holidays: Set[date]) -> datetime:
    cur = ts_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    while not is_nse_trading_day(cur, holidays):
        cur = (cur + timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)
    return cur


def align_to_nse_interval_floor(ts_utc: datetime, minutes: int, holidays: Set[date]) -> datetime:
    step = max(1, int(minutes))
    ts_ist = ts_utc.astimezone(IST).replace(second=0, microsecond=0)
    day_open = ts_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    day_close = ts_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if not is_nse_trading_day(ts_ist, holidays):
        prev = (ts_ist - timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
        while not is_nse_trading_day(prev, holidays):
            prev = (prev - timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
        return prev.astimezone(timezone.utc)

    cur_min = ts_ist.hour * 60 + ts_ist.minute
    if cur_min <= NSE_OPEN_MIN:
        return day_open.astimezone(timezone.utc)
    if cur_min >= NSE_CLOSE_MIN:
        return day_close.astimezone(timezone.utc)

    delta = int((ts_ist - day_open).total_seconds() // 60)
    slot = (delta // step) * step
    aligned = day_open + timedelta(minutes=slot)
    if aligned > day_close:
        aligned = day_close
    return aligned.astimezone(timezone.utc)


def next_nse_slot_at_or_after(ts_utc: datetime, minutes: int, holidays: Set[date]) -> datetime:
    step = max(1, int(minutes))
    cur = ts_utc.astimezone(IST).replace(second=0, microsecond=0)
    while True:
        if not is_nse_trading_day(cur, holidays):
            cur = next_nse_trading_day_start(cur + timedelta(days=1), holidays)
            continue
        day_open = cur.replace(hour=9, minute=15, second=0, microsecond=0)
        day_close = cur.replace(hour=15, minute=30, second=0, microsecond=0)
        cur_min = cur.hour * 60 + cur.minute
        if cur_min < NSE_OPEN_MIN:
            cur = day_open
            continue
        if cur_min > NSE_CLOSE_MIN:
            cur = next_nse_trading_day_start(cur + timedelta(days=1), holidays)
            continue
        delta = int((cur - day_open).total_seconds() // 60)
        rem = delta % step
        if rem != 0:
            cur = cur + timedelta(minutes=(step - rem))
            continue
        if cur > day_close:
            cur = next_nse_trading_day_start(cur + timedelta(days=1), holidays)
            continue
        return cur.astimezone(timezone.utc)
