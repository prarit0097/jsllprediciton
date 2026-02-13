from datetime import date, datetime, timedelta, timezone

from jeena_sikho_tournament.market_calendar import (
    IST,
    is_nse_market_open,
    is_nse_run_window,
    next_nse_slot_at_or_after,
)


def test_market_closed_on_holiday():
    holidays = {date(2026, 1, 26)}
    # 2026-01-26 10:00 IST
    ts_utc = datetime(2026, 1, 26, 4, 30, tzinfo=timezone.utc)
    assert is_nse_market_open(ts_utc, holidays) is False
    assert is_nse_run_window(ts_utc, holidays) is False


def test_next_slot_skips_weekend():
    holidays = set()
    # Friday 15:35 IST -> next slot should be Monday 09:15 IST for 60m grid.
    fri_after_close_utc = datetime(2026, 2, 13, 10, 5, tzinfo=timezone.utc)
    nxt = next_nse_slot_at_or_after(fri_after_close_utc, 60, holidays).astimezone(IST)
    assert nxt.weekday() == 0
    assert nxt.hour == 9 and nxt.minute == 15


def test_next_slot_skips_holiday():
    holidays = {date(2026, 2, 16)}
    # Monday holiday, should jump to Tuesday open slot.
    mon_preopen_utc = datetime(2026, 2, 16, 3, 0, tzinfo=timezone.utc)
    nxt = next_nse_slot_at_or_after(mon_preopen_utc, 60, holidays).astimezone(IST)
    assert nxt.date() == date(2026, 2, 17)
    assert nxt.hour == 9 and nxt.minute == 15
