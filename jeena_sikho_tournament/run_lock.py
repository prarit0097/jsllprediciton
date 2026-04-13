from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _data_dir(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir or os.getenv("APP_DATA_DIR", "data"))


def lock_path(data_dir: Optional[Path] = None) -> Path:
    return _data_dir(data_dir) / "run.lock"


def _stale_seconds() -> int:
    return max(60, int(os.getenv("RUN_LOCK_STALE_SECONDS", "21600")))


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def read_run_lock(data_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = lock_path(data_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_run_locked(
    data_dir: Optional[Path] = None,
    *,
    stale_seconds: Optional[int] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    payload = read_run_lock(data_dir)
    if not payload:
        return False, None
    acquired_at = _parse_iso(payload.get("acquired_at"))
    ttl = max(60, int(stale_seconds or _stale_seconds()))
    if acquired_at is None or (datetime.now(timezone.utc) - acquired_at) > timedelta(seconds=ttl):
        try:
            lock_path(data_dir).unlink(missing_ok=True)
        except Exception:
            pass
        return False, None
    return True, payload


def acquire_run_lock(
    owner: str,
    *,
    data_dir: Optional[Path] = None,
    stale_seconds: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    path = lock_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    active, payload = is_run_locked(data_dir, stale_seconds=stale_seconds)
    if active and payload is not None:
        return False, payload

    payload = {
        "owner": owner,
        "pid": os.getpid(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload.update(metadata)

    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            return True, payload
        except FileExistsError:
            active, existing = is_run_locked(data_dir, stale_seconds=stale_seconds)
            if active and existing is not None:
                return False, existing
    return False, payload


def release_run_lock(owner: Optional[str] = None, *, data_dir: Optional[Path] = None) -> bool:
    path = lock_path(data_dir)
    if not path.exists():
        return False
    payload = read_run_lock(data_dir)
    if owner and payload and payload.get("owner") not in {None, owner}:
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except Exception:
        return False
