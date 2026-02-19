from pathlib import Path
import os
from typing import Optional


def load_env(path: Optional[Path] = None, overwrite: bool = True) -> None:
    env_path = path or (Path(__file__).resolve().parent.parent / '.env')
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            if (not overwrite) and (key in os.environ):
                continue
            os.environ[key] = val
