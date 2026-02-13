"""Jeena Sikho market tournament package."""

import importlib
import sys

from .config import TournamentConfig


def _register_legacy_aliases() -> None:
    # Backward compatibility for pre-rename pickled models/modules.
    aliases = {
        "btc_tournament": "jeena_sikho_tournament",
        "btc_dashboard": "jeena_sikho_dashboard",
    }
    for old_pkg, new_pkg in aliases.items():
        try:
            pkg = importlib.import_module(new_pkg)
        except Exception:
            continue
        sys.modules.setdefault(old_pkg, pkg)


_register_legacy_aliases()

__all__ = ["TournamentConfig"]

