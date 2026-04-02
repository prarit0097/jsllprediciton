# JSLL Tournament Package

This package contains the training, prediction, data-quality, and repair logic used by the JSLL web app.

## Responsibilities

- fetch and stitch OHLCV data
- align data to NSE market sessions and holidays
- engineer horizon-aware features
- build supervised targets
- run candidate tournaments
- pick champions by task/horizon
- generate predictions and evaluation artifacts
- detect drift and trigger retrain recommendations
- repair/backfill missing data windows

## Main Entry Commands

Run scheduled tournament flow:

```powershell
python -m jeena_sikho_tournament.run_hourly
```

Equivalent wrapper used elsewhere in the repo:

```powershell
python -m jeena_sikho.run_hourly
```

Run repair flow:

```powershell
python -m jeena_sikho_tournament.run_repair
```

Wrapper alias:

```powershell
python -m jeena_sikho.run_repair
```

Run doctor:

```powershell
python -m jeena_sikho_tournament.doctor
```

## Main Modules

- `config.py`: configuration model and env-driven knobs
- `data_sources.py`: source stitching and priority handling
- `features.py`: feature engineering and supervised-frame creation
- `predict.py`: prediction helpers
- `tournament.py`: candidate evaluation and champion selection
- `registry.py`: registry/champion structures
- `validator.py`: leakage/data-quality validation helpers
- `repair.py`: repair/backfill logic
- `market_calendar.py`: NSE session calendar logic
- `run_hourly.py`: orchestration entrypoint
- `run_repair.py`: repair orchestration entrypoint
- `doctor.py`: diagnostics/self-check

## Notes

- The package is now JSLL/NSE-oriented, not the older BTC-focused setup.
- Runtime behavior depends heavily on `.env` values such as `MARKET_TIMEFRAMES`, `RUN_MODE`, `STRICT_DATA_QUALITY`, and holiday settings.
- Production dashboard consumers read the outputs of this package through `jeena_sikho_dashboard.services`.
