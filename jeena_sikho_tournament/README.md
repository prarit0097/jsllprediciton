# BTC Tournament

Personal Bitcoin model tournament that trains many models on 10-minute candles and selects champions.

## Quick Start

1) Install deps:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Backfill from 2015 (auto on first run):

```
python -m jeena_sikho_tournament.run_hourly
```

Alias (same behavior, clearer name for 10m setup):

```
python -m jeena_sikho_tournament.run_10m
```

If you use a CryptoCompare API key, put it in `.env`:

```
CRYPTOCOMPARE_API_KEY=YOUR_KEY
```

3) View champion status:

Open `data/registry.json`.

4) Run prediction:

```
python -m jeena_sikho_tournament.predict
```

5) Run doctor (self-check + smoke test):

```
python -m jeena_sikho_tournament.doctor
```

## Data backfill and sources

- Data starts at **2015-01-01 00:00:00 UTC** by default.
- Multi-source stitching order: Binance (ccxt) > CryptoCompare (if available) > yfinance.
- For 10m candles, Binance may not support the interval; CryptoCompare histoMinute (aggregate=10) is used.
- The system merges by timestamp, keeps highest-priority source, and reports coverage (earliest, total, missing-interval gaps).

## Config knobs

Edit `jeena_sikho_tournament/config.py`:

- `start_date_utc`: earliest timestamp (default 2015-01-01)
- `timeframe`: candle interval (default `10m`)
- `candle_minutes`: derived from timeframe
- `ohlcv_table`: storage table derived from timeframe (e.g., `ohlcv_10m`)
- `max_candidates_total`: total cap across all tournaments
- `max_candidates_per_target`: per-target cap
- `max_workers`: parallel workers
- `model_timeout_sec`: per-model timeout
- `run_mode`: `hourly|six_hourly|daily|all`
- `enable_dl`: enable heavy MLP candidates (daily only)

## Candidate counts

Default model specs count is ~92. With 5 feature sets, that yields ~460 raw candidates.
Default caps reduce this to:
- `max_candidates_total=240`
- `max_candidates_per_target=100`

To reduce load, lower those caps. To increase, raise caps and install optional boosters:
- `xgboost`
- `lightgbm`
- `catboost`

## Scheduler

You can run hourly with any scheduler (Task Scheduler, cron) using:

```
python -m jeena_sikho_tournament.run_hourly
```

Override run mode via env var:

```
set RUN_MODE=hourly
python -m jeena_sikho_tournament.run_hourly
```

## Doctor checks and common fixes

The doctor command validates structure, dependencies, storage, data coverage, feature leakage guards, model zoo size, a dry-run tournament, registry/champion, and prediction output.

Common fixes:
- Missing deps: `pip install -r requirements.txt`
- Stale data: run `python -m jeena_sikho_tournament.run_hourly` to refresh cache
- Too few candidates: increase `max_candidates_total` / `max_candidates_per_target`

