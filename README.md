# Jeena Sikho Prediction Platform (JSLL)

Production-style Django project for NSE-focused price prediction, tournament-based model selection, and horizon-wise monitoring.

## What It Does

- Tracks live JSLL market price (via yfinance)
- Runs multi-timeframe tournaments (`1h`, `2h`, `1d`)
- Trains direction/return/range models and selects champions
- Produces horizon-wise predictions with:
  - confidence score
  - uncertainty band
  - calibration + bias correction
- Computes backtest-style metrics per horizon:
  - MAE, MAPE, hit-rate, directional utility, calibration RMSE
- Supports NSE session logic + holiday-aware scheduling
- Adds drift-aware retrain gating

## Tech Stack

- Python 3.11+
- Django 4.2
- pandas / numpy / scikit-learn
- optional boosters (xgboost / lightgbm / catboost)
- SQLite for OHLCV, runs, scores, and predictions

## Project Structure

- `jeena_sikho_tournament/`: training pipeline, features, data sources, scheduling, drift logic
- `jeena_sikho_dashboard/`: APIs, service layer, dashboard templates/static
- `btcsite/`: Django settings + URL routing
- `data/`: runtime db, registries, logs, model artifacts

## Setup

1. Create and activate virtual environment
2. Install dependencies
3. Configure `.env`
4. Run server

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
```

Open: `http://127.0.0.1:8000`

## Run Tournament Manually

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

To bypass NSE time-window checks:

```powershell
$env:FORCE_RUN='1'
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Nightly gap-repair + validation job:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_repair
```

## Key Environment Variables

- `MARKET_YFINANCE_SYMBOL=JSLL.NS`
- `MARKET_TIMEFRAMES=1h,2h,1d`
- `RUN_MODE=daily`
- `MAX_CANDIDATES_TOTAL`, `MAX_CANDIDATES_PER_TARGET`
- `TOURNAMENT_CV_FOLDS`
- `AUTO_RETRAIN_ON_DRIFT`
- `RETURN_TARGET_MODE=volnorm_logret`
- `STRICT_HORIZON_FEATURE_POOL=1`, `STRICT_HORIZON_MODEL_POOL=1`
- `STRICT_DATA_QUALITY=1`, `MAX_MISSING_RATIO`
- `COMPLETENESS_MIN_PCT`, `COMPLETENESS_LOOKBACK_DAYS`
- `AUTO_REPAIR_ON_DQ_FAIL`, `BACKFILL_GAP_REPAIR`, `BACKFILL_LOOKBACK_DAYS`
- `NSE_ADHOC_HOLIDAYS`, `REPAIR_LOOKBACK_DAYS`
- `LOW_CONFIDENCE_PCT`, `LOW_CONFIDENCE_SKIP_PCT`
- `NSE_HOLIDAYS`, `NSE_HOLIDAY_FILE`

## Notes

- This repo is configured for Jeena Sikho / JSLL workflow.
- Legacy BTC modules/routes have been removed from active flow.
