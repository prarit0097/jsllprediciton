# JSLL Project Deep Notes

## 1. What This Project Is

This repository is the JSLL prediction web app for Jeena Sikho Lifecare Limited. It combines:

- a Django web layer
- a dashboard-facing service layer
- an NSE-aware tournament/prediction engine
- local SQLite/runtime artifact storage

Its job is not just to show price. It tries to continuously answer:

- current market price kya hai
- different horizons par model kya predict kar raha hai
- kaunsa model champion hai
- prediction kitni confidence ke saath aayi hai
- recent production performance stable hai ya drift ho raha hai
- data complete aur healthy hai ya nahi

## 2. Why It Exists

Plain price dashboards sirf latest quote dikhate hain. Yeh system usse aage jaata hai:

- market data collect karta hai
- features engineer karta hai
- horizon-wise tournaments chalata hai
- best model family choose karta hai
- predictions ko confidence/band ke saath serve karta hai
- prediction vs actual matching maintain karta hai
- drift aur data-quality failure detect karta hai

In short, this is a monitoring + prediction ops app, not a simple quote widget.

## 3. High-Level Architecture

Request path:

1. Browser `/` hit karta hai
2. `btcsite.urls` request ko dashboard view tak route karta hai
3. `jeena_sikho_dashboard.views.dashboard()` template render karta hai
4. Frontend JS periodic API calls bhejta hai
5. `jeena_sikho_dashboard.services` live price, summary, scoreboard, predictions aur match data assemble karta hai
6. Tournament code `jeena_sikho_tournament` se model/state/load logic use hota hai
7. Runtime state SQLite/files/JSON registry se read hota hai

Background flow:

1. Scheduler `jeena_sikho.run_hourly` ya service-level runner invoke karta hai
2. Tournament/data refresh pipeline execute hoti hai
3. Registry, scores, predictions, drift metrics update hote hain
4. Dashboard next poll par naya state dikha deta hai

## 4. Top-Level Folder Map

- `btcsite/`: Django project shell
- `jeena_sikho/`: thin entrypoint wrappers for hourly and repair jobs
- `jeena_sikho_dashboard/`: user-facing app, APIs, runtime DB access, dashboard static/template
- `jeena_sikho_tournament/`: model training, feature engineering, validation, repair, calendar, diagnostics
- `price/`: older/simple price app
- `templates/`: shared root templates
- `static/`: shared static files
- `tests/`: standalone project tests
- `scripts/`: helper launch commands

## 5. Web Routes

Defined via [`btcsite/urls.py`](./btcsite/urls.py) and [`jeena_sikho_dashboard/urls.py`](./jeena_sikho_dashboard/urls.py).

Main routes:

- `/`: main dashboard
- `/price/`: simple legacy page
- `/api/price`: basic price JSON
- `/api/jeena-sikho/price`
- `/api/jeena-sikho/price_at`
- `/api/jeena-sikho/tournament/summary`
- `/api/jeena-sikho/tournament/scoreboard`
- `/api/jeena-sikho/tournament/run`
- `/api/jeena-sikho/tournament/run/status`
- `/api/jeena-sikho/prediction/latest`
- `/api/jeena-sikho/prediction/refresh`

Legacy redirects:

- `/btc/` -> `/`
- `/jeena-sikho/` -> `/`

## 6. Dashboard Layer

Main template:

- [`jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html`](./jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html)

Main frontend JS:

- [`jeena_sikho_dashboard/static/jeena_sikho_dashboard.js`](./jeena_sikho_dashboard/static/jeena_sikho_dashboard.js)

Dashboard shows:

- current price
- update time / stale state
- manual historical price lookup
- latest prediction row or multi-horizon list
- prediction confidence and band
- last matched prediction vs actual
- horizon metrics like MAE, MAPE, hit-rate, utility, calibration RMSE
- tournament summary
- drift state
- horizon champion summary
- completeness summary
- scoreboard with filters

Frontend behavior:

- page load par APIs fetch hoti hain
- polling loop data refresh karta hai
- prediction refresh endpoint ko time-based manner mein trigger kiya ja sakta hai
- UI multi-horizon aur single-primary-prediction dono modes handle karti hai

## 7. Service Layer

Main backend logic file:

- [`jeena_sikho_dashboard/services.py`](./jeena_sikho_dashboard/services.py)

This file handles:

- live quote sourcing
- FX conversion for USD-priced feeds
- NSE market-state interpretation
- tournament summary assembly
- prediction retrieval and refresh
- pending prediction settlement
- scoreboard access
- confidence calculation
- backtest report shaping
- drift/completeness summaries

Notable behaviors:

- price source fallback exists across yfinance/binance/coinbase/kraken
- `.NS` / Indian equity symbols trigger NSE-aware market handling
- live quote response can include `price_inr`, `fx_rate`, `fx_stale`, `price_mode`
- cached stale result fallback exists if fresh fetch fails
- run-state JSON file helps async tournament status
- prediction target timestamps respect NSE interval alignment
- prediction generation now uses a dedicated inference frame built from the latest completed feature row
- predicted price is anchored to that feature-row close instead of mixing a lagged model row with arbitrary current live price
- recent bias and linear calibration can now prefer same-regime history when enough matched samples exist
- prediction responses now expose provenance fields such as forecast anchor time/price, selection basis, and calibration regime
- stale data detection now uses a current watermark check instead of only old-age heuristics

## 8. Tournament / ML Layer

Main package:

- [`jeena_sikho_tournament/`](./jeena_sikho_tournament)

This package is responsible for:

- OHLCV acquisition/stitching
- feature generation
- supervised dataset creation
- candidate/model evaluation
- champion selection
- prediction generation
- diagnostics
- drift-based retrain decisions
- repair/backfill flows
- holdout-aware tournament selection
- full-data refit of selected artifacts before saving
- served-ensemble governance and candidate-budget control

Important modules:

- `features.py`: engineered features and supervised targets
- `predict.py`: prediction helpers
- `tournament.py`: main tournament logic
- `data_sources.py`: source stitching and freshness preference
- `market_calendar.py`: NSE time/holiday/session logic
- `validator.py`: data validation checks
- `repair.py`: missing/gap repair
- `run_hourly.py`: scheduled orchestrator
- `run_repair.py`: repair entry flow
- `doctor.py`: self-check/smoke diagnostics

Accuracy hardening now implemented in this layer:

- optional `test` window is now treated as a real outer holdout for selection instead of being discarded
- target clipping for tournament scoring is learned from training folds, not from the full dataset
- selected top models are refit on the full pre-live dataset before model artifacts are saved
- random global candidate truncation has been replaced by a stratified budget across tasks/families/feature sets
- the served ensemble only updates when champion replacement actually happens
- return-model promotion is now holdout-price-first instead of being dominated by trading-score heuristics
- per-run artifact files now record served artifact id, holdout metrics, train/holdout windows, and calibration bucket summaries
- return ensembles now prefer more diverse members and can use learned non-negative member weights

## 9. Feature Engineering Shape

The tournament package already contains a richer feature pipeline than a basic momentum-only model.

Observed feature families include:

- returns across windows
- rolling means/std/z-scores
- RSI
- ATR-style volatility
- MACD / signal / histogram
- Bollinger width
- VWAP distance
- volume regime features
- trend/range/high-vol/low-vol flags
- gap from previous close
- NSE session-aware time features
- candle body/range/wick structure
- close-location-in-bar context
- distance from session open
- previous-day high/low break flags
- multi-lag candle returns (`ret_2c`, `ret_3c`)
- realized-volatility ratio context (`vol_24` vs `vol_168`)

Targets are also controlled and normalized for more stable training.

Important current nuance:

- `make_supervised()` still builds the generic supervised frame for storage/research convenience
- tournament evaluation now applies fold-local target shaping inside `tournament.py`, which removes the prior full-dataset leakage path
- inference uses `make_inference_frame()` so serving no longer depends on the lagged labeled row
- target scaling is now more horizon-aware than the original single generic scale path

## 10. Prediction Behavior

Predictions are not just one raw number. The app tracks:

- predicted price
- low/high band
- confidence %
- horizon/timeframe
- regime
- later actual matched price
- match %
- error metrics

The service layer also stores and updates pending predictions once target time passes.

Implemented prediction-path improvements:

- latest predictions now use the latest completed feature snapshot and its own anchor close price
- holdout-aware selection metrics now influence champion scoring and scoreboard rows
- regime-aware bias correction and regime-aware linear calibration are applied when enough matched history exists
- context-rich feature sets are now available to the point-forecast path
- return-model search now includes more robust regression families when sklearn is installed
- low-confidence handling now widens uncertainty more than it mutates the point forecast
- missing target-candle settlement no longer falls back to live spot price
- dashboard text can now show forecast anchor time/price and target time without changing the overall layout

### Implemented Accuracy Program

Phase 1:

- disjoint holdout support
- leakage reduction in tournament scoring
- train/serve alignment
- regression tests for split integrity and serving anchor behavior

Phase 2:

- full-data refit before artifact save
- served-ensemble governance
- stratified candidate budgeting
- regression tests around tournament-selection helpers

Phase 3:

- richer context features
- regime-specific bias and calibration
- regression tests for context columns and regime-bucket correction

Phase 4:

- robust return-family expansion in the model zoo
- regression tests for horizon-filter/model-zoo exposure

### Final Accuracy Contract After Today

- all horizons still remain active and first-class
- JSLL-only data scope is preserved
- app still follows an always-show-latest UX
- latest forecast must now be derived from the latest completed qualified bar
- served return artifacts are promoted using holdout price accuracy first
- stale/misaligned data should prevent silent pseudo-accurate forecast generation

## 11. Runtime Storage

There are two storage styles here:

1. Django default DB:
- `db.sqlite3`

2. App/runtime artifacts under configured data dir:
- OHLCV SQLite files
- registry JSON
- log files
- run-state JSON
- saved model artifacts (`data/models/...`)
- optional stacking artifacts for return ensembles
- per-timeframe run artifacts (`run_artifact_{minutes}m.json`)

Dashboard-specific runtime tables are managed by [`jeena_sikho_dashboard/db.py`](./jeena_sikho_dashboard/db.py).

Stored data includes:

- runs
- scores
- predictions
- match metadata
- confidence flags
- regime info
- ensemble/run metadata used for accuracy tracking and selection
- calibration bucket summaries and served-artifact provenance

## 12. Environment and Configuration

Primary reference:

- [`.env.example`](./.env.example)

Important groups:

Branding / routing:

- `APP_BRAND_NAME`
- `APP_MARKET_LABEL`
- `APP_API_PREFIX`
- `APP_DATA_DIR`

Market selection:

- `MARKET_SYMBOL`
- `MARKET_YFINANCE_SYMBOL`
- `PRICE_SOURCE`

Tournament sizing:

- `RUN_MODE`
- `MAX_CANDIDATES_TOTAL`
- `MAX_CANDIDATES_PER_TARGET`
- `MAX_WORKERS`
- `TOURNAMENT_CV_FOLDS`

Prediction quality / production gates:

- `LOW_CONFIDENCE_PCT`
- `LOW_CONFIDENCE_SKIP_PCT`
- `LOW_CONF_DOWNWEIGHT`
- `PRED_BAND_Z`
- `PROD_MAX_MAPE`
- `PROD_MIN_HIT_RATE`
- `USE_TEST`
- `TEST_HOURS`
- `TARGET_WINSOR_LOWER`
- `TARGET_WINSOR_UPPER`
- `EVENT_DAY_DROP_FROM_TRAIN`
- `BIAS_WINDOW`
- `CALIBRATION_MIN_SAMPLES`
- `CALIBRATION_LOOKBACK`
- `ENSEMBLE_MAX_CORR`

Data quality / repair:

- `STRICT_DATA_QUALITY`
- `MAX_MISSING_RATIO`
- `COMPLETENESS_MIN_PCT`
- `AUTO_REPAIR_ON_DQ_FAIL`
- `BACKFILL_GAP_REPAIR`
- `NIGHTLY_REPAIR_AFTER_CLOSE`
- `PREOPEN_REFILL_ENABLE`
- `SCHEDULED_AUTO_REPAIR_ENABLE`

NSE calendar:

- `NSE_HOLIDAYS`
- `NSE_ADHOC_HOLIDAYS`
- `NSE_HOLIDAY_FILE`

## 13. Commands

Local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Hourly/tournament run:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Force run:

```powershell
$env:FORCE_RUN='1'
.\.venv\Scripts\python.exe -m jeena_sikho.run_hourly
```

Repair:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho.run_repair
```

Doctor:

```powershell
.\.venv\Scripts\python.exe -m jeena_sikho_tournament.doctor
```

Django checks:

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py migrate
```

Tests:

```powershell
python -m pytest tests -q
```

Targeted built-in regression suites can also be run without pytest:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_splits.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_leakage.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_feature_quality.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_phase1_alignment.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_phase2_tournament.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_phase3_context_and_calibration.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_phase4_model_zoo.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_phase5_prediction_contract.py"
```

Important note:

- `manage.py test` does not discover the standalone `tests/` suite in this repo
- `pytest` is not listed in `requirements.txt`, so some environments will need `pip install pytest`

## 14. Current VPS Deployment Notes

As provided by the maintainer, the live app currently runs like this:

- code path: `/opt/jsll/app`
- venv path: `/opt/jsll/venv`
- public URL: `https://api.seestox.com/jsll/`
- static URL: `https://api.seestox.com/jsll/static/`
- Gunicorn service: `jsll-gunicorn.service`
- scheduler service: `jsll-scheduler.service`
- socket: `/run/jsll/jsll.sock`

nginx mounts the app under `/jsll/` and strips that prefix before proxying to Gunicorn. That means Django itself still serves root-style paths, but users access the app through `/jsll/`.

Typical deploy/update sequence on VPS:

```bash
cd /opt/jsll/app
source /opt/jsll/venv/bin/activate
git fetch origin
git reset --hard origin/main
pip install -r requirements.txt
python manage.py migrate
python manage.py check
sudo systemctl restart jsll-gunicorn
sudo systemctl restart jsll-scheduler
```

Only use `git reset --hard` after taking a backup if the VPS folder contains local-only edits.

## 15. File-by-File Quick Notes

- [`manage.py`](./manage.py): loads env and boots Django using `btcsite.settings`
- [`btcsite/settings.py`](./btcsite/settings.py): Django config, installed apps, sqlite DB, static dirs
- [`btcsite/urls.py`](./btcsite/urls.py): project URL entrypoint
- [`price/views.py`](./price/views.py): simple/legacy quote page + JSON response
- [`jeena_sikho_dashboard/views.py`](./jeena_sikho_dashboard/views.py): dashboard page and API view functions
- [`jeena_sikho_dashboard/services.py`](./jeena_sikho_dashboard/services.py): core aggregation/business logic, inference alignment, regime-aware bias/calibration
- [`jeena_sikho_dashboard/services.py`](./jeena_sikho_dashboard/services.py): also handles provenance fields, stale-data gating, and prediction contract shaping
- [`jeena_sikho_dashboard/db.py`](./jeena_sikho_dashboard/db.py): runtime SQLite schema and accessors, recent ready-prediction retrieval with regime support
- [`jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html`](./jeena_sikho_dashboard/templates/jeena_sikho_dashboard.html): main dashboard HTML
- [`jeena_sikho_dashboard/static/jeena_sikho_dashboard.js`](./jeena_sikho_dashboard/static/jeena_sikho_dashboard.js): dashboard client logic
- [`jeena_sikho_tournament/features.py`](./jeena_sikho_tournament/features.py): features, supervised-frame generation, inference-frame generation, context features
- [`jeena_sikho_tournament/models_zoo.py`](./jeena_sikho_tournament/models_zoo.py): candidate families including robust return models
- [`jeena_sikho_tournament/market_calendar.py`](./jeena_sikho_tournament/market_calendar.py): canonical NSE slot rules used by alignment and settlement logic
- [`jeena_sikho_tournament/validator.py`](./jeena_sikho_tournament/validator.py): data validation plus freshness/watermark assessment
- [`jeena_sikho_tournament/tournament.py`](./jeena_sikho_tournament/tournament.py): holdout-aware scoring, price-first promotion for return models, full-data refit, candidate budgeting, ensemble governance, run-artifact persistence
- [`jeena_sikho_tournament/run_hourly.py`](./jeena_sikho_tournament/run_hourly.py): periodic orchestration
- [`jeena_sikho_tournament/run_repair.py`](./jeena_sikho_tournament/run_repair.py): repair orchestration
- [`jeena_sikho_tournament/doctor.py`](./jeena_sikho_tournament/doctor.py): health/smoke checks
- [`plans/accuracy-improvement-phases.md`](./plans/accuracy-improvement-phases.md): phased implementation notes for the accuracy program
- [`tests/test_phase1_alignment.py`](./tests/test_phase1_alignment.py): holdout/leakage/serving-alignment regression tests
- [`tests/test_phase2_tournament.py`](./tests/test_phase2_tournament.py): refit/governance/budgeting regression tests
- [`tests/test_phase3_context_and_calibration.py`](./tests/test_phase3_context_and_calibration.py): context-feature and regime-calibration tests
- [`tests/test_phase4_model_zoo.py`](./tests/test_phase4_model_zoo.py): robust model-zoo exposure tests
- [`tests/test_phase5_prediction_contract.py`](./tests/test_phase5_prediction_contract.py): freshness, settlement, and provenance contract tests
- [`scripts/run_hourly_task.cmd`](./scripts/run_hourly_task.cmd): helper launcher
- [`scripts/run_nightly_repair.cmd`](./scripts/run_nightly_repair.cmd): helper launcher

## 16. Gaps / Risks

Current repo risks visible from inspection:

- `pytest` test runner is implied by repo layout but not pinned in `requirements.txt`
- the checked-in local runtime snapshot is still too thin to prove real JSLL accuracy gains end-to-end
- full tournament validation still depends on having populated JSLL data plus the full sklearn/booster dependency set available in the runtime
- `doctor` is still a partial smoke-check surface and not a complete production-certification gate
- `scripts/run_hourly_task.cmd` is still stale and BTC-era path specific
- deployment is path-prefixed in production, so any future hardcoded root-relative frontend links must be checked carefully
- production still needs real matched-prediction history over time before any strong live-accuracy claims should be made

## 17. Bottom Line

This repository is a JSLL-specific market intelligence application with:

- live dashboard UX
- NSE-aware tournament ML pipeline
- prediction confidence/banding
- holdout-aware evaluation and improved train/serve consistency
- full-data model refit and better candidate-selection discipline
- richer context features and regime-aware post-processing
- price-first promotion logic for displayed forecasts
- stale-data and settlement-contract hardening
- drift and data-quality controls
- deployment-ready service structure

It is materially more than a basic Django CRUD or price app. The core value is operational prediction monitoring for one market instrument with scheduled retraining, production-facing summaries, and now a materially stronger accuracy-improvement framework in the tournament and serving path.

## 20. Today's Implementation + Live Deploy Status

Today’s finalized work added and/or confirmed:

- holdout-price-first return-model promotion
- canonical NSE slot alignment for settlement-sensitive paths
- no live-spot fallback for missing historical target-candle settlement
- watermark/freshness-based stale-data detection
- forecast provenance fields in prediction responses
- anchor/target time visibility in dashboard prediction text
- diverse ensemble-member selection plus learned member weighting
- regression coverage for the prediction contract (`test_phase5_prediction_contract.py`)

Live VPS deployment status from today:

- latest pushed branch: `main`
- latest deployed commit on remote push path during today’s session: `3b93b73`
- VPS update steps completed:
  - `git pull origin main`
  - `python manage.py collectstatic --noinput`
  - `python manage.py check`
  - `sudo systemctl restart jsll-gunicorn`
  - `sudo systemctl restart jsll-scheduler`
- observed result:
  - `jsll-gunicorn` running after restart
  - `jsll-scheduler` running after restart
  - scheduler logs showed normal “Outside NSE run window; skipping” behavior when checked outside market window

## 18. Production Incident Notes

During live VPS deployment/debugging, the following production issues were identified and fixed:

1. `DisallowedHost` because settings did not correctly read the VPS env shape.
2. Duplicate `Host` header from nginx because `proxy_params` already set `Host` and the site config added another `proxy_set_header Host $host;`.
3. `DJANGO_*` env aliases were present in `.env`, but settings originally only read plain `DEBUG` / `ALLOWED_HOSTS` style keys.
4. `collectstatic` failed because `STATIC_ROOT` env support was missing.
5. Dashboard JS asset path was hardcoded root-relative and did not respect the `/jsll/` path prefix.
6. Scheduler failed because `timedelta` import was missing in tournament code.
7. Historical `last_ready` dashboard rendering was too strict and could hide legacy matched rows.

## 19. Recommended VPS Runbook

### Safe Deploy / Update

```bash
cd /opt/jsll/app
source /opt/jsll/venv/bin/activate
git fetch origin
git reset --hard origin/main
python manage.py collectstatic --noinput
python manage.py check
sudo systemctl restart jsll-gunicorn
sudo systemctl restart jsll-scheduler
```

### Full Data Refresh

```bash
cd /opt/jsll/app
source /opt/jsll/venv/bin/activate

python -m jeena_sikho_tournament.run_repair

curl -X POST https://api.seestox.com/jsll/api/jeena-sikho/tournament/run \
  -H "Content-Type: application/json" \
  -d '{}'

sleep 10

curl https://api.seestox.com/jsll/api/jeena-sikho/tournament/run/status
curl -X POST https://api.seestox.com/jsll/api/jeena-sikho/prediction/refresh
curl https://api.seestox.com/jsll/api/jeena-sikho/prediction/latest
curl https://api.seestox.com/jsll/api/jeena-sikho/tournament/summary
```

### Health / Logs

```bash
sudo systemctl status jsll-gunicorn --no-pager
sudo systemctl status jsll-scheduler --no-pager
sudo journalctl -u jsll-gunicorn -n 50 --no-pager -l
sudo journalctl -u jsll-scheduler -n 80 --no-pager -l
sudo journalctl -u jsll-scheduler --since "10 minutes ago" --no-pager -l
```

### Browser Refresh

After deploy or static changes:

- use `Ctrl+Shift+R`
- or open a new incognito tab

### Important `.env` Values

```env
APP_BASE_PREFIX=/jsll
APP_API_PREFIX=/jsll/api/jeena-sikho
DJANGO_DEBUG=0
DJANGO_ALLOWED_HOSTS=api.seestox.com,seestox.com,www.seestox.com
DJANGO_CSRF_TRUSTED_ORIGINS=https://api.seestox.com,https://seestox.com,https://www.seestox.com
DJANGO_STATIC_URL=/jsll/static/
DJANGO_STATIC_ROOT=/opt/jsll/app/staticfiles
MAX_MISSING_RATIO=0.08
COMPLETENESS_MIN_PCT=90
AUTO_REPAIR_ON_DQ_FAIL=1
```

### Current Long-Horizon Reality

- `2d` to `7d` rows can still show pending/insufficient-sample states until enough true matched actuals exist
- this is not always a deploy failure
- it can simply mean those horizons do not yet have enough completed outcomes in the runtime DB
