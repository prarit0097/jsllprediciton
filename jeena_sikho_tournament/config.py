from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_FILE = "jeena_sikho_ohlcv.sqlite3"
DEFAULT_REGISTRY_FILE = "jeena_sikho_registry.json"
DEFAULT_LOG_FILE = "jeena_sikho_tournament.log"


@dataclass
class TournamentConfig:
    base_dir: Path = Path(".")
    data_dir: Path = DEFAULT_DATA_DIR
    db_path: Path = DEFAULT_DATA_DIR / DEFAULT_DB_FILE
    registry_path: Path = DEFAULT_DATA_DIR / DEFAULT_REGISTRY_FILE
    log_path: Path = DEFAULT_DATA_DIR / DEFAULT_LOG_FILE

    symbol: str = "BTC/USDT"
    yfinance_symbol: str = "BTC-USD"
    timeframe: str = "10m"
    candle_minutes: int = 10
    ohlcv_table: str = "ohlcv_10m"
    start_date_utc: str = "2015-01-01 00:00:00"
    data_lookback_years: int = 20
    update_every_hours: int = 1

    train_days: int = 180
    val_hours: int = 720
    test_hours: int = 24
    use_test: bool = False

    fee_slippage: float = 0.0008
    min_val_points: int = 500
    champion_margin: float = 0.02
    champion_margin_override: float = 0.05
    champion_cooldown_hours: int = 6

    max_candidates_total: int = 300
    max_candidates_per_target: int = 120
    max_workers: int = 4
    model_timeout_sec: int = 20
    random_seed: int = 42

    history_keep: int = 100
    stability_weight: float = 0.2

    run_mode: str = "hourly"
    enable_dl: bool = False

    ensemble_top_k: int = 3
    bias_window: int = 20
    bias_max_abs: float = 0.01

    feature_windows: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 24, 48, 72, 96, 168])
    cv_folds: int = 5
    close_hit_bps: float = 15.0

    def __post_init__(self) -> None:
        env_data_dir = _env_first("APP_DATA_DIR", "DATA_DIR")
        if env_data_dir:
            self.data_dir = Path(env_data_dir)

        env_db_file = _env_first("APP_MARKET_DB_FILE", "APP_DB_FILE")
        env_registry_file = _env_first("APP_REGISTRY_FILE")
        env_log_file = _env_first("APP_LOG_FILE")

        self.db_path = self.data_dir / (env_db_file or DEFAULT_DB_FILE)
        self.registry_path = self.data_dir / (env_registry_file or DEFAULT_REGISTRY_FILE)
        self.log_path = self.data_dir / (env_log_file or DEFAULT_LOG_FILE)

        env_symbol = _env_first("MARKET_SYMBOL", "ASSET_SYMBOL")
        if env_symbol:
            self.symbol = env_symbol.strip()
        env_yf_symbol = _env_first("MARKET_YFINANCE_SYMBOL")
        if env_yf_symbol:
            self.yfinance_symbol = env_yf_symbol.strip()

        env_timeframes = _env_first("MARKET_TIMEFRAMES", "TIMEFRAMES")
        if env_timeframes:
            tokens = [t.strip() for t in env_timeframes.replace("|", ",").replace(";", ",").split(",") if t.strip()]
            if tokens:
                self.timeframe = tokens[0]
        env_tf = _env_first("MARKET_TIMEFRAME", "TIMEFRAME")
        if env_tf and not env_timeframes:
            self.timeframe = env_tf
        env_cm = _env_first("MARKET_CANDLE_MINUTES", "CANDLE_MINUTES")
        if env_cm and env_cm.isdigit() and not env_timeframes:
            self.candle_minutes = int(env_cm)
        env_table = _env_first("MARKET_OHLCV_TABLE", "OHLCV_TABLE")
        if env_table and not env_timeframes:
            self.ohlcv_table = env_table

        self.candle_minutes = _timeframe_to_minutes(self.timeframe, self.candle_minutes)
        if not env_table or env_timeframes:
            if self.candle_minutes == 60:
                self.ohlcv_table = "ohlcv"
            else:
                self.ohlcv_table = f"ohlcv_{self.candle_minutes}m"

        env_total = os.getenv("MAX_CANDIDATES_TOTAL")
        if env_total and env_total.isdigit():
            self.max_candidates_total = int(env_total)
        env_per = os.getenv("MAX_CANDIDATES_PER_TARGET")
        if env_per and env_per.isdigit():
            self.max_candidates_per_target = int(env_per)
        env_workers = os.getenv("MAX_WORKERS")
        if env_workers and env_workers.isdigit():
            self.max_workers = int(env_workers)
        env_dl = os.getenv("ENABLE_DL")
        if env_dl is not None:
            self.enable_dl = env_dl.strip().lower() in {"1", "true", "yes", "on"}
        env_k = os.getenv("ENSEMBLE_TOP_K")
        if env_k and env_k.isdigit():
            self.ensemble_top_k = max(1, int(env_k))
        env_bias_window = os.getenv("BIAS_WINDOW")
        if env_bias_window and env_bias_window.isdigit():
            self.bias_window = max(1, int(env_bias_window))
        env_bias_max = os.getenv("BIAS_MAX_ABS")
        if env_bias_max:
            try:
                self.bias_max_abs = float(env_bias_max)
            except ValueError:
                pass
        env_windows = os.getenv("FEATURE_WINDOWS")
        if env_windows:
            tokens = [t.strip() for t in env_windows.replace("|", ",").replace(";", ",").split(",") if t.strip()]
            parsed = []
            for token in tokens:
                if token.isdigit():
                    parsed.append(int(token))
            if parsed:
                self.feature_windows = parsed
        env_cv_folds = os.getenv("TOURNAMENT_CV_FOLDS") or os.getenv("CV_FOLDS")
        if env_cv_folds and env_cv_folds.isdigit():
            self.cv_folds = max(1, int(env_cv_folds))
        env_close_hit = os.getenv("RETURN_CLOSE_HIT_BPS")
        if env_close_hit:
            try:
                self.close_hit_bps = max(1.0, float(env_close_hit))
            except ValueError:
                pass
        env_train_days = os.getenv("TRAIN_DAYS")
        if env_train_days and env_train_days.isdigit():
            self.train_days = int(env_train_days)
        env_val_hours = os.getenv("VAL_HOURS")
        if env_val_hours and env_val_hours.isdigit():
            self.val_hours = int(env_val_hours)
        env_test_hours = os.getenv("TEST_HOURS")
        if env_test_hours and env_test_hours.isdigit():
            self.test_hours = int(env_test_hours)
        env_use_test = os.getenv("USE_TEST")
        if env_use_test is not None:
            self.use_test = env_use_test.strip().lower() in {"1", "true", "yes", "on"}
        env_min_val = os.getenv("MIN_VAL_POINTS")
        if env_min_val and env_min_val.isdigit():
            self.min_val_points = max(1, int(env_min_val))


def _env_first(*keys: str) -> str:
    for key in keys:
        val = os.getenv(key)
        if val is not None and val.strip() != "":
            return val.strip()
    return ""


def _timeframe_to_minutes(timeframe: str, fallback: int) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("d") and tf[:-1].isdigit():
        return int(tf[:-1]) * 24 * 60
    return fallback
