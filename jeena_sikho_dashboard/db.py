import sqlite3
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_db_path() -> Path:
    data_dir = Path(os.getenv("APP_DATA_DIR", "data"))
    db_file = os.getenv("APP_MARKET_DB_FILE") or os.getenv("APP_DB_FILE")
    if db_file:
        return data_dir / db_file
    jeena_default = data_dir / "jeena_sikho_ohlcv.sqlite3"
    legacy = data_dir / "btc_ohlcv.sqlite3"
    if jeena_default.exists() or not legacy.exists():
        return jeena_default
    return legacy


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(get_db_path(), timeout=5)
    try:
        con.execute("PRAGMA busy_timeout = 5000")
        con.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    return con


def ensure_tables() -> None:
    with connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tournament_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT,
                run_started_at TEXT,
                run_finished_at TEXT,
                timeframe TEXT,
                candle_minutes INTEGER,
                run_mode TEXT,
                candidate_count INTEGER,
                duration_seconds REAL,
                max_workers INTEGER,
                train_days INTEGER,
                val_hours INTEGER,
                max_candidates_total INTEGER,
                max_candidates_per_target INTEGER,
                enable_dl INTEGER,
                ensemble_top_k INTEGER
            )
            """
        )
        cols_runs = {row[1] for row in con.execute("PRAGMA table_info(tournament_runs)").fetchall()}
        if "run_started_at" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN run_started_at TEXT")
        if "run_finished_at" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN run_finished_at TEXT")
        if "timeframe" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN timeframe TEXT")
        if "candle_minutes" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN candle_minutes INTEGER")
        if "duration_seconds" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN duration_seconds REAL")
        if "max_workers" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN max_workers INTEGER")
        if "train_days" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN train_days INTEGER")
        if "val_hours" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN val_hours INTEGER")
        if "max_candidates_total" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN max_candidates_total INTEGER")
        if "max_candidates_per_target" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN max_candidates_per_target INTEGER")
        if "enable_dl" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN enable_dl INTEGER")
        if "ensemble_top_k" not in cols_runs:
            con.execute("ALTER TABLE tournament_runs ADD COLUMN ensemble_top_k INTEGER")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tournament_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                rank INTEGER,
                target TEXT,
                feature_set TEXT,
                model_name TEXT,
                family TEXT,
                final_score REAL,
                primary_metric_name TEXT,
                primary_metric_value REAL,
                trading_score REAL,
                stability_penalty REAL,
                is_champion INTEGER,
                run_at TEXT,
                FOREIGN KEY(run_id) REFERENCES tournament_runs(id)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_at TEXT,
                current_price REAL,
                predicted_return REAL,
                predicted_price REAL,
                actual_price_1h REAL,
                match_percent REAL,
                status TEXT,
                model_name TEXT,
                feature_set TEXT,
                run_id INTEGER,
                prediction_target TEXT,
                prediction_horizon_min INTEGER,
                timeframe TEXT,
                timeframe_minutes INTEGER
            )
            """
        )
        cols = {row[1] for row in con.execute("PRAGMA table_info(btc_predictions)").fetchall()}
        if "prediction_target" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN prediction_target TEXT")
        if "prediction_horizon_min" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN prediction_horizon_min INTEGER")
        if "timeframe" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN timeframe TEXT")
        if "timeframe_minutes" not in cols:
            con.execute("ALTER TABLE btc_predictions ADD COLUMN timeframe_minutes INTEGER")


def insert_run(
    run_at: str,
    run_mode: str,
    candidate_count: int,
    run_started_at: Optional[str] = None,
    run_finished_at: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    max_workers: Optional[int] = None,
    timeframe: Optional[str] = None,
    candle_minutes: Optional[int] = None,
    train_days: Optional[int] = None,
    val_hours: Optional[int] = None,
    max_candidates_total: Optional[int] = None,
    max_candidates_per_target: Optional[int] = None,
    enable_dl: Optional[bool] = None,
    ensemble_top_k: Optional[int] = None,
) -> int:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO tournament_runs (
                run_at, run_started_at, run_finished_at, timeframe, candle_minutes, run_mode, candidate_count,
                duration_seconds, max_workers, train_days, val_hours, max_candidates_total, max_candidates_per_target,
                enable_dl, ensemble_top_k
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_at,
                run_started_at,
                run_finished_at,
                timeframe,
                candle_minutes,
                run_mode,
                candidate_count,
                duration_seconds,
                max_workers,
                train_days,
                val_hours,
                max_candidates_total,
                max_candidates_per_target,
                1 if enable_dl else 0 if enable_dl is not None else None,
                ensemble_top_k,
            ),
        )
        return int(cur.lastrowid)


def insert_scores(run_id: int, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_tables()
    values = [
        (
            run_id,
            r.get("rank"),
            r.get("target"),
            r.get("feature_set"),
            r.get("model_name"),
            r.get("family"),
            r.get("final_score"),
            r.get("primary_metric_name"),
            r.get("primary_metric_value"),
            r.get("trading_score"),
            r.get("stability_penalty"),
            1 if r.get("is_champion") else 0,
            r.get("run_at"),
        )
        for r in rows
    ]
    with connect() as con:
        con.executemany(
            """
            INSERT INTO tournament_scores (
                run_id, rank, target, feature_set, model_name, family, final_score,
                primary_metric_name, primary_metric_value, trading_score, stability_penalty,
                is_champion, run_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )


def get_latest_run() -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, run_at, run_started_at, run_finished_at, timeframe, candle_minutes, run_mode, candidate_count,
                   duration_seconds, max_workers, train_days, val_hours, max_candidates_total,
                   max_candidates_per_target, enable_dl, ensemble_top_k
            FROM tournament_runs
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "run_at": row[1],
        "run_started_at": row[2],
        "run_finished_at": row[3],
        "timeframe": row[4],
        "candle_minutes": row[5],
        "run_mode": row[6],
        "candidate_count": row[7],
        "duration_seconds": row[8],
        "max_workers": row[9],
        "train_days": row[10],
        "val_hours": row[11],
        "max_candidates_total": row[12],
        "max_candidates_per_target": row[13],
        "enable_dl": row[14],
        "ensemble_top_k": row[15],
    }


def get_recent_runs(
    limit: int = 20,
    run_mode: Optional[str] = None,
    timeframe: Optional[str] = None,
    candle_minutes: Optional[int] = None,
    train_days: Optional[int] = None,
    val_hours: Optional[int] = None,
    max_workers: Optional[int] = None,
    max_candidates_total: Optional[int] = None,
    max_candidates_per_target: Optional[int] = None,
    enable_dl: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    ensure_tables()
    query = """
        SELECT run_at, run_started_at, run_finished_at, timeframe, candle_minutes, run_mode, candidate_count,
               duration_seconds, max_workers, train_days, val_hours, max_candidates_total, max_candidates_per_target,
               enable_dl, ensemble_top_k
        FROM tournament_runs
    """
    clauses: List[str] = []
    params_list: List[Any] = []
    if run_mode:
        clauses.append("run_mode = ?")
        params_list.append(run_mode)
    if timeframe:
        clauses.append("timeframe = ?")
        params_list.append(timeframe)
    if candle_minutes is not None:
        clauses.append("candle_minutes = ?")
        params_list.append(candle_minutes)
    if train_days is not None:
        clauses.append("train_days = ?")
        params_list.append(train_days)
    if val_hours is not None:
        clauses.append("val_hours = ?")
        params_list.append(val_hours)
    if max_workers is not None:
        clauses.append("max_workers = ?")
        params_list.append(max_workers)
    if max_candidates_total is not None:
        clauses.append("max_candidates_total = ?")
        params_list.append(max_candidates_total)
    if max_candidates_per_target is not None:
        clauses.append("max_candidates_per_target = ?")
        params_list.append(max_candidates_per_target)
    if enable_dl is not None:
        clauses.append("enable_dl = ?")
        params_list.append(1 if enable_dl else 0)
    params: Tuple[Any, ...]
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    params = tuple(params_list + [limit])
    query += " ORDER BY id DESC LIMIT ?"
    with connect() as con:
        cur = con.execute(query, params)
        rows = cur.fetchall()
    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "run_at": r[0],
                "run_started_at": r[1],
                "run_finished_at": r[2],
                "timeframe": r[3],
                "candle_minutes": r[4],
                "run_mode": r[5],
                "candidate_count": r[6],
                "duration_seconds": r[7],
                "max_workers": r[8],
                "train_days": r[9],
                "val_hours": r[10],
                "max_candidates_total": r[11],
                "max_candidates_per_target": r[12],
                "enable_dl": r[13],
                "ensemble_top_k": r[14],
            }
        )
    return results


def get_scores(run_id: int, limit: int = 500) -> List[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT rank, target, feature_set, model_name, family, final_score,
                   primary_metric_name, primary_metric_value, trading_score,
                   stability_penalty, is_champion, run_at
            FROM tournament_scores
            WHERE run_id = ?
            ORDER BY target, rank
            LIMIT ?
            """,
            (run_id, limit),
        )
        rows = cur.fetchall()
    results = []
    for r in rows:
        results.append(
            {
                "rank": r[0],
                "target": r[1],
                "feature_set": r[2],
                "model_name": r[3],
                "family": r[4],
                "final_score": r[5],
                "primary_metric": {"name": r[6], "value": r[7]},
                "trading_score": r[8],
                "stability_penalty": r[9],
                "is_champion": bool(r[10]),
                "run_at": r[11],
            }
        )
    return results


def get_champions(run_id: int) -> Dict[str, Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT target, model_name, feature_set, final_score
            FROM tournament_scores
            WHERE run_id = ? AND is_champion = 1
            """,
            (run_id,),
        )
        rows = cur.fetchall()
    champions: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        target = r[0]
        model_name = r[1]
        feature_set = r[2]
        model_id = model_name
        if feature_set:
            model_id = f"{model_name}__{feature_set}"
        champions[target] = {
            "model_id": model_id,
            "model_name": model_name,
            "feature_set_id": feature_set,
            "final_score": r[3],
        }
    return champions


def get_latest_prediction() -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                   prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
        "timeframe": row[13],
        "timeframe_minutes": row[14],
    }


def get_latest_ready_prediction() -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id, prediction_target, prediction_horizon_min
            FROM btc_predictions
            WHERE status = 'ready'
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
    }


def insert_prediction(row: Dict[str, Any]) -> int:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO btc_predictions (
                predicted_at, current_price, predicted_return, predicted_price,
                actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("predicted_at"),
                row.get("current_price"),
                row.get("predicted_return"),
                row.get("predicted_price"),
                row.get("actual_price_1h"),
                row.get("match_percent"),
                row.get("status"),
                row.get("model_name"),
                row.get("feature_set"),
                row.get("run_id"),
                row.get("prediction_target"),
                row.get("prediction_horizon_min"),
                row.get("timeframe"),
                row.get("timeframe_minutes"),
            ),
        )
        return int(cur.lastrowid)


def update_prediction(pred_id: int, actual_price: float, match_percent: Optional[float], status: str) -> None:
    ensure_tables()
    with connect() as con:
        con.execute(
            """
            UPDATE btc_predictions
            SET actual_price_1h = ?, match_percent = ?, status = ?
            WHERE id = ?
            """,
            (actual_price, match_percent, status, pred_id),
        )


def list_pending_predictions(cutoff_iso: str) -> List[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            WHERE status = 'pending' AND predicted_at <= ?
            ORDER BY predicted_at ASC
            """,
            (cutoff_iso,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "predicted_at": r[1],
            "current_price": r[2],
            "predicted_return": r[3],
            "predicted_price": r[4],
            "prediction_horizon_min": r[5],
            "timeframe": r[6],
            "timeframe_minutes": r[7],
        }
        for r in rows
    ]


def get_latest_prediction_for_timeframe(timeframe: str) -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                   prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            WHERE timeframe = ?
            ORDER BY id DESC LIMIT 1
            """,
            (timeframe,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
        "timeframe": row[13],
        "timeframe_minutes": row[14],
    }


def get_latest_ready_prediction_for_timeframe(timeframe: str) -> Optional[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT id, predicted_at, current_price, predicted_return, predicted_price,
                   actual_price_1h, match_percent, status, model_name, feature_set, run_id,
                   prediction_target, prediction_horizon_min, timeframe, timeframe_minutes
            FROM btc_predictions
            WHERE status = 'ready' AND timeframe = ?
            ORDER BY id DESC LIMIT 1
            """,
            (timeframe,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "predicted_at": row[1],
        "current_price": row[2],
        "predicted_return": row[3],
        "predicted_price": row[4],
        "actual_price_1h": row[5],
        "match_percent": row[6],
        "status": row[7],
        "model_name": row[8],
        "feature_set": row[9],
        "run_id": row[10],
        "prediction_target": row[11],
        "prediction_horizon_min": row[12],
        "timeframe": row[13],
        "timeframe_minutes": row[14],
    }


def get_recent_ready_predictions(timeframe: str, limit: int) -> List[Dict[str, Any]]:
    ensure_tables()
    with connect() as con:
        cur = con.execute(
            """
            SELECT predicted_return, current_price, actual_price_1h, predicted_price, match_percent
            FROM btc_predictions
            WHERE status = 'ready' AND timeframe = ?
            ORDER BY id DESC LIMIT ?
            """,
            (timeframe, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "predicted_return": r[0],
            "current_price": r[1],
            "actual_price_1h": r[2],
            "predicted_price": r[3],
            "match_percent": r[4],
        }
        for r in rows
    ]


def get_ohlcv_close_at(timestamp_iso: str, table: str = "ohlcv") -> Optional[float]:
    try:
        with connect() as con:
            cur = con.execute(
                f"SELECT close FROM {table} WHERE timestamp_utc = ? LIMIT 1",
                (timestamp_iso,),
            )
            row = cur.fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return float(row[0])
