import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


class Storage:
    def __init__(self, db_path: Path, table_name: str = "ohlcv"):
        self.db_path = db_path
        self.table = table_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _table_columns(self, con: sqlite3.Connection) -> set:
        cur = con.execute(f"PRAGMA table_info({self.table})")
        return {row[1] for row in cur.fetchall()}

    def init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    timestamp_utc TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT
                )
                """
            )
            cols = self._table_columns(con)
            if self.table == "ohlcv":
                if "timestamp" in cols and "timestamp_utc" not in cols:
                    con.execute("ALTER TABLE ohlcv RENAME TO ohlcv_old")
                    con.execute(
                        """
                        CREATE TABLE ohlcv (
                            timestamp_utc TEXT PRIMARY KEY,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume REAL,
                            source TEXT
                        )
                        """
                    )
                    con.execute(
                        """
                        INSERT INTO ohlcv (timestamp_utc, open, high, low, close, volume, source)
                        SELECT timestamp, open, high, low, close, volume, 'legacy' FROM ohlcv_old
                        """
                    )
                    con.execute("DROP TABLE ohlcv_old")
                elif "source" not in cols:
                    con.execute("ALTER TABLE ohlcv ADD COLUMN source TEXT")

    def load(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        with self._connect() as con:
            df = pd.read_sql_query(
                f"SELECT timestamp_utc, open, high, low, close, volume, source FROM {self.table} ORDER BY timestamp_utc",
                con,
                parse_dates=["timestamp_utc"],
            )
        if df.empty:
            return df
        df = df.set_index("timestamp_utc")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def upsert(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.init_db()
        df = df.copy()
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df.sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.isna()]
        rows = [
            (
                idx.isoformat(),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
                str(row.source),
            )
            for idx, row in df.iterrows()
        ]
        with self._connect() as con:
            con.executemany(
                f"""
                INSERT OR REPLACE INTO {self.table} (timestamp_utc, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def trim(self, min_timestamp: pd.Timestamp) -> None:
        if not self.db_path.exists():
            return
        with self._connect() as con:
            con.execute(
                f"DELETE FROM {self.table} WHERE timestamp_utc < ?",
                (min_timestamp.isoformat(),),
            )

    def clean_nans(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._connect() as con:
            cur = con.execute(
                f"""
                DELETE FROM {self.table}
                WHERE timestamp_utc IS NULL
                   OR timestamp_utc = 'NaT'
                   OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL
                """
            )
            return cur.rowcount
