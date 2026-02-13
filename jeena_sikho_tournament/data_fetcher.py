import datetime as dt
from typing import Optional

import pandas as pd


class DataFetcher:
    def __init__(self, symbol: str, yfinance_symbol: str, timeframe: str):
        self.symbol = symbol
        self.yfinance_symbol = yfinance_symbol
        self.timeframe = timeframe

    def fetch_ccxt(self, since: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        try:
            import ccxt  # type: ignore
        except Exception:
            return pd.DataFrame()

        exchange = ccxt.binance({"enableRateLimit": True})
        all_rows = []
        since_ms = since
        while True:
            candles = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since_ms, limit=limit)
            if not candles:
                break
            all_rows.extend(candles)
            last_ts = candles[-1][0]
            since_ms = last_ts + 1
            if len(candles) < limit:
                break
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"]).set_index("timestamp")
        return df

    def fetch_yfinance(self, start: Optional[dt.datetime] = None) -> pd.DataFrame:
        try:
            import yfinance as yf  # type: ignore
        except Exception:
            return pd.DataFrame()

        if start is None:
            start = dt.datetime.utcnow() - dt.timedelta(days=730)
        end = dt.datetime.utcnow()
        data = yf.download(
            self.yfinance_symbol,
            start=start,
            end=end,
            interval="1h",
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            return pd.DataFrame()
        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        data.index = pd.to_datetime(data.index, utc=True)
        return data[["open", "high", "low", "close", "volume"]]

    def fetch(self, start: Optional[dt.datetime] = None) -> pd.DataFrame:
        if start is None:
            start = dt.datetime.utcnow() - dt.timedelta(days=730)
        since_ms = int(start.timestamp() * 1000)
        df = self.fetch_ccxt(since=since_ms)
        if df.empty:
            df = self.fetch_yfinance(start=start)
        return df
