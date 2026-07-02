from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from app.core.config import settings


class CCXTFetcher:
    def __init__(self, with_keys: bool = False, use_testnet: bool | None = None):
        import ccxt

        exchange_name = settings.EXCHANGE_NAME.lower()
        exchange_cls = getattr(ccxt, exchange_name, ccxt.binance)
        config = {"enableRateLimit": True, "options": {"defaultType": "future"}}
        if with_keys and settings.EXCHANGE_API_KEY and settings.EXCHANGE_SECRET:
            config.update({"apiKey": settings.EXCHANGE_API_KEY, "secret": settings.EXCHANGE_SECRET})
            if settings.EXCHANGE_PASSWORD:
                config["password"] = settings.EXCHANGE_PASSWORD
        self.exchange = exchange_cls(config)
        if use_testnet if use_testnet is not None else settings.EXCHANGE_TESTNET:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass


    def fetch_ticker(self, symbol: str) -> dict:
        ticker = self.exchange.fetch_ticker(symbol)
        last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask") or 0
        return {
            "last": float(last or 0),
            "bid": float(ticker.get("bid") or last or 0),
            "ask": float(ticker.get("ask") or last or 0),
            "timestamp": int(ticker.get("timestamp") or self.exchange.milliseconds()),
        }

    def fetch_ohlcv_list(self, symbol: str, timeframe: str, limit: int = 100):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return self._to_df(rows)

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        end_ms: int,
        limit: int = 1000,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        rows = []
        since = since_ms
        expected = max(1, (end_ms - since_ms) // max(self.exchange.parse_timeframe(timeframe) * 1000, 1))
        while since < end_ms:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            for item in batch:
                if item[0] <= end_ms:
                    rows.append(item)
            last = batch[-1][0]
            next_since = last + self.exchange.parse_timeframe(timeframe) * 1000
            if next_since <= since:
                break
            since = next_since
            if progress_cb:
                progress_cb(len(rows), expected)
            if len(batch) < limit:
                break
        return self._to_df(rows)

    @staticmethod
    def _to_df(rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        return df[["open", "high", "low", "close", "volume"]].astype(float)
