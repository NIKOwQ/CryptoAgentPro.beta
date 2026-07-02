from __future__ import annotations

import pandas as pd


def compute_all_indicators(df: pd.DataFrame) -> dict:
    if df.empty:
        return _empty()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    lowest = low.rolling(9).min()
    highest = high.rolling(9).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, pd.NA) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    return {
        "macd": {"dif": _last(dif), "dea": _last(dea), "hist": _last(hist)},
        "rsi": _last(rsi, 50.0),
        "boll": {"upper": _last(upper), "mid": _last(mid), "lower": _last(lower)},
        "kdj": {"k": _last(k, 50.0), "d": _last(d, 50.0), "j": _last(j, 50.0)},
    }


def _last(series: pd.Series, default: float = 0.0) -> float:
    value = series.dropna().iloc[-1] if not series.dropna().empty else default
    try:
        return float(value)
    except Exception:
        return default


def _empty() -> dict:
    return {
        "macd": {"dif": 0.0, "dea": 0.0, "hist": 0.0},
        "rsi": 50.0,
        "boll": {"upper": 0.0, "mid": 0.0, "lower": 0.0},
        "kdj": {"k": 50.0, "d": 50.0, "j": 50.0},
    }
