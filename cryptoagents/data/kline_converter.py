from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ConvertedKline:
    F: float
    S: float
    L: float
    H: float
    timestamp: int
    direction: str

    def to_standard_string(self) -> str:
        return f"F={self.F:.6f} S={self.S:.6f} L={self.L:.6f} H={self.H:.6f} {self.direction} @{self.timestamp}"


class KlineConverter:
    def convert_df_rows(self, df: pd.DataFrame, limit: int | None = None) -> list[ConvertedKline]:
        if limit is not None and limit > 0:
            df = df.tail(limit)
        result: list[ConvertedKline] = []
        for idx, row in df.iterrows():
            open_price = float(row["open"])
            close_price = float(row["close"])
            timestamp = int(pd.Timestamp(idx).timestamp()) if not isinstance(idx, (int, float)) else int(idx)
            result.append(ConvertedKline(
                F=open_price,
                S=close_price,
                L=float(row["low"]),
                H=float(row["high"]),
                timestamp=timestamp,
                direction="U" if close_price >= open_price else "D",
            ))
        return result
