from __future__ import annotations

import hashlib
import hmac
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger
from cryptoagents.exchange.base import ExchangeBase

logger = get_logger("binance")


class BinanceExchange(ExchangeBase):
    name = "binance"
    base_url = "https://fapi.binance.com"

    def __init__(self, api_key: str = "", secret: str = "", passphrase: str = "",
                 testnet: bool = False, with_keys: bool = True):
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.with_keys = with_keys and bool(api_key)
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"

    def _symbol_to_binance(self, symbol: str) -> str:
        """BTC/USDT -> BTCUSDT"""
        return symbol.replace("/", "").replace("-", "")

    def _sign(self, params: dict) -> str:
        qs = urlencode(params)
        return hmac.new(self.secret.encode(), qs.encode(), hashlib.sha256).hexdigest()

    def _headers(self) -> dict[str, str]:
        h = {}
        if self.with_keys:
            h["X-MBX-APIKEY"] = self.api_key
        return h

    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False) -> dict:
        import httpx
        params = dict(params or {})
        if signed and self.with_keys:
            params["timestamp"] = str(int(time.time() * 1000))
            params["signature"] = self._sign(params)
        qs = urlencode(params)
        url = f"{self.base_url}{path}?{qs}"
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with httpx.Client(timeout=15, verify=ctx) as c:
                if method == "GET":
                    r = c.get(url, headers=self._headers())
                elif method == "DELETE":
                    r = c.delete(url, headers=self._headers())
                else:
                    r = c.post(url, headers=self._headers())
            if r.status_code != 200:
                logger.warning(f"Binance API {r.status_code}: {r.text[:200]}")
                return {"error": r.text[:500]}
            return r.json()
        except Exception as exc:
            logger.error(f"Binance request failed {path}: {exc}")
            return {"error": str(exc)}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100,
                    since: int | None = None) -> pd.DataFrame:
        sym = self._symbol_to_binance(symbol)
        params: dict[str, Any] = {"symbol": sym, "interval": timeframe, "limit": str(min(limit, 1500))}
        if since:
            params["startTime"] = str(since)
        data = self._request("GET", "/fapi/v1/klines", params)
        if not isinstance(data, list):
            return pd.DataFrame()
        records = []
        for r in data:
            records.append({
                "timestamp": pd.to_datetime(int(r[0]), unit="ms"),
                "open": float(r[1]), "high": float(r[2]),
                "low": float(r[3]), "close": float(r[4]),
                "volume": float(r[5]),
            })
        df = pd.DataFrame(records).set_index("timestamp")
        return df

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        sym = self._symbol_to_binance(symbol)
        data = self._request("GET", "/fapi/v1/ticker/price", {"symbol": sym})
        if "price" not in data:
            return {"last": 0, "bid": 0, "ask": 0, "timestamp": 0}
        return {"last": float(data["price"]), "bid": float(data["price"]),
                "ask": float(data["price"]), "timestamp": int(time.time() * 1000)}

    def fetch_positions(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        if not self.with_keys:
            return []
        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        if not isinstance(data, list):
            return []
        out = []
        for r in data:
            amt = float(r.get("positionAmt", 0))
            if amt == 0:
                continue
            sym = r.get("symbol", "")
            # BTCUSDT -> BTC/USDT
            if sym.endswith("USDT"):
                base = sym[:-4]
                symbol = f"{base}/USDT"
            else:
                symbol = sym
            if symbols and symbol not in symbols and sym not in symbols:
                continue
            out.append({
                "symbol": symbol, "direction": "LONG" if amt > 0 else "SHORT",
                "qty": abs(amt), "entry_price": float(r.get("entryPrice", 0)),
                "leverage": int(float(r.get("leverage", 1))),
                "contracts": abs(amt),
            })
        return out

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if not self.with_keys:
            return
        sym = self._symbol_to_binance(symbol)
        self._request("POST", "/fapi/v1/leverage",
                      {"symbol": sym, "leverage": str(leverage)}, signed=True)

    def _filters(self, symbol: str) -> dict[str, Decimal]:
        sym = self._symbol_to_binance(symbol)
        info = self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": sym})
        result = {"step": Decimal("0.001"), "tick": Decimal("0.01")}
        if isinstance(info, dict):
            for item in info.get("symbols", []):
                if item.get("symbol") != sym:
                    continue
                for flt in item.get("filters", []):
                    if flt.get("filterType") == "LOT_SIZE":
                        result["step"] = Decimal(str(flt.get("stepSize", "0.001")))
                    elif flt.get("filterType") == "PRICE_FILTER":
                        result["tick"] = Decimal(str(flt.get("tickSize", "0.01")))
        return result

    def _format_quantity(self, symbol: str, qty: float) -> str:
        step = self._filters(symbol)["step"]
        q = Decimal(str(qty)).quantize(step, rounding=ROUND_DOWN)
        if q <= 0:
            q = step
        return format(q.normalize(), "f")

    def _format_price(self, symbol: str, price: float) -> str:
        tick = self._filters(symbol)["tick"]
        p = Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN)
        return format(p.normalize(), "f")

    def cancel_all_open_orders(self, symbol: str) -> dict[str, Any]:
        if not self.with_keys:
            return {"status": "error", "message": "no API key"}
        sym = self._symbol_to_binance(symbol)
        normal = self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": sym}, signed=True)
        algo = self._request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": sym}, signed=True)
        errors = []
        for item in (normal, algo):
            if isinstance(item, dict) and "error" in item:
                errors.append(str(item["error"])[:160])
        if errors:
            return {"status": "error", "message": " | ".join(errors)}
        return {"status": "ok", "normal": normal, "algo": algo}

    def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        if not self.with_keys:
            return []
        sym = self._symbol_to_binance(symbol)
        data = self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": sym}, signed=True)
        return data if isinstance(data, list) else []

    def place_market_order(self, symbol: str, side: str, qty: float,
                           reduce_only: bool = False, sl_price: float = 0,
                           tp_price: float = 0, leverage: int = 0) -> dict[str, Any]:
        if not self.with_keys:
            return {"id": "", "status": "error", "message": "no API key"}
        sym = self._symbol_to_binance(symbol)
        bn_side = "BUY" if side.lower() in ("buy", "long") else "SELL"
        qty_str = self._format_quantity(symbol, qty)
        params: dict[str, Any] = {
            "symbol": sym, "side": bn_side, "type": "MARKET",
            "quantity": qty_str,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        data = self._request("POST", "/fapi/v1/order", params, signed=True)
        if "orderId" not in data:
            return {"id": "", "status": "error", "message": str(data.get("error", data))[:200]}

        if reduce_only:
            return {"id": str(data["orderId"]), "status": "ok", "message": "order placed"}

        protection_errors: list[str] = []
        close_side = "SELL" if bn_side == "BUY" else "BUY"
        if sl_price:
            sl = self._request("POST", "/fapi/v1/algoOrder", {
                "symbol": sym, "side": close_side,
                "algoType": "CONDITIONAL",
                "type": "STOP_MARKET",
                "triggerPrice": self._format_price(symbol, sl_price),
                "quantity": qty_str,
                "reduceOnly": "true",
                "workingType": "MARK_PRICE",
            }, signed=True)
            if "error" in sl:
                protection_errors.append(str(sl["error"])[:180])
        if tp_price:
            tp = self._request("POST", "/fapi/v1/algoOrder", {
                "symbol": sym, "side": close_side,
                "algoType": "CONDITIONAL",
                "type": "TAKE_PROFIT_MARKET",
                "triggerPrice": self._format_price(symbol, tp_price),
                "quantity": qty_str,
                "reduceOnly": "true",
                "workingType": "MARK_PRICE",
            }, signed=True)
            if "error" in tp:
                protection_errors.append(str(tp["error"])[:180])
        if protection_errors:
            self.cancel_all_open_orders(symbol)
            close = self._request("POST", "/fapi/v1/order", {
                "symbol": sym, "side": close_side, "type": "MARKET",
                "quantity": qty_str, "reduceOnly": "true",
            }, signed=True)
            close_msg = "closed" if "orderId" in close else str(close.get("error", close))[:120]
            return {"id": str(data["orderId"]), "status": "error",
                    "message": "protection order failed; entry " + close_msg + ": " + " | ".join(protection_errors)[:220]}
        return {"id": str(data["orderId"]), "status": "ok", "message": "order placed"}

    def fetch_account_balance(self) -> dict[str, Any]:
        if not self.with_keys:
            return {"total": 0, "available": 0, "wallet_balance": 0, "unrealized_pnl": 0}
        account = self._request("GET", "/fapi/v2/account", signed=True)
        if isinstance(account, dict) and "error" not in account:
            return {
                "total": float(account.get("totalMarginBalance", 0) or 0),
                "available": float(account.get("availableBalance", 0) or 0),
                "wallet_balance": float(account.get("totalWalletBalance", 0) or 0),
                "unrealized_pnl": float(account.get("totalUnrealizedProfit", 0) or 0),
                "initial_margin": float(account.get("totalInitialMargin", 0) or 0),
                "maint_margin": float(account.get("totalMaintMargin", 0) or 0),
            }
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        if not isinstance(data, list):
            return {"total": 0, "available": 0, "wallet_balance": 0, "unrealized_pnl": 0}
        for b in data:
            if b.get("asset") == "USDT":
                total = float(b.get("balance", 0) or 0)
                return {"total": total,
                        "available": float(b.get("availableBalance", 0) or 0),
                        "wallet_balance": total,
                        "unrealized_pnl": 0}
        return {"total": 0, "available": 0, "wallet_balance": 0, "unrealized_pnl": 0}

    def fetch_income_history(self, days: int = 7, limit: int = 1000) -> list[dict[str, Any]]:
        """Fetch recent futures income rows used for net realized PnL reporting."""
        if not self.with_keys:
            return []
        start = int((time.time() - days * 86400) * 1000)
        data = self._request("GET", "/fapi/v1/income", {
            "startTime": str(start),
            "limit": str(min(max(limit, 1), 1000)),
        }, signed=True)
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for row in data:
            sym = row.get("symbol", "")
            if sym.endswith("USDT"):
                symbol = f"{sym[:-4]}/USDT"
            else:
                symbol = sym
            income_type = row.get("incomeType", "")
            income = float(row.get("income", 0) or 0)
            out.append({
                "symbol": symbol,
                "income_type": income_type,
                "income": income,
                "asset": row.get("asset", ""),
                "time": int(row.get("time", 0) or 0),
                "trade_id": str(row.get("tradeId", "") or row.get("tranId", "")),
                "info": row.get("info", ""),
            })
        return out

    def fetch_realized_pnl_summary(self, days: int = 7) -> dict[str, Any]:
        rows = self.fetch_income_history(days=days, limit=1000)
        realized_rows = [r for r in rows if r["income_type"] == "REALIZED_PNL" and r["income"] != 0]
        fee_rows = [r for r in rows if r["income_type"] in ("COMMISSION", "FUNDING_FEE")]
        realized = sum(r["income"] for r in realized_rows)
        fees = sum(r["income"] for r in fee_rows)
        wins = [r for r in realized_rows if r["income"] > 0]
        losses = [r for r in realized_rows if r["income"] < 0]
        return {
            "realized_pnl": round(realized + fees, 8),
            "gross_realized_pnl": round(realized, 8),
            "fees": round(fees, 8),
            "closed_trades": len(realized_rows),
            "win_rate": round(len(wins) / len(realized_rows) * 100, 1) if realized_rows else 0.0,
            "wins": len(wins),
            "losses": len(losses),
            "days": days,
        }

    def fetch_realized_pnl_history(self, days: int = 7, limit: int = 100) -> list[dict[str, Any]]:
        rows = [r for r in self.fetch_income_history(days=days, limit=1000)
                if r["income_type"] == "REALIZED_PNL" and r["income"] != 0]
        rows.sort(key=lambda r: r["time"], reverse=True)
        out = []
        for idx, row in enumerate(rows[:limit]):
            out.append({
                "id": row["trade_id"] or idx,
                "symbol": row["symbol"],
                "direction": "",
                "qty": 0,
                "leverage": 0,
                "entry_price": 0,
                "exit_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "pnl": row["income"],
                "pnl_pct": 0,
                "strategy_id": "exchange",
                "mode": "testnet" if self.testnet else "live",
                "opened_at": 0,
                "closed_at": row["time"] // 1000,
            })
        return out
