"""Stage 4 · Kalshi data: markets, settlements, and trade-level data (with taker side).

Usage (PROVIDED CLI):
    python -m pipeline.kalshi --series KXWTI --status settled --trades
Saves data/raw/kalshi_markets_<series>.parquet and kalshi_trades_<series>.parquet.

Kalshi splits data into LIVE and HISTORICAL endpoints. Anything settled/created
before the cutoff from GET /historical/cutoff lives only under /historical/...
The client below asks the right endpoint automatically.

READ-ONLY. This project is paper trading only: there is no order-placing code
here, and there must never be.
"""
from __future__ import annotations

import argparse
import base64
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import requests

from common.calendar import settle_time, snapshot_time
from common.config import cfg_get, data_path, get_logger, load_config, save_parquet

log = get_logger("kalshi")


# PROVIDED
# OOP: a client class keeps connection details (base URL, session, keys) in one
#      object so every request method reuses them. requests.Session also reuses
#      the TCP connection, so it's faster than calling requests.get each time.
class KalshiClient:
    def __init__(self, base_url: str, key_id: str | None = None, private_key_path: str | None = None,
                 pause: float = 0.25):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.pause = pause
        self.key_id = key_id
        self._private_key = None
        if key_id and private_key_path and Path(os.path.expanduser(private_key_path)).exists():
            from cryptography.hazmat.primitives import serialization  # only needed with auth
            with open(os.path.expanduser(private_key_path), "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    @classmethod
    def from_config(cls, cfg: dict) -> "KalshiClient":
        """OOP: a @classmethod is an alternative constructor: KalshiClient.from_config(cfg)."""
        return cls(cfg_get(cfg, "kalshi.base_url"),
                   key_id=os.environ.get(cfg_get(cfg, "api_keys.kalshi_key_id_env", ""), None),
                   private_key_path=cfg_get(cfg, "api_keys.kalshi_private_key_path"),
                   pause=cfg_get(cfg, "kalshi.request_pause_seconds", 0.25))

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """RSA-PSS signed headers. Public market data works WITHOUT these."""
        if self._private_key is None:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        sign_path = "/trade-api/v2" + path.split("?")[0]
        sig = self._private_key.sign(
            (ts + method + sign_path).encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def get(self, path: str, params: dict | None = None, retries: int = 4) -> dict:
        for attempt in range(retries):
            resp = self.session.get(self.base_url + path, params=params,
                                    headers=self._auth_headers("GET", path), timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            time.sleep(self.pause)
            return resp.json()
        resp.raise_for_status()
        return {}

    def paginate(self, path: str, key: str, params: dict | None = None) -> Iterator[dict]:
        """Yield every item across all pages (cursor-based).

        SYNTAX: `yield` makes this a GENERATOR: it hands back items one at a time
                instead of building a giant list. Use it like a list in a for loop:
                `for market in client.paginate(...)`, or wrap it: list(client.paginate(...)).
        """
        params = dict(params or {}, limit=1000)
        while True:
            page = self.get(path, params)
            yield from page.get(key, [])
            cursor = page.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    def historical_cutoff(self) -> dict[str, pd.Timestamp]:
        raw = self.get("/historical/cutoff")
        return {k: pd.Timestamp(v) for k, v in raw.items() if isinstance(v, (str, int)) and k.endswith("_ts")}

    def markets(self, series_ticker: str, status: str | None = None, include_historical: bool = True) -> list[dict]:
        params = {"series_ticker": series_ticker}
        if status:
            params["status"] = status
        out = list(self.paginate("/markets", "markets", params))
        if include_historical:
            try:
                out += list(self.paginate("/historical/markets", "markets", {"series_ticker": series_ticker}))
            except requests.HTTPError as e:
                log.warning("historical markets unavailable: %s", e)
        # de-duplicate by ticker (a market can appear in both during the hand-off)
        return list({m["ticker"]: m for m in out}.values())

    def trades(self, ticker: str, min_ts: int | None = None, max_ts: int | None = None) -> list[dict]:
        params = {"ticker": ticker}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        out = list(self.paginate("/markets/trades", "trades", params))
        try:
            out += list(self.paginate("/historical/trades", "trades", params))
        except requests.HTTPError as e:
            log.warning("historical trades unavailable: %s", e)
        return list({t["trade_id"]: t for t in out}.values())


# PROVIDED
def _to_et(series: pd.Series) -> pd.Series:
    """UTC strings -> naive America/New_York timestamps (project convention)."""
    return pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)


# PROVIDED
def _num(raw: dict, *names: str, scale: float = 1.0) -> float | None:
    """First field present among names. Kalshi moved from cents (`yes_price`) to dollar strings
    (`yes_price_dollars`) and fixed-point counts (`count_fp`); this accepts either."""
    for n in names:
        v = raw.get(n)
        if v not in (None, ""):
            return float(v) * (scale if not n.endswith(("_dollars", "_fp")) else 1.0)
    return None


# PROVIDED
def markets_to_frame(markets: list[dict]) -> pd.DataFrame:
    rows = [{
        "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "status": m.get("status"),
        "result": m.get("result"), "strike_type": m.get("strike_type"),
        "floor_strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
        "open_time": m.get("open_time"), "close_time": m.get("close_time"),
        "expected_expiration_time": m.get("expected_expiration_time") or m.get("expiration_time"),
        "yes_bid": _num(m, "yes_bid_dollars", "yes_bid", scale=0.01),
        "yes_ask": _num(m, "yes_ask_dollars", "yes_ask", scale=0.01),
        "last_price": _num(m, "last_price_dollars", "last_price", scale=0.01),
        "volume": _num(m, "volume_fp", "volume"),
        "rules_primary": m.get("rules_primary", ""), "rules_secondary": m.get("rules_secondary", ""),
    } for m in markets]
    df = pd.DataFrame(rows)
    for col in ("open_time", "close_time", "expected_expiration_time"):
        if col in df and df[col].notna().any():
            df[col] = _to_et(df[col])
    return df


# PROVIDED
def trades_to_frame(trades: list[dict]) -> pd.DataFrame:
    """Columns: trade_id, ticker, time (ET), yes_price ($), count, taker_side ("yes"/"no")."""
    rows = [{
        "trade_id": t.get("trade_id"), "ticker": t.get("ticker"), "time": t.get("created_time"),
        "yes_price": _num(t, "yes_price_dollars", "yes_price", scale=0.01),
        "count": _num(t, "count_fp", "count"),
        "taker_side": t.get("taker_outcome_side") or t.get("taker_side"),
    } for t in trades]
    df = pd.DataFrame(rows, columns=["trade_id", "ticker", "time", "yes_price", "count", "taker_side"])
    if len(df):
        df["time"] = _to_et(df["time"])
    return df.sort_values("time").reset_index(drop=True)


# YOUR TURN
def snapshot_prices(trades: pd.DataFrame, delay_minutes: int = 5, window_minutes: int = 30) -> pd.DataFrame:
    """The CLEAN daily market price: volume-weighted price of trades just after the settle.

    For each (ticker, date), keep trades with
        settle_time(date) + delay  <=  time  <  settle_time(date) + delay + window
    and compute
        VWAP = Σ(price · count) / Σ(count)

    WHY: with free daily data your F is only fresh right after 14:30. Comparing to
         Kalshi prices from the same window means model and crowd saw the same price.
         Comparing to a 10am price would reward the crowd for knowing today's move.

    Args:
        trades: output of trades_to_frame (columns ticker, time, yes_price, count).

    Returns:
        DataFrame [ticker, date, as_of, market_price, n_trades, volume], where
        as_of = snapshot_time(date, delay, window) (the END of the window).

    Example:
        trades at 14:40 (0.40 × 10) and 14:50 (0.50 × 30) on 2026-10-12
        -> market_price = (4 + 15) / 40 = 0.475, n_trades = 2, volume = 40
    """
    # TODO 1: Copy trades; add a `date` column = time normalized to midnight.
    # TODO 2: Compute each trade's window start/end from its date (vectorized:
    #         date + pd.Timedelta(hours=14, minutes=30 + delay_minutes), etc.).
    # TODO 3: Keep trades inside the window.
    # TODO 4: Group by [ticker, date]; compute VWAP, count of trades, total volume.
    # TODO 5: Add as_of with snapshot_time; return the columns listed above.
    # SYNTAX: you can't call settle_time() on a whole column at once, but adding a
    #         Timedelta to a datetime column works element-wise: df["date"] + pd.Timedelta(...)
    # HINT: for the VWAP, first make a column px_x_qty = yes_price * count, then sum both per group.
    # PITFALL: thin windows. If n_trades is tiny the "price" is one person's opinion;
    #          report results with a minimum-trades filter as well as without.
    raise NotImplementedError


# PROVIDED
def main() -> None:
    ap = argparse.ArgumentParser(description="Download Kalshi WTI markets and trades.")
    ap.add_argument("--series", required=True, help="series ticker, e.g. KXWTI")
    ap.add_argument("--status", default=None, help="open | closed | settled (default: all)")
    ap.add_argument("--trades", action="store_true", help="also download trades for every market")
    args = ap.parse_args()

    cfg = load_config()
    client = KalshiClient.from_config(cfg)
    markets = markets_to_frame(client.markets(args.series, args.status))
    save_parquet(markets, data_path(cfg, "raw", f"kalshi_markets_{args.series}.parquet"))
    log.info("saved %d markets", len(markets))
    if args.trades:
        frames = [trades_to_frame(client.trades(t)) for t in markets["ticker"]]
        trades = pd.concat(frames, ignore_index=True) if frames else trades_to_frame([])
        save_parquet(trades, data_path(cfg, "raw", f"kalshi_trades_{args.series}.parquet"))
        log.info("saved %d trades", len(trades))


if __name__ == "__main__":
    main()
