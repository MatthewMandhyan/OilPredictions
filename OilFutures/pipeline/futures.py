"""Stage 4 · WTI settlement prices BY CONTRACT MONTH.

Free source used here: EIA publishes daily NYMEX WTI settles for the 1st–4th
nearest contracts (series RCLC1..RCLC4), back to the 1980s. Those are labeled by
POSITION ("contract 1"), not by month, so you convert position -> month using
the expiry calendar. That's your job in this file.

PITFALL: a "continuous front-month" series (RCLC1 alone, or Yahoo's CL=F) jumps
         on every roll day because it switches contracts. Kalshi settles on a
         SPECIFIC month, so always work with month-labeled prices.
PITFALL: Kalshi's rules name ICE WTI; EIA's data is NYMEX WTI. The two settle
         within cents of each other, which is fine for learning. A paid source
         (ICE, Databento, Nasdaq Data Link) gets you the exact series.
NOTE:    EIA updates this data with a delay, so for LIVE predictions use a same-day
         source (fetch_yahoo_settles below) after 14:30 ET.

Usage (PROVIDED CLI):  python -m pipeline.futures --start 2015-01-01
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests

from common.calendar import HOLIDAYS, add_trading_days, is_trading_day, settle_time
from common.config import cfg_get, data_path, env_key, get_logger, load_config, save_parquet

log = get_logger("futures")
EIA_URL = "https://api.eia.gov/v2/petroleum/pri/fut/data/"


# PROVIDED
def fetch_eia_futures(api_key: str, start: str, end: str | None = None,
                      series: tuple[str, ...] = ("RCLC1", "RCLC2", "RCLC3", "RCLC4")) -> pd.DataFrame:
    """Daily settles by position. Returns columns [date, position, settle]."""
    rows, offset = [], 0
    while True:
        params = {"api_key": api_key, "frequency": "daily", "data[0]": "value", "start": start,
                  "offset": offset, "length": 5000, "sort[0][column]": "period", "sort[0][direction]": "asc"}
        if end:
            params["end"] = end
        for i, s in enumerate(series):
            params[f"facets[series][{i}]"] = s
        resp = requests.get(EIA_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()["response"]["data"]
        rows += data
        if len(data) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "position", "settle"])
    return pd.DataFrame({
        "date": pd.to_datetime(df["period"]),
        "position": df["series"].str.extract(r"(\d)$")[0].astype(int),
        "settle": pd.to_numeric(df["value"]),
    }).dropna().sort_values(["date", "position"]).reset_index(drop=True)


# PROVIDED
def fetch_yahoo_settles(symbol: str, days: int = 30) -> pd.DataFrame:
    """Recent daily closes for ONE month from Yahoo's unofficial chart endpoint, e.g. "CLZ26.NYM".

    Unofficial and may break; fine for live paper trading, not for research.
    Returns columns [date, settle].
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    resp = requests.get(url, params={"range": f"{days}d", "interval": "1d"},
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    r = resp.json()["chart"]["result"][0]
    dates = pd.to_datetime(r["timestamp"], unit="s").normalize()
    return pd.DataFrame({"date": dates, "settle": r["indicators"]["quote"][0]["close"]}).dropna()


# PROVIDED
def yahoo_symbol(futures_month: str) -> str:
    """"2026-12" -> "CLZ26.NYM" (futures month codes: F G H J K M N Q U V X Z)."""
    codes = "FGHJKMNQUVXZ"
    y, m = futures_month.split("-")
    return f"CL{codes[int(m) - 1]}{y[2:]}.NYM"


# YOUR TURN
def wti_last_trade_date(futures_month: str, business_days_before_25th: int = 4) -> pd.Timestamp:
    """Last trading day of a WTI futures month.

    Rule (ICE WTI; NYMEX uses 3 instead of 4):
        anchor = the 25th of the month BEFORE the contract month
                 (if the 25th isn't a business day, the last business day before it)
        last trade = anchor minus `business_days_before_25th` business days

    Examples:
        >>> wti_last_trade_date("2026-12")    # Nov 25 2026 is a Wednesday
        Timestamp('2026-11-19 00:00:00')
        >>> wti_last_trade_date("2027-01")    # Dec 25 is a holiday -> anchor Dec 24
        Timestamp('2026-12-18 00:00:00')
    """
    # TODO 1: Parse "YYYY-MM" and find the 25th of the PREVIOUS month.
    #         SYNTAX: pd.Period("2026-12", freq="M") - 1 gives Period("2026-11"); .to_timestamp() makes a date.
    # TODO 2: Step back one day at a time until is_trading_day(anchor) is True (a while loop).
    # TODO 3: Shift back with add_trading_days(anchor, -n).
    # PITFALL: verify against the exchange's published expiry calendar for a few
    #          months. A one-day error silently puts F on the wrong contract.
    raise NotImplementedError


# YOUR TURN
def month_for_position(date: pd.Timestamp, position: int, business_days_before_25th: int = 4) -> str:
    """Which contract month is the `position`-th nearest on `date`.

    Position 1 = the first month whose last trade date is on or after `date`.
    Position k = k − 1 months after that.

    Example:
        >>> month_for_position(pd.Timestamp("2026-11-18"), 1)
        '2026-12'
        >>> month_for_position(pd.Timestamp("2026-11-20"), 1)    # Dec expired on the 19th
        '2027-01'
        >>> month_for_position(pd.Timestamp("2026-11-20"), 2)
        '2027-02'
    """
    # TODO 1: Start with the month after `date`'s month (pd.Period(date, freq="M") + 1).
    # TODO 2: While that month's last trade date < date, move one month later.
    # TODO 3: Add position − 1 months; return str(period) (gives "YYYY-MM").
    raise NotImplementedError


# YOUR TURN
def build_curve_frame(positions: pd.DataFrame, business_days_before_25th: int = 4) -> pd.DataFrame:
    """Turn [date, position, settle] into the FuturesCurve format.

    Returns columns [date, futures_month, settle, available_at] where
    available_at = settle_time(date) (the settle is public at 14:30 ET that day).

    WHY: from here on, nothing refers to "position 1"; everything uses real months.
    """
    # TODO 1: Map each (date, position) row to its month with month_for_position.
    #         SYNTAX: a list comprehension over two columns at once:
    #         [f(d, p) for d, p in zip(df["date"], df["position"])]
    # TODO 2: available_at with settle_time (same zip/list trick, or df["date"] + Timedelta).
    # TODO 3: Return just the four columns, sorted by [futures_month, date].
    # PITFALL: use the expiry rule of the exchange the DATA comes from. EIA's positions are
    #          NYMEX (3 days); Kalshi's settlement month follows ICE (4 days).
    # PITFALL: the same month can appear as position 2 one day and position 1 the next.
    #          That's correct: it's the same contract, so its prices form one series.
    raise NotImplementedError


# PROVIDED
def main() -> None:
    ap = argparse.ArgumentParser(description="Download WTI settles by contract month (EIA).")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    cfg = load_config()
    pos = fetch_eia_futures(env_key(cfg, "eia_env"), args.start, args.end,
                            tuple(cfg_get(cfg, "futures.eia_positions")))
    # EIA's RCLC1..4 are NYMEX positions, so label them with the NYMEX expiry rule (3), not ICE's (4).
    curve = build_curve_frame(pos, cfg_get(cfg, "futures.nymex_expiry_business_days_before_25th", 3))
    save_parquet(curve, data_path(cfg, "processed", "wti_curve.parquet"))
    log.info("saved %d rows across %d months", len(curve), curve["futures_month"].nunique())


if __name__ == "__main__":
    main()
