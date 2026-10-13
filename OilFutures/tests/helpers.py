# PROVIDED: small builders shared by the tests.
from __future__ import annotations

import numpy as np
import pandas as pd

from common.calendar import settle_time, trading_days
from models.base import FuturesCurve, MarketData


def make_curve(prices, month: str = "2026-12", start: str = "2026-01-02") -> MarketData:
    """One futures month with the given daily settles on consecutive trading days."""
    days = trading_days(start, pd.Timestamp(start) + pd.Timedelta(days=len(prices) * 2 + 30))[: len(prices)]
    df = pd.DataFrame({
        "date": days,
        "futures_month": month,
        "settle": np.asarray(prices, dtype=float),
        "available_at": [settle_time(d) for d in days],
    })
    return MarketData(FuturesCurve(df))


def gbm_prices(n: int, sigma: float, F0: float = 70.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    r = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(n - 1)
    return F0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def data_from_returns(r, start: str = "2010-01-04"):
    """Synthetic single-month curve from a return series. Returns (MarketData, trading days)."""
    r = np.asarray(r, dtype=float)
    days = trading_days(start, pd.Timestamp(start) + pd.Timedelta(days=int(len(r) * 1.6) + 40))[: len(r) + 1]
    prices = 70 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    df = pd.DataFrame({"date": days, "futures_month": "2099-12", "settle": prices,
                       "available_at": [settle_time(d) for d in days]})
    return MarketData(FuturesCurve(df)), days
