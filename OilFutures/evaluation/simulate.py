"""Stage 3 · Synthetic world: GBM prices, contracts, outcomes and a noisy "market".

WHY: before trusting a backtest on real data, run it where you KNOW the right
answer. Prices here really are GBM, so a correct GBM model must come out
calibrated and must beat a market that is the truth plus noise. If it doesn't,
the bug is in your harness, not in the market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.calendar import settle_time, snapshot_time, trading_days
from models.base import Contract, FuturesCurve, MarketData
from models.gbm import prob_above

SYNTHETIC_MONTH = "2099-12"


# PROVIDED
def simulate_market_data(n_days: int = 1000, F0: float = 70.0, sigma: float = 0.35, seed: int = 0,
                         start: str = "2015-01-02") -> MarketData:
    """One synthetic futures month that never expires, following driftless GBM."""
    rng = np.random.default_rng(seed)
    days = trading_days(start, pd.Timestamp(start) + pd.Timedelta(days=int(n_days * 1.6) + 30))[:n_days]
    dt = 1 / 252
    r = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days - 1)
    prices = F0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    df = pd.DataFrame({
        "date": days,
        "futures_month": SYNTHETIC_MONTH,
        "settle": prices,
        "available_at": [settle_time(d) for d in days],
    })
    return MarketData(FuturesCurve(df))


# PROVIDED
def make_synthetic_contracts(data: MarketData, sigma_true: float, horizons: tuple[int, ...] = (5,),
                             strike_sds: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
                             first_index: int = 80, step: int = 1) -> pd.DataFrame:
    """Daily "above" contracts at strikes spaced in std devs around the current price.

    Returns one row per contract with columns:
        contract (Contract), ticker, date, as_of (post-settle snapshot), outcome (0/1), true_prob
    """
    df = data.curve.df.sort_values("date").reset_index(drop=True)
    dates, prices = df["date"].to_numpy(), df["settle"].to_numpy()
    rows = []
    for i in range(first_index, len(df), step):
        for h in horizons:
            j = i + h
            if j >= len(df):
                continue
            F, T = prices[i], h / 252
            for k in strike_sds:
                K = round(float(F * np.exp(k * sigma_true * np.sqrt(T))), 2)
                c = Contract(ticker=f"SYN-{pd.Timestamp(dates[i]):%Y%m%d}-H{h}-K{K:.2f}", contract_type="above",
                             strike=K, settle_date=pd.Timestamp(dates[j]), futures_month=SYNTHETIC_MONTH)
                rows.append({
                    "contract": c, "ticker": c.ticker, "date": pd.Timestamp(dates[i]),
                    "as_of": snapshot_time(dates[i]),
                    "outcome": int(prices[j] > K),
                    "true_prob": prob_above(F, K, sigma_true, T),
                })
    return pd.DataFrame(rows)


# YOUR TURN
def add_logit_noise(p: np.ndarray, noise_sd: float, rng: np.random.Generator, eps: float = 1e-4) -> np.ndarray:
    """A fake "market": the true probability distorted by random noise in log-odds space.

    Formula:  logit(p) = ln(p / (1 − p));   market = sigmoid( logit(p) + noise_sd · Z )
              sigmoid(x) = 1 / (1 + e^{−x})

    WHY log-odds: adding noise directly to p could push it below 0 or above 1.
        In log-odds space any real number maps back into (0, 1).

    Example:
        >>> add_logit_noise(np.array([0.2, 0.5]), 0.0, np.random.default_rng(0))
        array([0.2, 0.5])
    """
    # TODO 1: Clip p into [eps, 1 − eps] (logit(0) is −inf).
    # TODO 2: logit -> add noise -> sigmoid.
    raise NotImplementedError


# PROVIDED
def synthetic_backtest_inputs(n_days: int = 1000, sigma: float = 0.35, noise_sd: float = 0.4,
                              seed: int = 0) -> tuple[MarketData, pd.DataFrame]:
    """One call that builds data + contracts + a noisy market_price column."""
    data = simulate_market_data(n_days=n_days, sigma=sigma, seed=seed)
    contracts = make_synthetic_contracts(data, sigma_true=sigma)
    contracts["market_price"] = add_logit_noise(contracts["true_prob"].to_numpy(), noise_sd,
                                                np.random.default_rng(seed + 1))
    return data, contracts
