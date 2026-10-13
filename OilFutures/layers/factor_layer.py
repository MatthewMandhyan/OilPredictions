"""Stage 10 · Factor layer: HAR volatility + outside data, with LASSO picking what helps.

Paper: Luo, Klein, Walther & Ji (2024), "Forecasting realized volatility of crude
oil futures prices based on machine learning", Journal of Forecasting 43(5).
(Earlier SSRN version: 3701000.)

General idea (read the paper for the exact setup):
    target   = average realized variance over the next h days
    features = HAR lags (daily, weekly, monthly RV)
             + outside factors: EIA inventories, CFTC positioning, dollar, rates,
               OVX, GDELT tone, Google Trends, scheduled-event counts
    model    = LASSO (an L1 penalty shrinks useless coefficients to exactly zero),
               penalty chosen by time-series cross-validation, refit every
               walk-forward window
    output   = a variance forecast that feeds Bates:
               BatesModel(variance_forecaster=FactorHARVol([...]))

# YOUR TURN (research): implement the feature set and transformations from the
# paper's methodology section (which lags, which transforms, which ML variants they
# compare). The skeleton below is a plain, honest baseline to extend from.

This file also holds the DIRECTION TEST agreed in planning: do any factors
predict tomorrow's RETURN out of sample? (Expect: barely, if at all.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from common.calendar import settle_time, snapshot_time
from models.base import MarketData
from models.har import HARVol, har_features, har_target


# YOUR TURN
def factor_matrix(data: MarketData, dates: pd.DatetimeIndex, names: list[str]) -> pd.DataFrame:
    """Point-in-time factor values for each date, as known at that date's snapshot (≈15:05 ET).

    Returns DataFrame indexed by `dates`, one column per name.
    """
    # TODO: times = [snapshot_time(d) for d in dates]; use data.features_asof(times, names);
    #       set the index back to `dates`.
    # WHY snapshot time and not midnight: a 10:30 EIA release IS known at 15:05 the same day.
    raise NotImplementedError


# YOUR TURN
def event_counts(dates: pd.DatetimeIndex, events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """For each date t: how many events of each type fall in the next `horizon` calendar-ish days.

    Count events e with  t < e.date <= t + horizon trading days (approximate with
    t + ceil(horizon · 7/5) calendar days), AND e.announced_at < snapshot_time(t).

    Returns DataFrame indexed by dates, one column per event_type, named "evt_<type>".
    """
    # SYNTAX: f"evt_{etype}" builds a column name; df.groupby("event_type") loops over types.
    raise NotImplementedError


# OOP: FactorHARVol INHERITS from HARVol (it IS a HAR forecaster, with extra inputs)
#      and overrides only estimate(). horizon, min_obs and days_per_year come from the
#      parent's __init__ through super().__init__(...).
class FactorHARVol(HARVol):
    name = "factor_har"

    def __init__(self, factor_names: list[str], events: pd.DataFrame | None = None, horizon: int = 5,
                 min_obs: int = 250, cv_splits: int = 5, days_per_year: int = 252, seed: int = 0):
        super().__init__(horizon=horizon, min_obs=min_obs, days_per_year=days_per_year)
        self.factor_names = factor_names
        self.events = events
        self.cv_splits = cv_splits
        self.seed = seed
        self.selected_: list[str] = []
        self.model_: LassoCV | None = None

    # YOUR TURN
    def design_matrix(self, data: MarketData, rv: pd.Series) -> pd.DataFrame:
        """[rv_d, rv_w, rv_m] + factor columns (+ event-count columns if events given), indexed like rv."""
        # SYNTAX: pd.concat([a, b, c], axis=1) glues DataFrames side by side on their index.
        raise NotImplementedError

    # YOUR TURN
    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> float:
        """LASSO-HAR forecast -> annualized sigma.

        Steps:
            r = nearby_returns(as_of); rv = r²
            X = design_matrix(data, rv);  y = har_target(rv, horizon)
            train rows: X and y all present.  The LAST row of X (today) is what we predict from.
            scale X with StandardScaler fit on TRAIN rows only
            LassoCV(cv=TimeSeriesSplit(cv_splits), random_state=seed).fit(...)
            selected_ = factor/event columns with non-zero coefficients (HAR columns excluded)
            forecast = predict(today's scaled row), floored at 0.1 · mean(rv)
            return sqrt(days_per_year · forecast)
        """
        # WHY TimeSeriesSplit: ordinary K-fold shuffles time and trains on the future.
        #     TimeSeriesSplit always validates on data AFTER its training folds.
        # WHY scale: LASSO's penalty treats all coefficients alike, so a factor measured
        #     in millions of barrels would be unfairly cheap to use without scaling.
        # PITFALL: fitting the scaler on ALL rows (including today) leaks a little future
        #          information. Fit on train rows, then transform today.
        raise NotImplementedError


# =============================================================================
# Direction test (research exercise)
# =============================================================================

# YOUR TURN
def directional_test(y: pd.Series, X: pd.DataFrame, min_train: int = 500, step: int = 20,
                     block: int = 20, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Do factors X_t predict the next return y_t (= r_{t+1}) OUT OF SAMPLE?

    Walk-forward (expanding window):
        for each chunk of `step` rows after the first `min_train`:
            fit OLS  y = a + X·b  on ALL rows before the chunk
            predict the chunk;  benchmark = mean of y over the same training rows
    Out-of-sample R² (Campbell & Thompson, 2008):
        R²_oos = 1 − Σ (y − ŷ_model)² / Σ (y − ŷ_benchmark)²
    Confidence interval: block bootstrap over consecutive `block`-row chunks of the
    out-of-sample errors (same idea as Stage 3).

    Returns:
        {"r2_oos": ..., "lo": ..., "hi": ..., "n": number of out-of-sample predictions}  (90% CI)

    Interpretation: R²_oos > 0 with lo > 0 is real evidence of direction skill. Daily oil
        returns usually give R²_oos ≈ 0 or below. If yours is big, suspect lookahead first.
    """
    # TODO 1: Align y and X, drop rows with NaN.
    # TODO 2: Walk-forward loop; collect squared errors of model and benchmark.
    # TODO 3: R²_oos on all collected errors; block bootstrap for lo/hi.
    # SYNTAX: np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)[0] -> [a, b...]
    raise NotImplementedError
