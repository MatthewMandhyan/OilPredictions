"""Stage 6 · Better volatility inputs for GBM: HAR forecast, OVX implied vol, scheduled-event bumps.

In Stage 1 the ONLY thing GBM disagrees with Kalshi about is sigma. So the
fastest way to improve GBM is a better sigma. Three estimators here, all
plugging into GBMModel's `vol` slot:

    GBMModel(vol=HARVol())                                  # forecast from past realized variance
    GBMModel(vol=ImpliedVol())                              # the options market's forecast (OVX)
    GBMModel(vol=EventAwareVol(HARVol(), events))           # + extra variance on OPEC/EIA/FOMC days

Paper behind HAR: Corsi (2009), "A Simple Approximate Long-Memory Model of
Realized Volatility", Journal of Financial Econometrics 7(2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.calendar import settle_time
from models.base import MarketData, VolatilityEstimator


# =============================================================================
# HAR
# =============================================================================

# YOUR TURN
def har_features(rv: pd.Series) -> pd.DataFrame:
    """Daily, weekly and monthly averages of realized variance, all ending at day t.

    Formula:
        rv_d(t) = RV_t
        rv_w(t) = (1/5)  Σ_{i=0..4}  RV_{t−i}
        rv_m(t) = (1/22) Σ_{i=0..21} RV_{t−i}

    WHY three horizons: traders act on different clocks (day traders, weekly funds,
        monthly allocators). Mixing three averages mimics "long memory" in volatility
        with a plain linear regression.

    Returns:
        DataFrame with columns [rv_d, rv_w, rv_m], same index as rv (NaN where not enough history).
    """
    # SYNTAX: rv.rolling(5).mean() is a moving average over the last 5 rows (including today).
    # SYNTAX: build a DataFrame from a dict of columns: pd.DataFrame({"a": s1, "b": s2})
    raise NotImplementedError


# YOUR TURN
def har_target(rv: pd.Series, horizon: int) -> pd.Series:
    """Average realized variance over the NEXT `horizon` days: (1/h) Σ_{i=1..h} RV_{t+i}.

    Example (horizon=2): rv = [1, 2, 3, 4]  ->  [2.5, 3.5, NaN, NaN]
    """
    # HINT 1: rv.rolling(h).mean() at row t+h is the mean of rows t+1..t+h.
    # HINT 2: .shift(-h) moves that value back to row t.
    # PITFALL: this column uses the FUTURE by design (it's what we predict). It must
    #          never be used as a feature, only as the regression target.
    raise NotImplementedError


# YOUR TURN
def fit_har(rv: pd.Series, horizon: int) -> np.ndarray:
    """OLS coefficients [b0, b_d, b_w, b_m] of   target = b0 + b_d·rv_d + b_w·rv_w + b_m·rv_m.

    Returns:
        numpy array of 4 coefficients.
    """
    # TODO 1: X = har_features(rv), y = har_target(rv, horizon).
    # TODO 2: Keep rows where X and y are all non-NaN (a boolean mask with .notna().all(axis=1)).
    # TODO 3: Add a column of ones for the intercept; solve least squares.
    # SYNTAX: np.column_stack([np.ones(n), X.to_numpy()]) adds the ones column;
    #         np.linalg.lstsq(A, y, rcond=None)[0] returns the coefficients.
    raise NotImplementedError


# OOP: HARVol INHERITS from VolatilityEstimator, so GBMModel accepts it in place of
#      HistoricalVol without a single change to GBMModel. In Stage 10 FactorHARVol
#      will in turn inherit from HARVol and only override the parts that differ.
class HARVol(VolatilityEstimator):
    """Volatility forecast for the next `horizon` days from a HAR regression on squared returns."""
    name = "har"

    def __init__(self, horizon: int = 5, min_obs: int = 250, days_per_year: int = 252):
        self.horizon = horizon
        self.min_obs = min_obs
        self.days_per_year = days_per_year
        self.coef_: np.ndarray | None = None   # SYNTAX: trailing _ = "learned from data" (sklearn convention)

    # YOUR TURN
    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> float:
        """Annualized sigma = sqrt(252 · forecast of average daily variance over the horizon).

        Steps:
            r  = data.curve.nearby_returns(as_of);  RV_t = r_t²
            coef = fit_har(RV, horizon)            (store on self.coef_)
            x_now = last row of har_features(RV)
            forecast = coef · [1, x_now]
            floor: forecast = max(forecast, 0.1 · mean(RV))    (OLS can predict negative variance)
            return sqrt(days_per_year · forecast)
        """
        # TODO: follow the steps; raise ValueError if fewer than self.min_obs returns.
        # PITFALL: squared daily returns are a NOISY variance proxy (one big day dominates).
        #          With intraday data you'd use the sum of squared 5-minute returns instead.
        raise NotImplementedError


# =============================================================================
# Implied volatility
# =============================================================================

class ImpliedVol(VolatilityEstimator):
    """Use the options market's own forecast: CBOE Crude Oil Volatility Index (FRED: OVXCLS).

    sigma = scale · OVX / 100, using the latest OVX value public before as_of
    (data.features must contain rows with name == feature_name; built in Stage 4).
    """
    name = "ovx"

    def __init__(self, feature_name: str = "OVXCLS", scale: float = 1.0):
        self.feature_name = feature_name
        self.scale = scale

    # YOUR TURN
    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> float:
        # TODO 1: value = data.features_asof([as_of], [self.feature_name]).iloc[0, 0]
        # TODO 2: Raise ValueError if it is NaN (nothing public yet).
        # TODO 3: Return self.scale * value / 100.
        # WHY `scale`: implied vol usually sits ABOVE later realized vol (option sellers
        #     charge a premium). Tune `scale` by walk-forward validation, never on the
        #     whole history at once.
        # PITFALL: OVX is built from USO ETF options (~30-day horizon), not from WTI
        #          futures options at your exact horizon. It's a proxy.
        raise NotImplementedError


# =============================================================================
# Scheduled events
# =============================================================================

# YOUR TURN
def event_extra_variance(returns: pd.Series, event_dates: pd.DatetimeIndex, min_events: int = 5) -> float:
    """How much EXTRA daily variance an event day carries, versus a normal day.

    Formula:  extra = mean(r² on event days) − mean(r² on all other days),  floored at 0
    Returns 0.0 if fewer than `min_events` event days appear in `returns`.

    Example: normal days have r² ≈ 0.0004 and EIA days r² ≈ 0.0010  ->  extra ≈ 0.0006
    """
    # TODO 1: mask = returns.index.isin(event_dates)
    # TODO 2: If mask.sum() < min_events: return 0.0
    # TODO 3: Compare mean squared returns on event vs non-event days; floor at 0.
    # SYNTAX: ~mask flips a boolean array (True <-> False).
    raise NotImplementedError


# OOP: EventAwareVol WRAPS another estimator (composition) instead of inheriting from
#      one. It asks the inner estimator for the normal sigma, then adds its own extra
#      piece. Because it wraps ANY estimator, one class works with HistoricalVol,
#      HARVol or ImpliedVol. This is called the "decorator pattern".
class EventAwareVol(VolatilityEstimator):
    """Base volatility + extra variance for each scheduled event before settlement.

    Total variance over a contract's life:
        σ²·T  +  Σ_{scheduled events e in the window} extra_var[type(e)]

    events: DataFrame [date, event_type, announced_at] (pipeline.fundamentals.load_events).
    """
    name = "event"

    def __init__(self, base: VolatilityEstimator, events: pd.DataFrame, min_events: int = 5):
        self.base = base
        self.events = events
        self.min_events = min_events
        self.extra_var_: dict[str, float] = {}

    # YOUR TURN
    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> float:
        """Return the base sigma, and as a side effect learn self.extra_var_ per event type."""
        # TODO 1: sigma = self.base.estimate(data, as_of)
        # TODO 2: returns = data.curve.nearby_returns(as_of)
        # TODO 3: For each event_type: past event dates (settle_time(date) < as_of) ->
        #         event_extra_variance(...). Store in self.extra_var_ (a dict).
        # SYNTAX: for etype, group in self.events.groupby("event_type"): ...
        raise NotImplementedError

    # YOUR TURN
    def horizon_variance(self, sigma: float, T: float, as_of: pd.Timestamp,
                         settle_date: pd.Timestamp) -> float:
        """sigma²·T plus extra_var for every event that is:
             - announced before as_of            (announced_at < as_of)
             - still ahead                       (settle_time(event date) > as_of)
             - on or before the settle date      (event date <= settle_date)
        """
        # OOP: this OVERRIDES VolatilityEstimator.horizon_variance. GBMModel calls the same
        #      method name and automatically gets this richer version. That's polymorphism.
        # PITFALL: only count events whose DATE was public at as_of. OPEC+ meetings get
        #          moved; using the final calendar in a backtest is lookahead.
        raise NotImplementedError
