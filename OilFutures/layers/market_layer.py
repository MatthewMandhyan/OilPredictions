"""Stage 11 · Market layer: adjust the engine's probability for how the Kalshi CROWD behaves.

The engine knows oil. The crowd has habits. This layer learns, with a logistic
regression, when the crowd's price carries information the engine lacks (e.g.
informed taker flow) and when it is just a habit to fade (e.g. overpaying for longshots).

Papers (read for method details; this skeleton is a plain baseline):
  * Bürgi, Deng & Whelan (2025), "Makers and Takers: The Economics of the Kalshi
    Prediction Market", CEPR DP20631: favorite–longshot bias, taker-flow features.
  * Portnaya (2026), "Do Prediction Markets Match Option Prices?", arXiv 2606.19517:
    track the gap (market − model) over time and by probability tercile.
  * Akey et al. (2026), "Who Wins and Who Loses in Prediction Markets? Evidence from
    Polymarket": slice the gap by tail contracts, high-volatility periods, post-shock windows.

# YOUR TURN (research): refine the feature definitions from the papers' methodology
# sections. Use the gap analysis functions at the bottom BEFORE trusting any feature.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from models.base import Contract, MarketData, Prediction, PriceModel

FEATURES = ["logit_engine", "logit_market", "longshot", "favorite", "taker_imbalance",
            "overreaction", "hours_to_close", "cross_strike_gap", "volume_spike"]
EPS = 1e-4


# YOUR TURN
def logit(p: np.ndarray | float, eps: float = EPS) -> np.ndarray:
    """ln(p / (1 − p)) after clipping p into [eps, 1 − eps]."""
    raise NotImplementedError


# YOUR TURN
def taker_imbalance(trades: pd.DataFrame, as_of: pd.Timestamp, lookback_hours: float = 24.0) -> float:
    """Net taker buying of YES over the lookback window, in [−1, 1].

    Formula:  (Σ count where taker_side == "yes" − Σ count where taker_side == "no") / Σ count
    Use trades with  as_of − lookback < time < as_of.   0.0 if there are none.

    WHY: takers pay the spread to trade NOW. A run of YES-takers is someone in a hurry,
         and sometimes that someone knows something.
    """
    raise NotImplementedError


# YOUR TURN
def cross_strike_inconsistency(strikes: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """How far each "above" price is from the nearest monotone (non-increasing in strike) ladder.

    Prices of "above K" MUST fall as K rises. Fit an isotonic (monotone) regression
    and return  price − fitted  for each contract (0 everywhere if the ladder is consistent).

    Example: strikes [70, 71, 72], prices [0.60, 0.40, 0.45]
             -> the 71 and 72 prices violate monotonicity; fitted [0.60, 0.425, 0.425]
             -> [0.0, −0.025, 0.025]
    """
    # SYNTAX: IsotonicRegression(increasing=False).fit_transform(x, y)
    # PITFALL: sort by strike first, then put results back in the ORIGINAL order.
    raise NotImplementedError


# YOUR TURN
def volume_spike(volume_today: float, past_volumes: np.ndarray) -> float:
    """ln( (volume_today + 1) / (median(past_volumes) + 1) ).  0.0 if there is no history."""
    raise NotImplementedError


# YOUR TURN
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into the FEATURES matrix.

    Input columns:
        engine_p, market_p                     today's engine probability and Kalshi price
        prev_engine_p, prev_market_p           the same, one snapshot earlier (NaN if none)
        taker_imbalance, hours_to_close, cross_strike_gap, volume_spike
    Definitions:
        logit_engine = logit(engine_p);  logit_market = logit(market_p)
        longshot = 1 if market_p < 0.15 else 0;   favorite = 1 if market_p > 0.85 else 0
        overreaction = (logit_market − logit(prev_market_p)) − (logit_engine − logit(prev_engine_p)),
                       0 where prev values are missing
          ("the crowd moved more than the fundamentals did", e.g. after a headline)
    Returns:
        DataFrame with exactly the FEATURES columns, in that order.
    """
    # SYNTAX: (df["market_p"] < 0.15).astype(int) turns True/False into 1/0.
    raise NotImplementedError


# OOP: MarketLayer wraps a scikit-learn model behind a tiny interface (fit / adjust)
#      that speaks OUR language (DataFrames with named features). If you later swap
#      logistic regression for something else, nothing outside this class changes.
class MarketLayer:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.model_: LogisticRegression | None = None

    # YOUR TURN
    def fit(self, df: pd.DataFrame) -> "MarketLayer":
        """Fit on historical rows (raw columns for build_features + `outcome`). Returns self."""
        # SYNTAX: returning self lets you chain: layer = MarketLayer().fit(train)
        # PITFALL: fit on PAST rows only (walk-forward), exactly like every other model.
        raise NotImplementedError

    # YOUR TURN
    def adjust(self, df: pd.DataFrame) -> np.ndarray:
        """Adjusted probabilities, clipped into [EPS, 1 − EPS] so they are never exactly 0 or 1."""
        raise NotImplementedError

    @property
    def coef_(self) -> dict[str, float]:
        """{feature: coefficient}: check the signs make sense before trusting the layer."""
        # SYNTAX: dict(zip(keys, values)) pairs two lists into a dictionary.
        return dict(zip(FEATURES, self.model_.coef_[0]))


# OOP: MarketAdjustedModel is COMPOSITION in action. It HAS an engine and HAS a layer,
#      and IS a PriceModel itself. So the backtester and ledger can't tell it apart
#      from a plain engine: Backtester(MarketAdjustedModel(BatesModel(), layer, lookup), ...)
MarketLookup = Callable[[Contract, pd.Timestamp], dict]


class MarketAdjustedModel(PriceModel):
    version = "1.0"

    def __init__(self, engine: PriceModel, layer: MarketLayer, market_lookup: MarketLookup):
        super().__init__()
        self.engine, self.layer, self.market_lookup = engine, layer, market_lookup
        self.name = f"{engine.name}+market"

    # YOUR TURN
    def fit(self, data: MarketData, as_of: pd.Timestamp) -> None:
        """Fit the ENGINE (the layer is fit separately on historical engine-vs-market rows)."""
        raise NotImplementedError

    # YOUR TURN
    def _adjusted(self, base: Prediction, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """engine prediction + market_lookup(contract, as_of) -> one-row DataFrame -> layer.adjust."""
        # TODO 1: row = {"engine_p": base.probability, **self.market_lookup(contract, as_of)}
        #         SYNTAX: {**a, **b} merges two dicts.
        # TODO 2: p = self.layer.adjust(pd.DataFrame([row]))[0]
        # TODO 3: return self._prediction(contract, p, as_of, engine_p=..., **base.inputs)
        #         (keep the engine's inputs so the lookahead check still sees F_available_at)
        raise NotImplementedError

    def prob_terminal(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        return self._adjusted(self.engine.prob_terminal(contract, as_of), contract, as_of)

    def prob_barrier(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        return self._adjusted(self.engine.prob_barrier(contract, as_of), contract, as_of)


# =============================================================================
# Gap analysis (do this BEFORE fitting the layer)
# =============================================================================

# YOUR TURN
def gap_by_tercile(df: pd.DataFrame) -> pd.DataFrame:
    """Portnaya-style table: gap = market_p − engine_p, by tercile of engine_p.

    Returns one row per tercile ("low", "mid", "high") with columns
        n, engine_mean, market_mean, outcome_mean, gap_mean
    Reading it: if outcome_mean sits closer to engine_mean than to market_mean in a
    tercile, the crowd is mispricing that tercile.
    """
    # SYNTAX: pd.qcut(series, 3, labels=["low", "mid", "high"]) splits into equal-count groups.
    raise NotImplementedError


# YOUR TURN
def gap_slices(df: pd.DataFrame, tail: float = 0.10) -> pd.DataFrame:
    """Akey-style slices of the gap. Adds boolean flags, then summarizes each:

        tail       engine_p < tail or engine_p > 1 − tail
        high_vol   sigma above its median
        post_shock |prev_return| > 2 · its std

    Returns DataFrame indexed by slice name ["tail", "high_vol", "post_shock", "all"]
    with columns n, gap_mean, brier_engine, brier_market.
    """
    # SYNTAX: a dict of name -> boolean mask, then a loop (or comprehension) over .items().
    raise NotImplementedError
