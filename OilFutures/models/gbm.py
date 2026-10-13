"""Stages 1–2 · GBM engine: the baseline every other model has to beat.

Stage 1: closed-form probabilities + historical volatility + the GBMModel class.
Stage 2: Monte Carlo paths, terminal probabilities from paths, barrier (max/min) probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from common.calendar import trading_days_until, year_fraction
from models.base import Contract, MarketData, Prediction, PriceModel, VolatilityEstimator

# OOP: why are prob_above / prob_below / prob_between plain FUNCTIONS and not
#      methods? They're pure math: same inputs, same output, no memory needed.
#      A class earns its place when something must REMEMBER state between calls
#      (like GBMModel remembering sigma after fit). Rule of thumb: no state, no class.


# =============================================================================
# STAGE 1: closed form
# =============================================================================

# YOUR TURN
def prob_above(F: float, K: float, sigma: float, T: float) -> float:
    """Probability the futures settle finishes ABOVE strike K under driftless GBM.

    Model:    ln F_T ~ Normal( ln F − ½σ²T ,  σ²T )
    Formula:  P(F_T > K) = N(d2)        (the "N(d2)" term of the Black-76 model)

    Args:
        F: current price of the futures month this contract settles on ($/bbl).
        K: strike ($/bbl).
        sigma: annualized volatility (0.35 means 35%).
        T: time to settlement in years of trading time.

    Returns:
        Probability in [0, 1].

    Example:
        >>> round(prob_above(70.0, 70.0, 0.35, 10/252), 4)
        0.4861
    """
    # TODO 1: If T <= 0 or sigma <= 0 there's no uncertainty left: return 1.0 if F > K else 0.0.
    # TODO 2: Compute the standard deviation of ln F_T (the total volatility over the horizon).
    # TODO 3: Compute d2.
    # TODO 4: Turn d2 into a probability with the standard normal CDF (norm.cdf).
    vol_T = ...  # HINT: look at the variance in the "Model" line of the docstring
    d2 = ...  # HINT 1: d2 is a z-score: how many std devs the CENTER of ln F_T sits above ln K.
              # HINT 2: z-score = (center − threshold) / std.  Reference: "Black-76 model, d2".
    # WHY no interest rate: a futures price already has zero drift under the pricing
    #     measure, so centering on F avoids guessing storage costs and carry.
    # PITFALL: F must be the contract MONTH named in the Kalshi terms, not a
    #          continuous front-month series. Near a roll they can differ by $1+.
    raise NotImplementedError


# YOUR TURN
def prob_below(F: float, K: float, sigma: float, T: float) -> float:
    """P(F_T < K).  Formula: 1 − P(F_T > K)   (a continuous price never lands exactly on K).

    Example:
        >>> round(prob_below(70.0, 70.0, 0.35, 10/252), 4)
        0.5139
    """
    # TODO: one line, reusing prob_above.
    raise NotImplementedError


# YOUR TURN
def prob_between(F: float, K_low: float, K_high: float, sigma: float, T: float) -> float:
    """P(K_low < F_T < K_high).  Formula: P(above K_low) − P(above K_high).

    Example:
        >>> round(prob_between(70.0, 69.0, 71.0, 0.35, 10/252), 4)
        0.1623
    """
    # TODO 1: Raise ValueError if K_high <= K_low.
    # TODO 2: Combine two prob_above calls. Sketch a number line if unsure which order.
    raise NotImplementedError


# YOUR TURN
# OOP: HistoricalVol is a small class, not a function, because it must plug into
#      GBMModel's `vol` slot, which expects "anything with .estimate()". Its
#      settings (window, days_per_year) are stored once in __init__, so every
#      later .estimate() call uses them without passing them again.
class HistoricalVol(VolatilityEstimator):
    """Annualized standard deviation of the last `window` daily log returns.

    Formula:  r_t = ln(F_t / F_{t−1}),    σ = std(r_{t−window+1..t}) · √252
    """
    name = "hist"

    def __init__(self, window: int = 20, days_per_year: int = 252):
        self.window = window
        self.days_per_year = days_per_year

    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> float:
        """Example: a series whose daily returns have std 0.02 gives ≈ 0.02 · √252 ≈ 0.317."""
        # TODO 1: Get roll-safe daily log returns: data.curve.nearby_returns(as_of)
        #         (already filtered to available_at < as_of; read it in models/base.py).
        # TODO 2: Raise ValueError if there are fewer than self.window returns.
        # TODO 3: Take the LAST self.window returns.
        # TODO 4: Annualize their sample std (ddof=1).
        recent = ...  # SYNTAX: negative slicing: series.iloc[-n:] is "the last n rows"
        sigma = ...  # HINT: daily std × sqrt(trading days per year)
        # PITFALL: ddof=1 (sample std) vs ddof=0 matters with small windows. pandas
        #          defaults to 1 and numpy to 0, a classic source of mismatches.
        raise NotImplementedError


# OOP: GBMModel is a class because it has a LIFECYCLE: fit() learns sigma and
#      stores data; later prob_terminal()/prob_barrier() use them. The methods
#      live inside the class because they all share that remembered state (self.sigma,
#      self.data) and because PriceModel requires them, so the backtester can treat
#      GBM, Heston and Bates identically.
class GBMModel(PriceModel):
    """Futures price follows geometric Brownian motion with zero drift.

    Model:   dF = σ·F·dW    ⇒    ln F_T ~ Normal( ln F₀ − ½σ²T ,  σ²T )
    """
    name = "gbm"
    version = "1.0"

    def __init__(self, vol: VolatilityEstimator | None = None, n_paths: int = 20_000,
                 seed: int = 0, days_per_year: int = 252):
        super().__init__()   # OOP: run PriceModel.__init__ first (sets self.data, self.fitted_as_of)
        # SYNTAX: `x or default` returns default when x is None (or any "falsy" value).
        self.vol = vol or HistoricalVol()
        self.n_paths = n_paths
        self.days_per_year = days_per_year
        self.rng = np.random.default_rng(seed)  # SYNTAX: modern numpy RNG; seed => reproducible
        self.sigma: float | None = None
        # OOP: `self.vol` is COMPOSITION: GBMModel HAS-A volatility estimator rather
        #      than IS-A one. Swap it at construction: GBMModel(vol=HARVol()) in Stage 6.

    # YOUR TURN
    def fit(self, data: MarketData, as_of: pd.Timestamp) -> None:
        """Estimate sigma with self.vol and remember the data for later predictions."""
        # TODO 1: Store data on self.data and as_of on self.fitted_as_of.
        # TODO 2: Set self.sigma using the volatility estimator.
        # WHY store data instead of a price: predict() is called later with a newer
        #     as_of and must look up the price public at THAT moment.
        raise NotImplementedError

    # YOUR TURN
    def prob_terminal(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """Above / below / between probability for one contract.

        Formula (Black-76):  P(above K) = N(d2);   below = 1 − above;
                             between = above(K_low) − above(K_high)
        """
        # TODO 1: Look up the settle quote for contract.futures_month as of as_of
        #         (self.data.curve.settle_quote -> .price and .available_at).
        # TODO 2: Compute T with year_fraction(as_of, contract.settle_date, self.days_per_year).
        # TODO 3: Ask the estimator for total variance:
        #         self.vol.horizon_variance(self.sigma, T, as_of, contract.settle_date).
        #         Convert it back to an annualized sigma for this horizon (if T > 0).
        # TODO 4: Branch on contract.contract_type and call the matching Stage 1 function.
        # TODO 5: Return self._prediction(contract, p, as_of, F=..., F_available_at=..., sigma=..., T=...)
        quote = ...
        sigma_eff = ...  # HINT: total variance = σ_eff² · T, solve for σ_eff
        # SYNTAX: Python 3.10+ also has `match contract.contract_type: case "above": ...`
        #         An if/elif chain works just as well here.
        # WHY record F_available_at: in Stage 4 a checker uses it to prove no lookahead.
        raise NotImplementedError

    # YOUR TURN (Stage 2)
    def prob_barrier(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """Max / min probability by Monte Carlo, monitoring DAILY settles only.

        "max": P(any settle from now to settle_date is > strike)
        "min": P(any settle from now to settle_date is < strike)
        """
        # TODO 1: Look up the current settle quote (as in prob_terminal).
        # TODO 2: If contract.start_date is set, check the settles already observed
        #         (self.data.curve.history(month, as_of), dates >= start_date).
        #         If the level was already crossed, the answer is 1.0.
        # TODO 3: n_days = trading_days_until(as_of, contract.settle_date). If 0, return 0.0.
        # TODO 4: Simulate self.n_paths paths with n_days steps (one step per settle).
        # TODO 5: Use prob_from_barrier on the paths.
        # PITFALL: Kalshi checks the SETTLE each day, not every tick. One step per
        #          trading day is the correct monitoring, not an approximation.
        raise NotImplementedError


# =============================================================================
# STAGE 2: Monte Carlo
# =============================================================================

# YOUR TURN
def simulate_gbm_paths(F0: float, sigma: float, T: float, n_steps: int, n_paths: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Simulate driftless GBM price paths.

    Exact step:  F_{t+Δt} = F_t · exp( −½σ²Δt + σ√Δt · Z ),   Z ~ N(0,1),  Δt = T / n_steps

    Returns:
        Array of shape (n_paths, n_steps + 1). Column 0 is F0, column -1 is F_T.

    Example:
        >>> paths = simulate_gbm_paths(70.0, 0.3, 1.0, 252, 10_000, np.random.default_rng(0))
        >>> paths.shape, paths[:, -1].mean()   # mean stays ≈ 70 (no drift in price)
        ((10000, 253), 70.0...)
    """
    # TODO 1: dt = T / n_steps.
    # TODO 2: Draw all shocks at once: Z with shape (n_paths, n_steps).
    # TODO 3: Build log-increments, then cumulative-sum them along the time axis.
    # TODO 4: Prepend a column of zeros (time 0) and exponentiate: F0 * exp(log path).
    Z = ...  # SYNTAX: rng.standard_normal((rows, cols)) draws a whole matrix at once
    log_steps = ...  # HINT: the exponent in the docstring formula, applied to the whole matrix
    log_path = ...  # HINT 1: np.cumsum(..., axis=1) adds up steps along each row
                    # HINT 2: np.hstack([np.zeros((n_paths, 1)), ...]) adds the t=0 column
    # WHY vectorize instead of `for path in range(n_paths)`: numpy runs the loop in C,
    #     roughly 100× faster. Write the math for a whole matrix at once.
    raise NotImplementedError


# YOUR TURN
def prob_from_terminal(final_prices: np.ndarray, contract_type: str, strike: float,
                       strike_high: float | None = None) -> float:
    """Fraction of simulated final prices that win the contract.

    Formula:  P̂ = (1/N) Σ 1[condition on F_T]

    Example:
        >>> prob_from_terminal(np.array([60., 70., 80., 90.]), "above", 75.0)
        0.5
    """
    # TODO: build a boolean array for the winning condition, then take its mean.
    # SYNTAX: (arr > K) is a boolean array; .mean() of booleans = fraction True.
    # SYNTAX: combine conditions with & (and) / | (or), with parentheses: (a > 1) & (a < 2)
    raise NotImplementedError


# YOUR TURN
def prob_from_barrier(paths: np.ndarray, contract_type: str, strike: float) -> float:
    """Fraction of paths whose monitored settles cross the level.

    "max": P( max_{t≥1} F_t > K )        "min": P( min_{t≥1} F_t < K )
    Column 0 is today's known price and is NOT a future observation.

    Example:
        >>> p = np.array([[70, 71, 76], [70, 69, 68]])
        >>> prob_from_barrier(p, "max", 75.0)
        0.5
    """
    # TODO 1: Drop column 0.
    # TODO 2: Take the max (or min) along each row (axis=1).
    # TODO 3: Compare to the strike and average.
    raise NotImplementedError


# YOUR TURN
def prob_touch_continuous(F: float, K: float, sigma: float, T: float, direction: str) -> float:
    """Reflection-principle probability of touching K if the price were watched CONTINUOUSLY.

    For X_t = ln(F_t/F) = νt + σW_t (Brownian motion with drift ν) and a barrier b > 0:
        P(max_{s≤T} X_s ≥ b) = N( (−b + νT)/(σ√T) ) + e^{2νb/σ²} · N( (−b − νT)/(σ√T) )
    For a lower barrier −a (a > 0):
        P(min_{s≤T} X_s ≤ −a) = N( (−a − νT)/(σ√T) ) + e^{−2νa/σ²} · N( (−a + νT)/(σ√T) )
    Reference: "first passage time of Brownian motion with drift" / reflection principle.

    Args:
        direction: "max" (K above F) or "min" (K below F).

    Example:
        >>> round(prob_touch_continuous(70.0, 75.0, 0.35, 20/252, "max"), 3)
        0.467
    """
    # TODO 1: If already past the barrier (max with F >= K, or min with F <= K) return 1.0.
    # TODO 2: Work out ν for driftless-in-PRICE GBM. HINT: what is the drift of ln F_t?
    # TODO 3: Compute b (or a) as a log distance, then apply the matching formula.
    # PITFALL: this formula watches every instant. Kalshi only checks daily settles,
    #          so the true probability is LOWER: a path can poke above K mid-day
    #          and fall back before the settle. Using this formula overprices
    #          max/min contracts. Test 2 shows the gap.
    raise NotImplementedError


# YOUR TURN (optional)
def prob_touch_discrete_bgk(F: float, K: float, sigma: float, T: float, n_monitor: int,
                            direction: str) -> float:
    """Broadie–Glasserman–Kou continuity correction for DISCRETE monitoring.

    Idea: a discretely-watched barrier behaves like a continuously-watched barrier
    shifted AWAY from the price by a factor exp(±β σ √Δt), with β ≈ 0.5826 and
    Δt = T / n_monitor. Plug the shifted barrier into prob_touch_continuous.
    Optional reading: Broadie, Glasserman & Kou (1997), "A continuity correction
    for discrete barrier options", Mathematical Finance 7(4).
    """
    # TODO: shift K outward (up for "max", down for "min"), then call prob_touch_continuous.
    raise NotImplementedError
