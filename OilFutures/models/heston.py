"""Stage 8 · Heston: volatility itself moves randomly (and is correlated with price).

Model (driftless futures):
    dF = √v · F · dW₁
    dv = κ(θ − v) dt + ξ √v dW₂,      corr(dW₁, dW₂) = ρ

    κ (kappa)  speed variance reverts to its long-run level
    θ (theta)  long-run variance (σ_long = √θ)
    ξ (xi)     volatility of variance ("vol of vol")
    ρ (rho)    price/variance correlation. Negative means "prices fall, vol rises".
    v0         variance today

What it adds over GBM: fat tails and skew (when ρ ≠ 0) in the price distribution,
and a variance that drifts back toward normal after a calm or a panic.

Estimation here is the LEARNABLE version: build a variance proxy from returns and
run regressions on it. Optional advanced methods: maximum likelihood with a
particle filter, or calibrating to option prices. Expect the proxy approach to
UNDERSTATE ξ and |ρ| (the 21-day average smooths away much of the variance's motion).

OOP: this is the first class you shape more yourself. Notice how much of
HestonModel.prob_terminal / prob_barrier looks like GBMModel's Monte Carlo code.
That duplication hints at a shared parent class (e.g. `MonteCarloModel` holding the
path -> probability logic). Refactoring into one is a great optional exercise
once the tests pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.calendar import trading_days_until
from models.base import Contract, MarketData, Prediction, PriceModel, VolatilityEstimator
from models.gbm import prob_from_barrier, prob_from_terminal


# PROVIDED
@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    @property
    def feller_ok(self) -> bool:
        """Feller condition 2κθ > ξ². If it FAILS, the true process can touch v = 0."""
        return 2 * self.kappa * self.theta > self.xi ** 2


# YOUR TURN
def variance_proxy(returns: pd.Series, window: int = 21, days_per_year: int = 252) -> pd.Series:
    """Annualized variance proxy: v̂_t = 252 · mean(r²_{t−window+1..t})."""
    raise NotImplementedError


# YOUR TURN
def estimate_heston_params(returns: pd.Series, window: int = 21, days_per_year: int = 252) -> HestonParams:
    """Regression estimates of κ, θ, ξ, ρ, v0 from daily log returns.

    Discretized variance equation (Δt = 1/252):
        v̂_{t+1} − v̂_t = κθΔt − κΔt · v̂_t + ε_t       ε_t ≈ ξ √(v̂_t Δt) · Z
    So regress  Δv̂ = a + b · v̂   and read off:
        κ = −b / Δt           θ = −a / b           ξ = std( ε / √(v̂ Δt) )
    Correlation:
        ρ = corr( r_{t+1},  v̂_{t+window} − v̂_t )      (variance change over the NEXT window)
    Today's variance:
        v0 = last v̂

    Clean-up: κ at least 0.01; ρ clipped to [−0.99, 0.99]; θ at least 1e-4.
    """
    # TODO 1: v = variance_proxy(returns, window); dv = v.shift(-1) − v
    # TODO 2: Keep rows where both are non-NaN; fit the line (np.polyfit(x, y, 1) -> [slope, intercept]).
    # TODO 3: κ, θ from the slope/intercept; residuals ε = dv − (a + b·v); ξ as above.
    # TODO 4: ρ from the forward variance change (v.shift(-window) − v) and returns.shift(-1).
    # SYNTAX: pd.concat([s1, s2], axis=1).dropna() lines two Series up and drops incomplete rows;
    #         then .corr().iloc[0, 1] is their correlation.
    # PITFALL: ρ is NOT stable in oil (it can flip sign between supply-shock and demand-shock
    #          regimes). Log it every refit: HestonModel.rho_log does that.
    raise NotImplementedError


# YOUR TURN
def simulate_heston_paths(F0: float, p: HestonParams, T: float, n_steps: int, n_paths: int,
                          rng: np.random.Generator, substeps: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler scheme. Prices are recorded at the n_steps monitoring points.

    Per small step h = T / (n_steps · substeps), with v⁺ = max(v, 0):
        Z₁, Z₃ ~ N(0,1) independent;   Z₂ = ρ Z₁ + √(1−ρ²) Z₃
        ln F  += −½ v⁺ h + √(v⁺ h) Z₁
        v     += κ(θ − v⁺) h + ξ √(v⁺ h) Z₂

    Returns:
        prices    shape (n_paths, n_steps + 1), column 0 = F0
        variances shape (n_paths, n_steps + 1), the v⁺ at each monitoring point (never negative)
    Reference: Lord, Koekkoek & van Dijk (2010), "A comparison of biased simulation
    schemes for stochastic volatility models" (full truncation).
    """
    # TODO 1: Arrays for ln F (start ln F0) and v (start v0), one entry per path.
    # TODO 2: Loop over n_steps; inside, loop `substeps` times applying the update (vectorized over paths).
    # TODO 3: After each monitoring step, store exp(lnF) and max(v, 0).
    # WHY truncation: plain Euler can make v negative, and then √v is NaN.
    #     Full truncation uses v⁺ inside the drift and diffusion so nothing breaks.
    # PITFALL: draw Z₁ ONCE per small step and use it for BOTH price and variance (via Z₂).
    #          Drawing them independently silently sets ρ = 0.
    raise NotImplementedError


# YOUR TURN
def rho_history(returns: pd.Series, lookback: int = 252, step: int = 21, window: int = 21) -> pd.Series:
    """ρ re-estimated on a trailing `lookback` window every `step` days. Index = last date used.

    WHY: if ρ flips sign over time, any single estimate is misleading. Plot this.
    """
    # SYNTAX: for end in range(lookback, len(returns) + 1, step): chunk = returns.iloc[end - lookback:end]
    raise NotImplementedError


class HestonModel(PriceModel):
    """Heston engine. Same interface as GBMModel, so the backtester and ledger need no changes."""
    name = "heston"
    version = "1.0"

    def __init__(self, n_paths: int = 20_000, substeps: int = 4, seed: int = 0, window: int = 21,
                 variance_forecaster: VolatilityEstimator | None = None, days_per_year: int = 252):
        super().__init__()
        self.n_paths, self.substeps, self.window, self.days_per_year = n_paths, substeps, window, days_per_year
        self.rng = np.random.default_rng(seed)
        self.variance_forecaster = variance_forecaster   # Stage 10 plugs the factor layer in here
        self.params: HestonParams | None = None
        self.rho_log: list[tuple[pd.Timestamp, float]] = []

    # PROVIDED hook. OOP: Bates OVERRIDES this to strip jumps before fitting.
    def _returns_for_fit(self, data: MarketData, as_of: pd.Timestamp) -> pd.Series:
        return data.curve.nearby_returns(as_of)

    # PROVIDED hook. OOP: Bates OVERRIDES this to add jumps to the paths.
    def _simulate(self, F0: float, T: float, n_steps: int) -> np.ndarray:
        prices, _ = simulate_heston_paths(F0, self.params, T, n_steps, self.n_paths, self.rng, self.substeps)
        return prices

    # YOUR TURN
    def fit(self, data: MarketData, as_of: pd.Timestamp) -> None:
        """Estimate params from self._returns_for_fit(...); optionally replace v0; log ρ.

        If self.variance_forecaster is set: v0 = forecaster.estimate(data, as_of) ** 2
        (the factor layer's better view of today's variance).
        """
        # SYNTAX: dataclasses.replace(params, v0=new_v0) copies a frozen dataclass with one change.
        raise NotImplementedError

    # YOUR TURN
    def prob_terminal(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """Monte Carlo terminal probability with one monitoring step per remaining settle."""
        # TODO 1: quote = self.data.curve.settle_quote(...); n = trading_days_until(as_of, contract.settle_date)
        # TODO 2: if n == 0 the answer is known from F (use prob_from_terminal on np.array([F])).
        # TODO 3: paths = self._simulate(F, n / days_per_year, n); p = prob_from_terminal(paths[:, -1], ...)
        # TODO 4: return self._prediction(..., F=..., F_available_at=..., v0=..., rho=..., n_days=n)
        # OOP: calling self._simulate (not simulate_heston_paths directly) is what lets
        #      Bates reuse this whole method unchanged.
        raise NotImplementedError

    # YOUR TURN
    def prob_barrier(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """Same as GBMModel.prob_barrier but with Heston paths (already-hit check included)."""
        raise NotImplementedError
