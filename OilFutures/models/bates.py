"""Stage 9 · Bates: Heston + sudden jumps (OPEC surprises, wars, pipeline outages).

Model:
    d ln F = (−½ v − λ k̄) dt + √v dW₁ + J dN
    dN  : Poisson process with intensity λ (jumps per year)
    J   ~ Normal(μ_J, σ_J²)  (log jump size)
    k̄   = E[e^J − 1] = exp(μ_J + ½σ_J²) − 1      ← the "compensator"
    v follows the Heston variance equation.

Recipe (learnable version):
    1. Flag jump days: returns more than k local standard deviations from the local mean.
    2. Jump parameters from the flagged days: λ, μ_J, σ_J.
    3. Fit the Heston part on returns WITH JUMP DAYS REMOVED. Otherwise a jump inflates θ and ξ.

OOP: BatesModel INHERITS from HestonModel and overrides just two hooks
(_returns_for_fit and _simulate). fit / prob_terminal / prob_barrier are reused
unchanged. super().fit(...) runs the parent's version and then you add to it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from models.base import MarketData
from models.heston import HestonModel, HestonParams


# PROVIDED
# OOP: dataclass inheritance: BatesParams has every HestonParams field PLUS three more,
#      so anything expecting HestonParams (like .feller_ok) still works on it.
@dataclass(frozen=True)
class BatesParams(HestonParams):
    lam: float = 0.0       # jumps per year
    mu_j: float = 0.0      # mean log jump size
    sigma_j: float = 0.0   # std of log jump size

    @property
    def compensator(self) -> float:
        """k̄ = exp(μ_J + ½σ_J²) − 1."""
        return float(np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0)


# YOUR TURN
def detect_jumps(returns: pd.Series, k: float = 4.0, window: int = 60) -> pd.Series:
    """Boolean Series: True where |r_t − m_t| > k · s_t.

    m_t, s_t = rolling mean and std of the PREVIOUS `window` returns (not including t).
    The first `window` rows are False (not enough history).

    WHY exclude day t from its own yardstick: a huge jump inflates the std it's
        compared against and can hide itself.
    """
    # HINT 1: returns.rolling(window).mean() includes today; .shift(1) makes it end yesterday.
    # HINT 2: comparisons with NaN give False, which handles the first rows for you.
    # PITFALL: a centered or full-sample std would use FUTURE returns. Strictly trailing only.
    raise NotImplementedError


# YOUR TURN
def estimate_jump_params(returns: pd.Series, is_jump: pd.Series, days_per_year: int = 252) -> tuple[float, float, float]:
    """(λ, μ_J, σ_J) from flagged days.

    Formulas:
        λ   = (number of jump days / number of days) · 252
        μ_J = mean of returns on jump days
        σ_J = std (ddof=1) of returns on jump days   (0.0 if fewer than 2 jumps)
    No jumps at all -> (0.0, 0.0, 0.0).
    """
    raise NotImplementedError


# YOUR TURN
def remove_jumps(returns: pd.Series, is_jump: pd.Series) -> pd.Series:
    """Drop the jump days (keep the index of the rest)."""
    # SYNTAX: boolean indexing with ~ (not): returns[~is_jump]
    raise NotImplementedError


# YOUR TURN
def simulate_bates_paths(F0: float, p: BatesParams, T: float, n_steps: int, n_paths: int,
                         rng: np.random.Generator, substeps: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Heston full-truncation step (see models/heston.py) plus, in every small step h:

        N ~ Poisson(λ h)                         number of jumps this step (almost always 0 or 1)
        J_total = N · μ_J + √N · σ_J · Z₄        sum of N normal jump sizes
        ln F += J_total − λ k̄ h                  jump plus compensator

    Returns (prices, variances) exactly like simulate_heston_paths.
    """
    # TODO: copy your Heston loop and add the jump lines.
    # SYNTAX: rng.poisson(lam * h, size=n_paths) draws one Poisson count per path.
    # PITFALL: forgetting −λ k̄ h makes the price drift: E[F_T] ≠ F0. The martingale test catches it.
    raise NotImplementedError


class BatesModel(HestonModel):
    name = "bates"
    version = "1.0"

    def __init__(self, jump_k: float = 4.0, jump_window: int = 60, **heston_kwargs):
        # SYNTAX: **heston_kwargs forwards any other keyword (n_paths, seed, ...) to HestonModel.
        super().__init__(**heston_kwargs)
        self.jump_k, self.jump_window = jump_k, jump_window
        self.jumps_: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # YOUR TURN
    def _returns_for_fit(self, data: MarketData, as_of: pd.Timestamp) -> pd.Series:
        """Detect jumps, store (λ, μ_J, σ_J) on self.jumps_, return the jump-free returns."""
        # OOP: HestonModel.fit calls self._returns_for_fit(...). Because `self` is a
        #      BatesModel, Python runs THIS version. The parent never needs to know.
        raise NotImplementedError

    # YOUR TURN
    def fit(self, data: MarketData, as_of: pd.Timestamp) -> None:
        """super().fit(data, as_of), then upgrade self.params to BatesParams with self.jumps_."""
        # SYNTAX: BatesParams(**asdict(self.params), lam=..., mu_j=..., sigma_j=...)
        #         asdict turns a dataclass into a dict; ** spreads it into keyword arguments.
        raise NotImplementedError

    # YOUR TURN
    def _simulate(self, F0: float, T: float, n_steps: int) -> np.ndarray:
        raise NotImplementedError
