"""Stage 12 (optional) · Neural-net challenger: a mixture density network for future log returns.

Paper: "How uncertain are oil prices? Bayesian regularized distribution neural
network for forecasting oil prices", Applied Economics (2025).

Idea: instead of assuming a lognormal (GBM) or simulating (Heston/Bates), a
network looks at today's inputs (HAR lags, Stage 10 factors, the horizon) and
outputs a whole DISTRIBUTION for ln(F_T / F_now) as a Gaussian mixture:
    density(y) = Σ_k w_k · Normal(y; μ_k, s_k²),    w_k ≥ 0,  Σ w_k = 1
Then  P(F_T > K) = Σ_k w_k · [1 − Φ((ln(K/F) − μ_k) / s_k)].

JUDGE IT ONLY by out-of-sample Brier skill vs the market (Stage 3/5 tools), not by
price-level error. A net can nail the average price and still get the tails wrong.
Expect it to struggle: a few thousand daily observations is very little data.

# YOUR TURN (research): the paper's "Bayesian regularization" (how the weight
# penalty is set and updated, the architecture, the inputs). The baseline below
# uses plain L2 weight decay (= a fixed Gaussian prior on the weights) as a
# placeholder. Replace it with the paper's method.

Requires PyTorch (pip install torch). The numpy helpers work without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from common.calendar import trading_days_until
from models.base import Contract, MarketData, Prediction, PriceModel
from models.har import har_features

try:  # PROVIDED: torch is only needed for this stage
    import torch
    from torch import nn
except (ImportError, OSError, ValueError):  # pragma: no cover  (not installed or broken install)
    torch, nn = None, None


# =============================================================================
# numpy helpers (no torch needed)
# =============================================================================

# YOUR TURN
def softmax(z: np.ndarray) -> np.ndarray:
    """Turn any real numbers into positive weights that sum to 1 (along the last axis).

    Formula:  w_k = exp(z_k − max z) / Σ_j exp(z_j − max z)
    WHY subtract max z: exp(1000) overflows; the shift cancels in the ratio.
    """
    # SYNTAX: keepdims=True keeps the summed axis so broadcasting divides each row correctly.
    raise NotImplementedError


# YOUR TURN
def mixture_prob_above(log_moneyness: np.ndarray, weights: np.ndarray, means: np.ndarray,
                       scales: np.ndarray) -> np.ndarray:
    """P(Y > x) for a Gaussian mixture, where Y = ln(F_T/F) and x = ln(K/F).

    Formula:  Σ_k w_k · (1 − Φ((x − μ_k) / s_k))

    Args:
        log_moneyness: array of x values (one per strike).
        weights, means, scales: arrays of length K for ONE mixture.
    Returns:
        array, same shape as log_moneyness.
    """
    # SYNTAX: broadcasting: x[:, None] has shape (n, 1); means[None, :] has shape (1, K);
    #         (x[:, None] − means[None, :]) is then (n, K): every strike against every component.
    raise NotImplementedError


# =============================================================================
# torch model
# =============================================================================

if nn is not None:
    # OOP: subclassing nn.Module is how PyTorch models are written. You define the
    #      layers in __init__ and the computation in forward(); PyTorch handles
    #      gradients, parameters, saving/loading. Same inheritance idea as PriceModel.
    class MixtureDensityNet(nn.Module):
        def __init__(self, n_inputs: int, n_components: int = 3, hidden: int = 32):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
            self.head = nn.Linear(hidden, 3 * n_components)   # logits, means, log-scales
            self.K = n_components

        # YOUR TURN
        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Return (weights, means, scales), each of shape (batch, K).

            weights = softmax(logits);  scales = exp(log_scales) + 1e-4  (always positive)
            """
            # SYNTAX: out.split(self.K, dim=1) cuts the last layer's output into three (batch, K) pieces.
            raise NotImplementedError


# YOUR TURN
def mixture_nll(weights: "torch.Tensor", means: "torch.Tensor", scales: "torch.Tensor",
                y: "torch.Tensor") -> "torch.Tensor":
    """Average negative log-likelihood of targets y (batch,) under each row's mixture.

    Formula:  −mean_i log Σ_k w_ik · N(y_i; μ_ik, s_ik²)
    Compute it in LOG space for stability:
        log N = −½((y−μ)/s)² − ln s − ½ ln(2π)
        log Σ_k w_k N_k = logsumexp_k( ln w_k + log N_k )      (torch.logsumexp)
    """
    # PITFALL: computing Σ w·N directly and then log() underflows to log(0) = −inf for
    #          far-out points. logsumexp avoids that.
    raise NotImplementedError


class NeuralNetModel(PriceModel):
    """Mixture-density engine. Terminal contracts only (barrier contracts need path simulation)."""
    name = "nn_mdn"
    version = "0.1"

    def __init__(self, horizons: tuple[int, ...] = (1, 5, 10, 20), n_components: int = 3, hidden: int = 32,
                 epochs: int = 200, lr: float = 0.01, weight_decay: float = 1e-3, seed: int = 0,
                 factor_names: list[str] | None = None):
        super().__init__()
        if torch is None:
            raise ImportError("Stage 12 needs PyTorch: pip install torch")
        self.horizons, self.K, self.hidden = horizons, n_components, hidden
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed
        self.factor_names = factor_names or []
        self.net: MixtureDensityNet | None = None
        self._mu: np.ndarray | None = None      # input standardization (fit on training rows)
        self._sd: np.ndarray | None = None

    # YOUR TURN
    def build_inputs(self, data: MarketData, as_of: pd.Timestamp) -> pd.DataFrame:
        """One row per date (up to as_of): HAR lags of squared returns (+ factors later).

        Columns: rv_d, rv_w, rv_m (scaled ×252 so they're annualized variances).
        (Stage 10 factors: append factor_matrix(data, dates, self.factor_names) here.)
        """
        raise NotImplementedError

    # YOUR TURN
    def build_training_set(self, data: MarketData, as_of: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """(X, y) with one row per (date t, horizon h):
            X = [inputs at t..., h / 252]
            y = ln(F_{t+h} / F_t)   using ONLY settles public before as_of
        """
        # PITFALL: y needs F_{t+h}. Rows whose t+h is not yet public at as_of must be dropped,
        #          or the network trains on the future.
        raise NotImplementedError

    # YOUR TURN
    def fit(self, data: MarketData, as_of: pd.Timestamp) -> None:
        """Standardize inputs, train the network with Adam on mixture_nll for self.epochs (full batch)."""
        # TODO 1: torch.manual_seed(self.seed); X, y = build_training_set(...)
        # TODO 2: self._mu, self._sd = X.mean(0), X.std(0) + 1e-8; standardize X.
        # TODO 3: self.net = MixtureDensityNet(X.shape[1], self.K, self.hidden)
        #         opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # TODO 4: loop epochs: opt.zero_grad(); loss = mixture_nll(*self.net(Xt), yt); loss.backward(); opt.step()
        # WHY weight_decay: it's the placeholder "prior". Without any regularization a net this
        #     size memorizes noise in a few thousand rows.
        raise NotImplementedError

    # YOUR TURN
    def prob_terminal(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """Mixture for (today's inputs, horizon n/252) -> probability via mixture_prob_above."""
        # TODO 1: F quote; n = trading_days_until(...); if n == 0 the outcome is known.
        # TODO 2: x = last row of build_inputs + [n/252], standardized with self._mu/_sd.
        # TODO 3: with torch.no_grad(): w, m, s = self.net(tensor); convert to numpy.
        # TODO 4: above = mixture_prob_above(ln(K/F), ...); below = 1 − above; between = difference.
        raise NotImplementedError

    def prob_barrier(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        raise NotImplementedError("the mixture models the terminal distribution only; use Bates for max/min")
