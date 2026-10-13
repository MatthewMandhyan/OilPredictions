"""Stage 3 · Scoring: is the model actually better than Kalshi, or just lucky?

OOP: this file has NO classes, on purpose. Every metric is a pure function:
     numbers in, number out, nothing to remember. Wrapping these in a class would
     add typing without adding anything. Knowing when NOT to use a class matters too.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # PROVIDED: draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# YOUR TURN
def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between probabilities and 0/1 outcomes. Lower is better.

    Formula:  BS = (1/N) Σ (p_i − y_i)²

    Example:
        >>> brier_score(np.array([0.5, 0.5]), np.array([1, 0]))
        0.25
    """
    # SYNTAX: np.asarray(x, dtype=float) accepts lists, Series or arrays and gives an array back.
    raise NotImplementedError


# YOUR TURN
def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Average negative log-likelihood. Punishes confident wrong answers hard.

    Formula:  LL = −(1/N) Σ [ y_i ln p_i + (1 − y_i) ln(1 − p_i) ]

    Example:
        >>> round(log_loss(np.array([0.9, 0.2]), np.array([1, 0])), 4)
        0.1643
    """
    # TODO 1: Clip p into [eps, 1 − eps] first.
    # PITFALL: without clipping, p = 0 with y = 1 gives ln(0) = −inf and the whole average breaks.
    raise NotImplementedError


# YOUR TURN
def brier_skill(p_model: np.ndarray, p_bench: np.ndarray, y: np.ndarray) -> float:
    """Brier skill vs a benchmark (Kalshi). > 0 means the model beat the market.

    Formula:  BSS = 1 − BS_model / BS_bench

    Example:
        >>> brier_skill(np.array([0.9, 0.1]), np.array([0.6, 0.4]), np.array([1, 0]))
        0.9375
    """
    raise NotImplementedError


# YOUR TURN
def calibration_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Group predictions into probability bins and compare predicted vs actual frequency.

    A calibrated model has p_mean ≈ y_mean in every bin ("when I say 30%, it happens 30% of the time").

    Returns:
        DataFrame with columns [bin, p_mean, y_mean, count], one row per NON-empty bin.

    Example:
        >>> calibration_table(np.array([0.05, 0.08, 0.92]), np.array([0, 0, 1]), n_bins=10)
           bin  p_mean  y_mean  count
        0    0   0.065     0.0      2
        1    9   0.920     1.0      1
    """
    # TODO 1: Build a DataFrame with columns p and y.
    # TODO 2: Assign each p to a bin number 0..n_bins-1 (equal-width bins on [0, 1]).
    # TODO 3: Group by bin and compute mean p, mean y and count.
    # TODO 4: Drop empty bins, reset the index so 'bin' is a column.
    df = ...
    bins = ...  # HINT 1: np.floor(p * n_bins) gives the bin; p = 1.0 would land in bin n_bins...
                # HINT 2: ...so clip it with np.minimum(..., n_bins - 1) and .astype(int)
    # SYNTAX: "named aggregation": df.groupby("bin").agg(p_mean=("p", "mean"), count=("y", "size"))
    raise NotImplementedError


# PROVIDED
def plot_calibration(tables: dict[str, pd.DataFrame], path: str, title: str = "Calibration") -> None:
    """Reliability diagram: one line per model/benchmark. tables = {"gbm": table, "kalshi": table}."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfect")
    for label, t in tables.items():
        ax.plot(t["p_mean"], t["y_mean"], marker="o", label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# YOUR TURN
def strike_distance_sd(K: np.ndarray, F: np.ndarray, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
    """How far the strike is from the price, in standard deviations of ln F_T.

    Formula:  |ln(K / F)| / (σ √T)

    WHY: "$5 away" means very different things for a 1-day and a 3-month contract.
         Measuring in std devs puts all contracts on one scale.

    Example:
        >>> strike_distance_sd(np.array([77.0]), np.array([70.0]), np.array([0.35]), np.array([0.25]))
        array([0.5446...])
    """
    # PITFALL: T = 0 divides by zero. Return np.inf there (np.where or np.divide with `where=`).
    raise NotImplementedError


# YOUR TURN
def add_buckets(df: pd.DataFrame, horizon_edges: list[float], horizon_labels: list[str],
                distance_edges: list[float]) -> pd.DataFrame:
    """Add `horizon_bucket` and `distance_bucket` columns for sliced reports.

    Uses df["horizon_days"] (CALENDAR days to settle) and df["strike_distance_sd"].

    Example:
        horizon_edges=[0, 7, 28, 91, 100000], labels ["1-7d", "1-4w", "1-3m", "3m+"]
        horizon_days 5 -> "1-7d";  horizon_days 30 -> "1-3m"
    """
    # TODO 1: Copy df (never modify the caller's DataFrame).
    # TODO 2: horizon_bucket = pd.cut(df["horizon_days"], bins=horizon_edges, labels=horizon_labels)
    # TODO 3: distance_bucket = pd.cut on strike_distance_sd with distance_edges (default labels are fine).
    # SYNTAX: pd.cut uses intervals that are open on the left: (0, 7] contains 7 but not 0.
    raise NotImplementedError


# PROVIDED
def sliced_report(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Brier skill vs market for each slice. df needs probability, market_price, outcome."""
    # SYNTAX: groupby(...).apply(lambda g: ...) runs a small function on each group.
    #         A lambda is a one-line nameless function: lambda g: len(g) is like def f(g): return len(g)
    return df.groupby(by, observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "brier_model": brier_score(g["probability"], g["outcome"]),
            "brier_market": brier_score(g["market_price"], g["outcome"]),
            "skill_vs_market": brier_skill(g["probability"], g["market_price"], g["outcome"]),
            "log_loss_model": log_loss(g["probability"], g["outcome"]),
        }),
        include_groups=False,
    )


# YOUR TURN
def bootstrap_brier_skill(df: pd.DataFrame, n_boot: int = 1000, block_col: str | None = "date",
                          ci: float = 0.90, seed: int = 0) -> dict[str, float]:
    """Confidence interval for Brier skill vs market, resampling whole DAYS (blocks).

    WHY blocks: all strikes on one day bet on the same oil price, so they win or
    lose together. 10 strikes on one day is closer to ONE piece of evidence than
    ten. Resampling whole days keeps that dependence; resampling rows pretends
    the contracts are independent and gives a falsely narrow interval.

    Algorithm (block bootstrap):
        repeat n_boot times:
            draw len(blocks) blocks WITH replacement
            skill_b = 1 − Σ(model sq. errors in drawn blocks) / Σ(market sq. errors in drawn blocks)
        lo, hi = the (1−ci)/2 and (1+ci)/2 quantiles of skill_b

    Args:
        df: columns probability, market_price, outcome, and block_col.
        block_col: column defining blocks; None means resample individual rows.

    Returns:
        {"point": skill on the full data, "lo": ..., "hi": ...}
    """
    # TODO 1: Point estimate with brier_skill on the full df.
    # TODO 2: Squared errors per row for model and market.
    # TODO 3: Sum them per block (groupby block_col). If block_col is None, each row is its own block.
    # TODO 4: rng = np.random.default_rng(seed); draw indices with rng.integers(0, n_blocks, size=(n_boot, n_blocks)).
    # TODO 5: For each draw, total model error / total market error -> skill. (Vectorize: index the per-block arrays
    #         with the whole index matrix, then .sum(axis=1).)
    # TODO 6: np.quantile for lo/hi.
    # PITFALL: if lo < 0 < hi you have NOT shown the model beats the market, however good the point estimate looks.
    raise NotImplementedError
