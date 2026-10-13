"""Stage 5 · Track record: rolling Brier skill vs Kalshi, calibration, paper P&L.

Usage (PROVIDED CLI):  python -m live.report
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.config import ROOT, cfg_get, data_path, load_config
from evaluation.scoring import bootstrap_brier_skill, brier_skill, calibration_table, plot_calibration
from live.ledger import Ledger


# YOUR TURN
def naive_trade(p: float, yes_bid: float, yes_ask: float) -> tuple[str, float, float] | None:
    """Stage 5 placeholder trade rule: buy one contract when the model disagrees with the quote.

    Rules:
        p > yes_ask  -> buy YES at yes_ask:          ("yes", yes_ask, 1.0)
        p < yes_bid  -> buy NO  at (1 − yes_bid):    ("no", 1 − yes_bid, 1.0)
        otherwise    -> None
    (Stage 7 replaces this with fees, edge thresholds and Kelly sizing.)

    WHY buying NO costs 1 − yes_bid: a NO contract pays $1 if YES loses. Selling YES at
        the bid is economically the same as buying NO at 1 − bid.
    """
    raise NotImplementedError


# YOUR TURN
def paper_pnl(df: pd.DataFrame) -> pd.Series:
    """Profit per row of a resolved ledger frame (fees ignored until Stage 7).

    Columns used: trade_side, trade_price, trade_contracts, outcome.
    Formula:
        YES: contracts · (outcome − price)
        NO : contracts · ((1 − outcome) − price)
        no trade or unresolved: 0

    Example: YES at 0.40 × 2 contracts, outcome 1  ->  2 · (1 − 0.40) = 1.20
    """
    # SYNTAX: np.select(conditions, choices, default) is a vectorized if/elif/else.
    raise NotImplementedError


# YOUR TURN
def rolling_skill(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Brier skill vs the Kalshi mid over a trailing window, for each day with predictions.

    Args:
        df: resolved rows with columns as_of, probability, mid, outcome.

    Returns:
        DataFrame [date, n, skill]: for each date d, use rows with
        d − window_days < date <= d.
    """
    # TODO 1: Add a `date` column (as_of normalized); sort; list the unique dates.
    # TODO 2: For each date d, slice the trailing window and compute brier_skill(probability, mid, outcome).
    # SYNTAX: build a list of dicts in the loop, then pd.DataFrame(list_of_dicts) at the end.
    raise NotImplementedError


# PROVIDED
def main() -> None:
    argparse.ArgumentParser(description="Print the live track record.").parse_args()
    cfg = load_config()
    with Ledger(ROOT / cfg_get(cfg, "paths.ledger_db")) as led:
        df = led.read_joined()
    done = df.dropna(subset=["outcome", "mid"])
    if done.empty:
        print("No resolved predictions yet.")
        return
    done = done.assign(pnl=paper_pnl(done), market_price=done["mid"], date=done["as_of"].dt.normalize())
    for model, g in done.groupby("model_name"):
        print(f"\n=== {model} ({len(g)} resolved) ===")
        print("Brier skill vs Kalshi:", bootstrap_brier_skill(g, n_boot=500))
        print("Paper P&L ($):", round(g["pnl"].sum(), 2), "on", int(g["trade_side"].notna().sum()), "trades")
        print(rolling_skill(g).tail(5).to_string(index=False))
        plot_calibration({model: calibration_table(g["probability"], g["outcome"]),
                          "kalshi": calibration_table(g["mid"], g["outcome"])},
                         str(data_path(cfg, "processed", f"live_calibration_{model}.png")))


if __name__ == "__main__":
    main()
