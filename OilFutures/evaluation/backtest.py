"""Stage 3 · Walk-forward backtester: replay history exactly as the live system would see it.

The backtester and the live system both call `model.predict(contract, as_of)`,
the SAME code path. A model can't be "good in backtest, different live".

Why backtest P&L is too optimistic (this matters from Stage 7 on):
    Probability scores (Brier) here are honest. Profit is not, for two reasons:
    1. Fees and spread: you can't trade at the mid price. Stage 7 adds both.
    2. Adverse selection on maker orders: a resting order gets filled MOSTLY when
       someone who knows more than you wants the other side. Picture bidding 40¢
       on "WTI above $75". If a headline sends oil down, informed sellers hit your
       40¢ bid within seconds, and you're filled on exactly the trades you'd now
       want to cancel. When oil rallies, nobody sells to you at 40¢, so the trades
       you'd have won never fill. A naive backtest saying "the price touched 40¢,
       so I got filled" counts all the winners and only some of the losers.
       evaluation/fills.py (Stage 7) uses the conservative rule: count a maker fill
       only if a later trade printed THROUGH your price.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from models.base import Contract, MarketData, Prediction, PriceModel
from evaluation.scoring import strike_distance_sd

# SYNTAX: a type alias for "a function taking (Prediction, Contract) and returning None".
Check = Callable[[Prediction, Contract], None]


# YOUR TURN
def walk_forward_windows(dates: Sequence[pd.Timestamp], refit_every: int) -> list[list[pd.Timestamp]]:
    """Split sorted unique dates into consecutive chunks of `refit_every`.

    The model is re-fit at the start of each chunk and predicts every date in it.

    Example:
        >>> walk_forward_windows(["d1", "d2", "d3", "d4", "d5"], 2)
        [['d1', 'd2'], ['d3', 'd4'], ['d5']]
    """
    # SYNTAX: a list comprehension builds a list in one line:
    #         [expression for item in iterable]   e.g. [x * 2 for x in range(3)] -> [0, 2, 4]
    # HINT: range(start, stop, step) with step = refit_every gives each chunk's start index;
    #       slicing dates[i : i + refit_every] gives the chunk.
    raise NotImplementedError


# OOP: why is the backtester a class? It bundles a model, the data and settings
#      (refit_every, checks) that every step of the loop needs. run() can then be
#      called on different contract sets without re-passing everything. And because
#      it only talks to `PriceModel`, it works for GBM, Heston, Bates or the neural
#      net without one line changing. That's polymorphism paying off.
class Backtester:
    def __init__(self, model: PriceModel, data: MarketData, refit_every: int = 20,
                 checks: Sequence[Check] = ()):
        self.model = model
        self.data = data
        self.refit_every = refit_every
        # SYNTAX: functions are objects too. `checks` is a list of functions we
        #         call on every prediction (Stage 4 passes assert_no_lookahead).
        self.checks = list(checks)
        self.fit_log: list[pd.Timestamp] = []   # when fit() was called (useful for tests/debugging)

    # PROVIDED
    @staticmethod
    def _record(pred: Prediction, contract: Contract, row: pd.Series) -> dict:
        """One output row. @staticmethod: needs no `self`; it lives here because it belongs to the backtester."""
        inp = pred.inputs
        rec = {
            "ticker": pred.ticker, "date": row["date"], "as_of": pred.as_of,
            "contract_type": contract.contract_type, "strike": contract.strike,
            "strike_high": contract.strike_high, "settle_date": contract.settle_date,
            "probability": pred.probability, "model": pred.model_name, "version": pred.model_version,
            "F": inp.get("F"), "sigma": inp.get("sigma"), "T": inp.get("T"),
            "horizon_days": (contract.settle_date - pred.as_of.normalize()).days,
        }
        for col in ("market_price", "outcome", "true_prob"):
            if col in row:
                rec[col] = row[col]
        return rec

    # YOUR TURN
    def run(self, contracts: pd.DataFrame) -> pd.DataFrame:
        """Walk forward through time: fit on the past, predict the next window, roll.

        Args:
            contracts: one row per contract snapshot with columns
                contract (Contract), date, as_of, and optionally market_price, outcome.

        Returns:
            DataFrame of records from self._record(...) plus a strike_distance_sd column.

        Algorithm:
            for each window of dates:
                fit_time = the EARLIEST as_of in the window
                model.fit(self.data, fit_time); append fit_time to self.fit_log
                for every contract row in the window:
                    pred = model.predict(contract, row.as_of)
                    run every check in self.checks on (pred, contract)
                    collect self._record(pred, contract, row)
        """
        # TODO 1: Sort contracts by as_of; get the sorted unique dates.
        # TODO 2: windows = walk_forward_windows(dates, self.refit_every)
        # TODO 3: Loop as in the docstring.
        # TODO 4: Build a DataFrame from the records; add strike_distance_sd(strike, F, sigma, T).
        # PITFALL: fitting ONCE on all the data and then predicting every date is the
        #          most common backtest bug. The window's fit must only see the past.
        # SYNTAX: `for _, row in df.iterrows():` loops over rows (`_` = "ignore the index").
        #         Fine here; for heavy number crunching prefer vectorized numpy.
        raise NotImplementedError
