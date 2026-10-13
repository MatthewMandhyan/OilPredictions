"""Stage 4 · Lookahead guard: prove no prediction used data that wasn't public yet.

Plug it into the backtester:  Backtester(model, data, checks=[assert_no_lookahead])
"""
from __future__ import annotations

import pandas as pd

from common.calendar import settle_time
from models.base import Contract, LookaheadError, Prediction


# YOUR TURN
def assert_no_lookahead(pred: Prediction, contract: Contract) -> None:
    """Raise LookaheadError if any input of `pred` became public at or after pred.as_of.

    Convention: every engine records, in pred.inputs, a key ending in "_available_at"
    for each time-stamped input it used (e.g. "F_available_at").

    Also raise if the contract had ALREADY settled when the prediction was made
    (predicting a known outcome is lookahead too):  pred.as_of >= settle_time(contract.settle_date)

    Example:
        pred.as_of = 2026-10-12 15:05, inputs = {"F_available_at": 2026-10-12 14:30}  -> OK
        pred.as_of = 2026-10-12 15:05, inputs = {"F_available_at": 2026-10-13 14:30}  -> raise
    """
    # TODO 1: Loop over pred.inputs.items(); for keys ending with "_available_at",
    #         compare pd.Timestamp(value) with pred.as_of.
    # TODO 2: Check the settle date rule.
    # SYNTAX: for key, value in d.items(): ...    and    key.endswith("_available_at")
    raise NotImplementedError


# YOUR TURN
def assert_features_point_in_time(features: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Raise LookaheadError if a feature table handed to a model contains rows with available_at >= as_of.

    Use it in tests and in Stage 10 before fitting the factor layer.
    """
    # HINT: boolean mask, then .any()
    raise NotImplementedError
