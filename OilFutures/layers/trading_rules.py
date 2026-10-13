"""Stage 7 · Trading rules: when to trade, which side, at what price, how big.

Your first paper-tradable system is GBM (+ HAR / events) + these rules.

Design choices and where they come from:
  * Minimum edge AFTER fees: a 2-point edge is worthless if fees eat 2 points.
  * Fractional Kelly sizing: full Kelly is optimal only if your probabilities are
    exactly right. They aren't, so bet a fraction (e.g. 1/4).
  * Maker orders by default, and extra edge required on cheap contracts:
    Bürgi, Deng & Whelan (2025), "Makers and Takers: The Economics of the Kalshi
    Prediction Market" (CEPR DP20631) found makers earned more than takers, and
    cheap (longshot) contracts were overpriced on average.
  * Caps per position and per day: strikes on one day are one bet on one oil price.

PITFALL: tune min_edge / kelly_fraction by WALK-FORWARD validation
         (walk_forward_tune below), never by picking whatever made the most money
         over the whole backtest. That just fits noise.
PAPER ONLY: an Order here is a record of what you WOULD do. Nothing is sent anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from common.config import cfg_get


# PROVIDED
@dataclass(frozen=True)
class Order:
    ticker: str
    side: str          # "yes" or "no"
    price: float       # price paid per contract for THAT side ($)
    contracts: int
    maker: bool
    edge: float        # expected profit per contract after fees, in $
    prob_win: float    # model probability that THIS side wins


# OOP: TradingRules is a dataclass WITH methods: the settings (data) and the
#      decisions that use them live together. Every method reads self.min_edge,
#      self.bankroll etc. instead of taking ten arguments. `replace(rules, min_edge=0.05)`
#      makes a modified copy, handy for tuning.
@dataclass(frozen=True)
class TradingRules:
    bankroll: float = 1000.0
    min_edge: float = 0.03
    kelly_fraction: float = 0.25
    max_position_frac: float = 0.02
    max_daily_exposure_frac: float = 0.10
    cheap_threshold: float = 0.15
    cheap_extra_edge: float = 0.03
    maker_default: bool = True
    taker_rate: float | None = None
    maker_rate: float | None = None

    # PROVIDED
    @classmethod
    def from_config(cls, cfg: dict) -> "TradingRules":
        t = cfg_get(cfg, "trading", {})
        return cls(bankroll=t["bankroll"], min_edge=t["min_edge"], kelly_fraction=t["kelly_fraction"],
                   max_position_frac=t["max_position_frac"], max_daily_exposure_frac=t["max_daily_exposure_frac"],
                   cheap_threshold=t["cheap_threshold"], cheap_extra_edge=t["cheap_extra_edge"],
                   maker_default=t["maker_default"], taker_rate=cfg_get(cfg, "fees.taker_rate"),
                   maker_rate=cfg_get(cfg, "fees.maker_rate"))

    # YOUR TURN
    def fee(self, price: float, contracts: float, maker: bool) -> float:
        """Kalshi-style trading fee in dollars, rounded UP to the cent.

        Formula:  fee = ceil_to_cent( rate · C · P · (1 − P) )      (VERIFY on Kalshi's fee schedule)

        Raises ValueError if the needed rate is None (still a placeholder in config.yaml).

        Example (rate 0.07, P = 0.50, C = 10):  0.07 · 10 · 0.25 = 0.175  ->  0.18
        """
        # HINT: ceil to the cent = math.ceil(x * 100) / 100. Subtract a tiny 1e-9 first
        #       so floating-point noise (0.1750000001) doesn't round up an extra cent.
        # WHY P·(1−P): the fee is largest for 50¢ contracts and shrinks toward 0¢ / $1.
        raise NotImplementedError

    # YOUR TURN
    def required_edge(self, price: float) -> float:
        """min_edge, plus cheap_extra_edge when price < cheap_threshold."""
        raise NotImplementedError

    # YOUR TURN
    @staticmethod
    def kelly_binary(prob_win: float, price: float) -> float:
        """Full-Kelly fraction of bankroll for a contract costing `price` that pays $1.

        Formula:  f* = (q − c) / (1 − c),   q = win probability, c = price.  Never negative.

        Example: q = 0.60, c = 0.50  ->  0.20
        """
        # OOP: @staticmethod: no `self` needed. It lives on the class because it belongs to sizing.
        raise NotImplementedError

    # YOUR TURN
    def decide(self, ticker: str, p_yes: float, yes_bid: float, yes_ask: float,
               maker: bool | None = None) -> Order | None:
        """Pick the better side (if any) and size it.

        Prices you'd pay:
            maker (post and wait):  YES at yes_bid,       NO at 1 − yes_ask
            taker (cross now):      YES at yes_ask,       NO at 1 − yes_bid
        For each side with price c and win probability q (p_yes for YES, 1 − p_yes for NO):
            edge = q − c − fee_per_contract,   fee_per_contract = rate · c · (1 − c)  (unrounded)
            tradable if edge >= required_edge(c)
        Take the side with the larger edge. Size:
            budget = min(kelly_fraction · kelly_binary(q, c) · bankroll,  max_position_frac · bankroll)
            contracts = floor(budget / c);  no order if contracts < 1.

        Returns:
            Order or None.
        """
        # TODO 1: maker = self.maker_default if maker is None.
        # TODO 2: Build the two candidate (side, c, q) tuples.
        # TODO 3: Compute edges; keep tradable ones; pick the max (or return None).
        # TODO 4: Size it; return Order(...).
        # SYNTAX: max(candidates, key=lambda x: x[3]) picks the tuple with the largest 4th item.
        raise NotImplementedError

    # YOUR TURN
    def apply_daily_cap(self, orders: list[Order]) -> list[Order]:
        """Shrink a day's orders so total cost (Σ price · contracts) <= max_daily_exposure_frac · bankroll.

        If over the cap, multiply every order's contracts by (cap / total), round DOWN,
        and drop orders that end up with 0 contracts.

        WHY: five strikes on the same day all bet on one oil price. Five "independent"
             Kelly bets on one outcome is a single oversized bet.
        """
        # SYNTAX: dataclasses.replace(order, contracts=new_n) copies a frozen dataclass with one field changed.
        raise NotImplementedError


# PROVIDED
def evaluate_rules(rows: pd.DataFrame, rules: TradingRules, maker: bool = False) -> pd.DataFrame:
    """Paper-trade a backtest frame (probability, market bid/ask, outcome, date) with TAKER fills.

    rows needs: ticker, date, probability, yes_bid, yes_ask, outcome.
    Returns one row per order with pnl (after fees). Maker fills need trade data: see evaluation/fills.py.
    """
    from evaluation.fills import order_pnl
    out = []
    for day, g in rows.groupby("date"):
        orders = [rules.decide(r.ticker, r.probability, r.yes_bid, r.yes_ask, maker=maker) for r in g.itertuples()]
        kept = rules.apply_daily_cap([o for o in orders if o is not None])
        outcome = dict(zip(g["ticker"], g["outcome"]))
        for o in kept:
            fee = rules.fee(o.price, o.contracts, o.maker)
            out.append({"date": day, "ticker": o.ticker, "side": o.side, "price": o.price,
                        "contracts": o.contracts, "pnl": order_pnl(o, outcome[o.ticker], fee)})
    return pd.DataFrame(out, columns=["date", "ticker", "side", "price", "contracts", "pnl"])


# YOUR TURN
def walk_forward_tune(rows: pd.DataFrame, base: TradingRules, min_edge_grid: list[float],
                      block_days: int = 60) -> pd.DataFrame:
    """Choose min_edge walk-forward: pick it on the PREVIOUS block, apply it to the NEXT.

    Algorithm:
        split the sorted unique dates into consecutive blocks of `block_days` dates
        block 0: use base.min_edge (no past to learn from)
        block k: for each candidate e in the grid, evaluate_rules(block k−1, replace(base, min_edge=e))
                 and score it by total log growth  Σ log(1 + pnl / bankroll);
                 pick the best e, then evaluate block k with it  (that result is OUT-OF-SAMPLE)

    Returns:
        DataFrame [block, start_date, chosen_min_edge, oos_pnl, n_trades]

    WHY log growth and not total profit: log growth punishes big drawdowns, which is
        what actually ends a trading account.
    PITFALL: the choice for block k must not depend on ANY data from block k or later.
             The test changes future outcomes and checks your earlier choices don't move.
    """
    # SYNTAX: reuse walk_forward_windows from evaluation.backtest for the blocks.
    raise NotImplementedError
