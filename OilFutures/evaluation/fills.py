"""Stage 7 · Fill model: would a paper order REALLY have been filled? And what did it earn?

WHY THIS FILE EXISTS: adverse selection.
    Taker orders (crossing the spread) fill immediately at the quoted price. Easy.
    Maker orders (resting limit orders) are the tricky ones. Who trades against a
    resting order? Often someone who just learned something.

    Example: you bid 40¢ for YES on "WTI above $75 Friday".
      * A headline says OPEC is raising output. Informed traders rush to SELL YES and
        hit your 40¢ bid within seconds. Filled! And now the fair price is 30¢.
      * Instead, oil rallies. Fair price jumps to 55¢. Nobody sells to you at 40¢,
        so you're NOT filled on the trade you'd have won.
    So maker fills are biased toward losers. A backtest rule like "the price touched
    40¢, so I was filled" counts every winner and only some losers, and P&L looks
    great until you go live.

    Conservative rule used here: a resting order counts as filled only if a LATER
    trade printed THROUGH your price (strictly worse for the other side). If someone
    sold at 39¢ while you bid 40¢, the queue at 40¢ must have been used up, so you
    were filled. A trade AT 40¢ might have filled someone ahead of you in line.
"""
from __future__ import annotations

import pandas as pd

from layers.trading_rules import Order


# YOUR TURN
def maker_filled(order: Order, later_trades: pd.DataFrame) -> bool:
    """Did a resting maker order fill, under the trade-through rule?

    later_trades: trades AFTER the order was placed and before you'd cancel it,
                  columns [yes_price, ...] (YES price in $ of each trade).

    Rules:
        YES bid at c:  filled if any later trade has yes_price < c
        NO  bid at c:  same as offering YES at (1 − c), so filled if any yes_price > 1 − c

    Examples:
        YES bid 0.40, later trades at [0.41, 0.40]  -> False   (nobody traded THROUGH 0.40)
        YES bid 0.40, later trades at [0.41, 0.39]  -> True
        NO  bid 0.55 (= offer YES at 0.45), later trades at [0.46] -> True
    """
    # PITFALL: compare with the RIGHT inequality and the right side's price.
    #          A NO price c corresponds to a YES price of 1 − c.
    raise NotImplementedError


# YOUR TURN
def order_pnl(order: Order, outcome: int, fee: float) -> float:
    """Dollar profit of a FILLED order once the contract settles.

    Formula:  contracts · (payout − price) − fee,   payout = outcome for YES, 1 − outcome for NO

    Example: YES, 3 contracts at 0.40, outcome 0, fee 0.06  ->  3 · (0 − 0.40) − 0.06 = −1.26
    """
    raise NotImplementedError
