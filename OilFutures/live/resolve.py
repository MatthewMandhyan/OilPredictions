"""Stage 5 · Fill in outcomes after Kalshi settles a market.

Usage (PROVIDED CLI; run daily, e.g. after 17:00 ET):
    python -m live.resolve --series KXWTI KXWTIW
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.config import ROOT, cfg_get, get_logger, load_config
from live.ledger import Ledger
from pipeline.kalshi import KalshiClient, markets_to_frame

log = get_logger("resolve")


# YOUR TURN
def outcome_from_market(market: pd.Series | dict) -> int | None:
    """1 if the market settled YES, 0 if NO, None if not settled yet (or voided).

    A market counts as settled when market["status"] is "settled" or "finalized"
    AND market["result"] is "yes" or "no". Anything else -> None.

    Examples:
        {"status": "settled", "result": "yes"}  -> 1
        {"status": "open", "result": ""}        -> None
        {"status": "settled", "result": ""}     -> None   (voided/unclear: don't guess)
    """
    # SYNTAX: `x in ("a", "b")` checks membership in a tuple.
    raise NotImplementedError


# YOUR TURN
def resolve(ledger: Ledger, markets: pd.DataFrame) -> int:
    """Record outcomes for every unresolved ledger ticker whose market has settled.

    Args:
        markets: markets_to_frame output (needs ticker, status, result).

    Returns:
        How many outcomes were recorded this run (0 if nothing new settled).
    """
    # TODO 1: unresolved = ledger.unresolved_tickers()
    # TODO 2: Index markets by ticker for quick lookup (markets.set_index("ticker")).
    # TODO 3: For each unresolved ticker that appears in markets: compute the outcome;
    #         if it isn't None, ledger.record_outcome(ticker, outcome) and count it.
    # PITFALL: unsettled markets must be SKIPPED, not recorded as 0. Writing a wrong 0
    #          is permanent, since the ledger is append-only.
    raise NotImplementedError


# PROVIDED
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config()
    client = KalshiClient.from_config(cfg)
    series = args.series or cfg_get(cfg, "kalshi.terminal_series") + cfg_get(cfg, "kalshi.barrier_series")
    markets = pd.concat([markets_to_frame(client.markets(s)) for s in series], ignore_index=True)
    with Ledger(ROOT / cfg_get(cfg, "paths.ledger_db")) as led:
        log.info("resolved %d contracts", resolve(led, markets))


if __name__ == "__main__":
    main()
