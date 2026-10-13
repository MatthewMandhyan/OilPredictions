"""Stage 5 · Run every registered model on today's open Kalshi contracts and log to the ledger.

Schedule it once per trading day, right after the clean snapshot window
(~15:10 ET), e.g. with cron on a machine in ET:
    10 15 * * 1-5  cd /path/to/OilFutures && python -m live.predict
PAPER TRADING ONLY: this records hypothetical trades and never sends orders.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import pandas as pd

from common.calendar import settle_time
from common.config import ROOT, cfg_get, data_path, get_logger, load_config, load_parquet
from live.ledger import Ledger, LedgerRow
from live.report import naive_trade
from models.base import Contract, FuturesCurve, MarketData, Prediction, PriceModel
from models.gbm import GBMModel, HistoricalVol
from pipeline.contracts import ContractMappingError, market_to_contract
from pipeline.futures import fetch_yahoo_settles, yahoo_symbol
from pipeline.kalshi import KalshiClient, markets_to_frame

log = get_logger("predict")

# SYNTAX: a dict of name -> lambda. Each lambda is a tiny "factory" that builds a FRESH
#         model when called: MODEL_REGISTRY["gbm"]() -> GBMModel(...). Add an engine by
#         adding one line. GBM stays forever as the baseline.
MODEL_REGISTRY: dict[str, Callable[[], PriceModel]] = {
    "gbm": lambda: GBMModel(vol=HistoricalVol(window=20)),
    # "gbm_har": lambda: GBMModel(vol=HARVol()),            # Stage 6
    # "bates":   lambda: BatesModel(),                       # Stage 9
}

TradeRule = Callable[[Prediction, pd.Series], "tuple[str, float, float] | None"]


# PROVIDED
def default_trade_rule(pred: Prediction, quote: pd.Series) -> tuple[str, float, float] | None:
    if pd.isna(quote.get("yes_bid")) or pd.isna(quote.get("yes_ask")):
        return None
    return naive_trade(pred.probability, quote["yes_bid"], quote["yes_ask"])


# YOUR TURN
def run_predictions(models: dict[str, PriceModel], quotes: pd.DataFrame, data: MarketData,
                    as_of: pd.Timestamp, ledger: Ledger, trade_rule: TradeRule | None = default_trade_rule) -> int:
    """Fit each model once, predict every quoted contract, append one ledger row per prediction.

    Args:
        models: {name: model instance}
        quotes: rows with columns contract (Contract), yes_bid, yes_ask
        trade_rule: function (pred, quote_row) -> (side, price, contracts) or None

    Returns:
        Number of rows appended.

    Behaviour: if ONE contract fails (missing price, bad month...), log a warning and
    continue with the rest. One bad market must not stop the daily run.
    """
    # TODO 1: for name, model in models.items(): model.fit(data, as_of)
    # TODO 2:   for _, q in quotes.iterrows(): predict, decide the trade, build a LedgerRow, append.
    # TODO 3: Wrap the per-contract part in try/except Exception as e: log.warning(...); continue
    # SYNTAX: try/except/continue keeps a loop alive when one item fails. Catching the broad
    #         `Exception` is OK in a top-level daily job, but log WHAT failed.
    raise NotImplementedError


# PROVIDED
def refresh_live_curve(curve_df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Append recent same-day settles from Yahoo for the months we need (EIA lags)."""
    extra = []
    for m in months:
        try:
            y = fetch_yahoo_settles(yahoo_symbol(m))
        except Exception as e:
            log.warning("no live prices for %s: %s", m, e)
            continue
        y["futures_month"] = m
        y["available_at"] = [settle_time(d) for d in y["date"]]
        extra.append(y)
    if not extra:
        return curve_df
    out = pd.concat([curve_df] + extra, ignore_index=True)
    return out.drop_duplicates(["futures_month", "date"], keep="first")


# PROVIDED
def main() -> None:
    ap = argparse.ArgumentParser(description="Daily live predictions -> ledger (paper only).")
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY))
    args = ap.parse_args()
    cfg = load_config()
    as_of = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    if as_of < settle_time(as_of) + pd.Timedelta(minutes=35):
        log.warning("running before today's settle + snapshot window; F will be yesterday's settle")

    client = KalshiClient.from_config(cfg)
    markets = pd.concat([markets_to_frame(client.markets(s, status="open", include_historical=False))
                         for s in cfg_get(cfg, "kalshi.terminal_series")], ignore_index=True)
    contracts: list[Contract] = []
    for _, m in markets.iterrows():
        try:
            contracts.append(market_to_contract(m, cfg_get(cfg, "futures.kalshi_roll_business_days", 2),
                                                cfg_get(cfg, "futures.expiry_business_days_before_25th", 4)))
        except ContractMappingError as e:
            log.warning("skip %s", e)
    by_ticker = markets.set_index("ticker")
    quotes = pd.DataFrame({"contract": contracts,
                           "yes_bid": [by_ticker.loc[c.ticker, "yes_bid"] for c in contracts],
                           "yes_ask": [by_ticker.loc[c.ticker, "yes_ask"] for c in contracts]})

    curve_df = load_parquet(data_path(cfg, "processed", "wti_curve.parquet"))
    curve_df = refresh_live_curve(curve_df, sorted({c.futures_month for c in contracts}))
    data = MarketData(FuturesCurve(curve_df))
    models = {name: MODEL_REGISTRY[name]() for name in args.models}   # SYNTAX: dict comprehension
    with Ledger(ROOT / cfg_get(cfg, "paths.ledger_db")) as led:
        n = run_predictions(models, quotes, data, as_of, led)
    log.info("logged %d predictions for %d contracts", n, len(contracts))


if __name__ == "__main__":
    main()
