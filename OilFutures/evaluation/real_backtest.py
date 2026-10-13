"""Stage 4 · Run the Stage 3 backtester with GBM on REAL Kalshi contracts. Fully PROVIDED glue.

Prerequisites (run once):
    python -m pipeline.futures --start 2015-01-01
    python -m pipeline.kalshi --series KXWTI --status settled --trades
Then:
    python -m evaluation.real_backtest --series KXWTI
Outputs a sliced report in the terminal and a calibration plot in data/processed/.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.config import cfg_get, data_path, get_logger, load_config, load_parquet
from evaluation.backtest import Backtester
from evaluation.lookahead import assert_no_lookahead
from evaluation.scoring import (add_buckets, bootstrap_brier_skill, calibration_table, plot_calibration,
                                sliced_report)
from models.base import FuturesCurve, MarketData
from models.gbm import GBMModel, HistoricalVol
from pipeline.contracts import ContractMappingError, market_to_contract
from pipeline.kalshi import snapshot_prices

log = get_logger("real_backtest")


def build_contract_rows(markets: pd.DataFrame, trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (contract, day with a clean post-settle price) for SETTLED markets."""
    snaps = snapshot_prices(trades, cfg_get(cfg, "timing.snapshot_delay_minutes", 5),
                            cfg_get(cfg, "timing.snapshot_window_minutes", 30))
    settled = markets[markets["result"].isin(["yes", "no"])]
    contracts, skipped = {}, 0
    for _, m in settled.iterrows():
        try:
            contracts[m["ticker"]] = market_to_contract(m, cfg_get(cfg, "futures.kalshi_roll_business_days", 2),
                                                        cfg_get(cfg, "futures.expiry_business_days_before_25th", 4))
        except ContractMappingError as e:
            skipped += 1
            log.warning("skip: %s", e)
    log.info("mapped %d contracts, skipped %d", len(contracts), skipped)
    outcome = dict(zip(settled["ticker"], (settled["result"] == "yes").astype(int)))
    rows = snaps[snaps["ticker"].isin(contracts)].copy()
    rows["contract"] = rows["ticker"].map(contracts)
    rows["outcome"] = rows["ticker"].map(outcome)
    # only snapshots strictly before the settle day
    rows = rows[[c.settle_date > d for c, d in zip(rows["contract"], rows["date"])]]
    return rows.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXWTI")
    ap.add_argument("--min-trades", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()

    curve = FuturesCurve(load_parquet(data_path(cfg, "processed", "wti_curve.parquet")))
    markets = load_parquet(data_path(cfg, "raw", f"kalshi_markets_{args.series}.parquet"))
    trades = load_parquet(data_path(cfg, "raw", f"kalshi_trades_{args.series}.parquet"))
    rows = build_contract_rows(markets, trades, cfg)
    rows = rows[rows["n_trades"] >= args.min_trades]
    log.info("%d contract-days to score", len(rows))

    model = GBMModel(vol=HistoricalVol(window=cfg_get(cfg, "backtest.vol_window", 20)))
    bt = Backtester(model, MarketData(curve), refit_every=cfg_get(cfg, "backtest.refit_every_days", 20),
                    checks=[assert_no_lookahead])
    res = bt.run(rows)
    res = add_buckets(res, cfg_get(cfg, "backtest.horizon_buckets_days"), cfg_get(cfg, "backtest.horizon_bucket_labels"),
                      cfg_get(cfg, "backtest.strike_distance_buckets_sd"))

    print("\nOverall:", bootstrap_brier_skill(res, n_boot=cfg_get(cfg, "backtest.bootstrap_n", 1000),
                                              ci=cfg_get(cfg, "backtest.bootstrap_ci", 0.9)))
    for by in ("horizon_bucket", "distance_bucket", "contract_type"):
        print(f"\nBy {by}:\n", sliced_report(res, by).round(4))
    out = data_path(cfg, "processed", f"calibration_{args.series}.png")
    plot_calibration({"gbm": calibration_table(res["probability"], res["outcome"]),
                      "kalshi": calibration_table(res["market_price"], res["outcome"])}, str(out))
    res.to_parquet(data_path(cfg, "processed", f"backtest_gbm_{args.series}.parquet"), index=False)
    log.info("calibration plot -> %s", out)


if __name__ == "__main__":
    main()
