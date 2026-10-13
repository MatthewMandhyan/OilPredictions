"""Stage 4 · Fundamentals: EIA inventories, CFTC positioning, FRED (dollar, rates, OVX), event calendar.

Every output row carries TWO times:
    date          the period the number DESCRIBES (e.g. week ending Friday)
    available_at  when the number became PUBLIC (e.g. the following Wednesday 10:30 ET)
A prediction at time t may only use rows with available_at < t.

Output format (long): columns [name, date, value, available_at]. That is what
MarketData.features expects (see models/base.py).

Usage (PROVIDED CLI):  python -m pipeline.fundamentals --start 2015-01-01
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

from common.calendar import add_trading_days, is_trading_day
from common.config import cfg_get, data_path, env_key, get_logger, load_config, save_parquet

log = get_logger("fundamentals")

# Federal holidays shift EIA and CFTC release days (Columbus Day too, unlike exchanges).
FEDERAL_HOLIDAYS = set(USFederalHolidayCalendar().holidays("2000-01-01", "2040-12-31"))


# PROVIDED
def is_federal_business_day(day: pd.Timestamp) -> bool:
    day = pd.Timestamp(day).normalize()
    return day.weekday() < 5 and day not in FEDERAL_HOLIDAYS


# =============================================================================
# Fetchers (PROVIDED). Each returns [date, value].
# =============================================================================

# PROVIDED
def fetch_eia_weekly(api_key: str, series: str = "WCESTUS1", route: str = "petroleum/stoc/wstk",
                     start: str = "2015-01-01") -> pd.DataFrame:
    """EIA weekly series. WCESTUS1 = US commercial crude stocks ex-SPR (thousand bbl).
    Other useful ones: WCRFPUS2 (production), WGTSTUS1 (gasoline stocks). VERIFY ids on eia.gov."""
    params = {"api_key": api_key, "frequency": "weekly", "data[0]": "value", "facets[series][]": series,
              "start": start, "length": 5000, "sort[0][column]": "period", "sort[0][direction]": "asc"}
    resp = requests.get(f"https://api.eia.gov/v2/{route}/data/", params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["response"]["data"])
    return pd.DataFrame({"date": pd.to_datetime(df["period"]), "value": pd.to_numeric(df["value"])})


# PROVIDED
def fetch_cftc_managed_money(start: str = "2015-01-01", market_code: str = "067651") -> pd.DataFrame:
    """CFTC disaggregated futures-only COT for NYMEX WTI (code 067651): managed money net long
    as a fraction of open interest. Dataset id / field names: VERIFY on publicreporting.cftc.gov."""
    url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    params = {"cftc_contract_market_code": market_code, "$limit": 50000,
              "$where": f"report_date_as_yyyy_mm_dd >= '{start}'"}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    long_ = pd.to_numeric(df["m_money_positions_long_all"])
    short = pd.to_numeric(df["m_money_positions_short_all"])
    oi = pd.to_numeric(df["open_interest_all"])
    return pd.DataFrame({"date": pd.to_datetime(df["report_date_as_yyyy_mm_dd"]),
                         "value": (long_ - short) / oi}).sort_values("date").reset_index(drop=True)


# PROVIDED
def fetch_fred(api_key: str, series_id: str, start: str = "2015-01-01") -> pd.DataFrame:
    """FRED daily series. Useful: OVXCLS (oil implied vol), DTWEXBGS (broad dollar), DGS10, DGS2."""
    resp = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=60,
                        params={"series_id": series_id, "api_key": api_key, "file_type": "json",
                                "observation_start": start})
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["observations"])
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]), "value": pd.to_numeric(df["value"], errors="coerce")})
    return out.dropna().reset_index(drop=True)


# PROVIDED
def load_events(path: str) -> pd.DataFrame:
    """Hand-maintained event calendar CSV with columns: date, event_type[, announced_at].

    event_type examples: OPEC, FOMC, EIA_WPSR. announced_at = when the DATE of the event
    was public (OPEC meetings get moved; FOMC dates are known a year ahead).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    if "announced_at" in df:
        df["announced_at"] = pd.to_datetime(df["announced_at"])
    else:
        df["announced_at"] = df["date"] - pd.Timedelta(days=30)  # assumption: known a month ahead
    return df


# =============================================================================
# available_at rules (YOUR TURN)
# =============================================================================

# YOUR TURN
def eia_weekly_available_at(week_ending: pd.Timestamp) -> pd.Timestamp:
    """When an EIA Weekly Petroleum Status Report value became public.

    Rule: data for the week ending Friday is released the following WEDNESDAY at 10:30 ET.
          If a federal holiday falls Monday–Wednesday of release week, release slips one day
          (Thursday). (EIA sometimes uses 11:00 on delayed weeks; 10:30 is fine for learning.)

    Examples:
        >>> eia_weekly_available_at(pd.Timestamp("2026-10-02"))   # Fri
        Timestamp('2026-10-07 10:30:00')                           # Wed
        >>> eia_weekly_available_at(pd.Timestamp("2026-10-09"))   # Columbus Day Mon 10-12
        Timestamp('2026-10-15 10:30:00')                           # Thu
    """
    # TODO 1: release = week_ending + 5 days (Friday -> Wednesday).
    # TODO 2: If any of Mon/Tue/Wed of that week is NOT a federal business day, add one day.
    # TODO 3: Add 10:30.
    # SYNTAX: any(...) with a generator: any(not is_federal_business_day(d) for d in days)
    # PITFALL: using `date` instead of `available_at` gives your model inventory
    #          numbers 5 days early. Backtests love that; live trading doesn't.
    raise NotImplementedError


# YOUR TURN
def cot_available_at(report_date: pd.Timestamp) -> pd.Timestamp:
    """CFTC Commitments of Traders: positions as of TUESDAY are published FRIDAY 15:30 ET.

    If Friday isn't a federal business day, use the next federal business day.

    Example:
        >>> cot_available_at(pd.Timestamp("2026-10-06"))
        Timestamp('2026-10-09 15:30:00')
    """
    raise NotImplementedError


# YOUR TURN
def daily_close_available_at(date: pd.Timestamp) -> pd.Timestamp:
    """A daily CLOSING value (OVX close, dollar index, yields) dated `date`.

    Our snapshot is at ~15:05 ET, BEFORE the 16:00+ closes. So a day-d close is only
    usable from the NEXT trading day. Conservative rule: next trading day at 08:00 ET.

    Example:
        >>> daily_close_available_at(pd.Timestamp("2026-10-09"))   # Friday
        Timestamp('2026-10-12 08:00:00')                            # Monday
    """
    # HINT: add_trading_days(date, 1) from common.calendar.
    # PITFALL: FRED dates the OVX close by the day it was MEASURED. Matching it to a
    #          15:05 snapshot on the same day uses a number that didn't exist yet.
    raise NotImplementedError


# PROVIDED
def to_feature_long(df: pd.DataFrame, name: str, available_at_fn) -> pd.DataFrame:
    """[date, value] + a rule -> [name, date, value, available_at]."""
    out = df[["date", "value"]].copy()
    out["name"] = name
    # SYNTAX: .map(f) applies a function to every element of a column.
    out["available_at"] = out["date"].map(available_at_fn)
    return out[["name", "date", "value", "available_at"]]


# PROVIDED
def main() -> None:
    ap = argparse.ArgumentParser(description="Download fundamentals into one long feature table.")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    cfg = load_config()
    parts = [to_feature_long(fetch_eia_weekly(env_key(cfg, "eia_env"), start=args.start),
                             "eia_crude_stocks", eia_weekly_available_at),
             to_feature_long(fetch_cftc_managed_money(args.start), "cot_mm_net", cot_available_at)]
    fred_key = env_key(cfg, "fred_env")
    for sid in ("OVXCLS", "DTWEXBGS", "DGS10", "DGS2"):
        parts.append(to_feature_long(fetch_fred(fred_key, sid, args.start), sid, daily_close_available_at))
    feats = pd.concat(parts, ignore_index=True)
    save_parquet(feats, data_path(cfg, "processed", "features_fundamentals.parquet"))
    log.info("saved %d feature rows", len(feats))


if __name__ == "__main__":
    main()
