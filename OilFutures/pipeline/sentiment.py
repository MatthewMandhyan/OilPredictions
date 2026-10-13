"""Stage 4 · Sentiment: GDELT news tone and Google Trends, timestamped point-in-time.

Usage (PROVIDED CLI):  python -m pipeline.sentiment --start 2018-01-01 [--trends path/to/trends.csv]
"""
from __future__ import annotations

import argparse

import pandas as pd
import requests

from common.config import data_path, get_logger, load_config, save_parquet
from pipeline.fundamentals import to_feature_long

log = get_logger("sentiment")


# PROVIDED
def fetch_gdelt_tone(query: str = '"crude oil" OR OPEC', start: str = "2018-01-01",
                     end: str | None = None) -> pd.DataFrame:
    """Daily average news tone from the GDELT DOC 2.0 API (timelinetone mode). Returns [date, value].

    GDELT's DOC API only reaches back a limited window; for long history use GDELT's
    raw files / BigQuery tables (see gdeltproject.org)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    params = {"query": query, "mode": "timelinetone", "format": "json",
              "startdatetime": pd.Timestamp(start).strftime("%Y%m%d000000"),
              "enddatetime": pd.Timestamp(end).strftime("%Y%m%d235959")}
    resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=60)
    resp.raise_for_status()
    points = resp.json()["timeline"][0]["data"]
    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.groupby("date", as_index=False)["value"].mean()


# PROVIDED
def load_google_trends_csv(path: str) -> pd.DataFrame:
    """CSV downloaded by hand from trends.google.com (weekly). Returns [date, value] with date = week START."""
    raw = pd.read_csv(path, skiprows=1)
    raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["value"] = pd.to_numeric(raw["value"].replace("<1", "0.5"), errors="coerce")
    return raw.dropna()


# YOUR TURN
def gdelt_available_at(date: pd.Timestamp) -> pd.Timestamp:
    """A DAILY GDELT average covers the whole day, so it's complete only after midnight.

    Rule: next calendar day at 00:15 (GDELT updates every 15 minutes).

    Example:
        >>> gdelt_available_at(pd.Timestamp("2026-10-09"))
        Timestamp('2026-10-10 00:15:00')
    """
    # WHY not same day: at the 15:05 snapshot, the "daily tone" still includes
    #     tonight's articles. Using it would leak the evening's news.
    raise NotImplementedError


# YOUR TURN
def trends_available_at(week_start: pd.Timestamp) -> pd.Timestamp:
    """Google Trends weekly value (week starts Sunday) is complete after the week ends.

    Rule: week_start + 7 days, at 00:00 (the following Sunday midnight).

    Example:
        >>> trends_available_at(pd.Timestamp("2026-10-04"))
        Timestamp('2026-10-11 00:00:00')
    """
    # PITFALL: Google Trends rescales the WHOLE history to 0–100 every time you download,
    #          relative to the peak in your chosen window. A value from a 2026 download
    #          already "knows" about later peaks. Treat Trends as a weak, rough feature.
    raise NotImplementedError


# PROVIDED
def main() -> None:
    ap = argparse.ArgumentParser(description="Download sentiment features.")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--trends", default=None, help="path to a Google Trends CSV you downloaded")
    args = ap.parse_args()
    cfg = load_config()
    parts = [to_feature_long(fetch_gdelt_tone(start=args.start), "gdelt_tone", gdelt_available_at)]
    if args.trends:
        parts.append(to_feature_long(load_google_trends_csv(args.trends), "trends_oil", trends_available_at))
    feats = pd.concat(parts, ignore_index=True)
    save_parquet(feats, data_path(cfg, "processed", "features_sentiment.parquet"))
    log.info("saved %d feature rows", len(feats))


if __name__ == "__main__":
    main()
