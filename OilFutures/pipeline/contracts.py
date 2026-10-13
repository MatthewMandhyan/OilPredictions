"""Stage 4 · Map each Kalshi market to a Contract, including WHICH futures month it settles on.

The single most expensive silent bug in this project is pricing a contract off
the wrong futures month. So this file follows one rule: FAIL LOUDLY. If the
rules text and the expiry calendar disagree, or neither gives an answer, raise
an error instead of guessing.
"""
from __future__ import annotations

import re

import pandas as pd

from common.calendar import add_trading_days
from models.base import Contract
from pipeline.futures import wti_last_trade_date

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MON3 = [m[:3].upper() for m in MONTHS]


class ContractMappingError(ValueError):
    """The contract's futures month or terms could not be determined safely."""


# YOUR TURN
def parse_futures_month(rules_text: str) -> str:
    """Find the futures contract month named in a market's rules text.

    Looks for a full month name followed DIRECTLY by a 4-digit year, e.g. "August 2026".
    (Dates like "June 26, 2026" have a day in between, so they don't match.)

    Returns:
        "YYYY-MM"

    Raises:
        ContractMappingError if no month is found, or if two DIFFERENT months are found.

    Example:
        >>> parse_futures_month("...settlement price of the August 2026 ICE WTI contract on Jun 26, 2026...")
        '2026-08'
    """
    # SYNTAX: regular expressions (the `re` module) find text patterns.
    #         "|".join(MONTHS) builds "January|February|..." meaning "any of these".
    #         r"(January|...)\s+(\d{4})": \s+ = one or more spaces, \d{4} = four digits,
    #         ( ) = capture groups you can read back. re.findall returns a list of tuples.
    # TODO 1: pattern = r"(" + "|".join(MONTHS) + r")\s+(\d{4})"
    # TODO 2: matches = re.findall(pattern, rules_text)
    # TODO 3: Convert each (name, year) to "YYYY-MM": MONTHS.index(name) + 1, formatted with f"{n:02d}".
    # TODO 4: Use a set to find the DISTINCT months; raise unless there is exactly one.
    # SYNTAX: a set comprehension removes duplicates: {f(x) for x in items}
    raise NotImplementedError


# YOUR TURN
def parse_ticker_date(event_ticker: str) -> pd.Timestamp:
    """Settle date encoded in a Kalshi event ticker.

    Format seen in practice: "KXWTI-26JUN2614" -> YY=26, MON=JUN, DD=26, then "14" (14:00/14:30 settle hour).

    Example:
        >>> parse_ticker_date("KXWTI-26JUN2614")
        Timestamp('2026-06-26 00:00:00')

    Raises:
        ContractMappingError if the pattern isn't found.
    """
    # HINT: r"-(\d{2})([A-Z]{3})(\d{2})" captures year, month code, day. MON3.index(code) + 1 is the month.
    # PITFALL: the order is YY MON DD, not DD MON YY. Check against a market page you can read.
    raise NotImplementedError


# YOUR TURN
def kalshi_front_month(settle_date: pd.Timestamp, roll_business_days: int = 2,
                       business_days_before_25th: int = 4) -> str:
    """The month Kalshi SHOULD reference for a settle date, from the calendar alone.

    Rule (VERIFY against current contract terms): use the nearest month, but switch
    to the next one starting `roll_business_days` business days before expiry.
    So month m is used on date d if  d < (last trade date of m) − roll_business_days.

    Examples (ICE expiry for Dec 2026 = 2026-11-19, so the switch day is 2026-11-17):
        >>> kalshi_front_month(pd.Timestamp("2026-11-16"))
        '2026-12'
        >>> kalshi_front_month(pd.Timestamp("2026-11-17"))
        '2027-01'
    """
    # TODO: start at the month after settle_date's month; while the rule fails, move one month later.
    raise NotImplementedError


# YOUR TURN
def market_to_contract(market: pd.Series | dict, roll_business_days: int = 2,
                       business_days_before_25th: int = 4) -> Contract:
    """Build a Contract from one row of pipeline.kalshi.markets_to_frame.

    Mapping of Kalshi `strike_type`:
        "greater" / "greater_or_equal"  -> "above",  strike = floor_strike
        "less" / "less_or_equal"        -> "below",  strike = cap_strike
        "between"                       -> "between", strike = floor_strike, strike_high = cap_strike
        anything else                   -> ContractMappingError
    (max/min series: add their mapping here once you've read their contract terms.)

    Futures month:
        a = parse_futures_month(rules_primary + " " + rules_secondary)   (may raise)
        b = kalshi_front_month(settle_date, ...)
        if the rules text names a month, it MUST equal b, else raise ContractMappingError.
        if the rules text names no month, use b.

    Settle date: parse_ticker_date(event_ticker).
    """
    # TODO 1: Read fields with market["..."] (works for both dict and Series).
    # TODO 2: Map strike_type with a dict lookup (see the table above).
    # TODO 3: Work out settle_date and the futures month as described; raise on disagreement.
    # TODO 4: Return Contract(ticker=..., contract_type=..., ...).
    # SYNTAX: try/except catches an error you expect:
    #         try: a = parse_futures_month(text)
    #         except ContractMappingError: a = None
    # WHY raise instead of picking one: a wrong month gives plausible-looking numbers.
    #     You'd never notice. An exception you can't miss.
    raise NotImplementedError
