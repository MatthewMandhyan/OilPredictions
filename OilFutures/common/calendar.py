"""Stage 1 · Trading calendar: how much time is left before a contract settles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday,
)

SETTLE_HOUR, SETTLE_MINUTE = 14, 30   # WTI settles at 14:30 ET (see config.timing)


# PROVIDED
# OOP: this is inheritance from a LIBRARY class. pandas already knows how to turn
#      holiday "rules" into dates; we only list which rules apply. We get all of
#      pandas' machinery for free by subclassing instead of writing our own.
class WTIHolidayCalendar(AbstractHolidayCalendar):
    """Approximate US exchange holidays (no settlement on these days)."""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


# PROVIDED: computed once at import. numpy's busday functions want datetime64[D].
HOLIDAYS: np.ndarray = (
    WTIHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31").values.astype("datetime64[D]")
)


# PROVIDED
def is_trading_day(day: pd.Timestamp) -> bool:
    return bool(np.is_busday(np.datetime64(pd.Timestamp(day).date()), holidays=HOLIDAYS))


# PROVIDED
def trading_days(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """All trading days in [start, end]."""
    days = pd.bdate_range(start, end)
    return days[np.is_busday(days.values.astype("datetime64[D]"), holidays=HOLIDAYS)]


# PROVIDED
def add_trading_days(day: pd.Timestamp, n: int) -> pd.Timestamp:
    """Shift a date by n trading days (n can be negative)."""
    d = np.busday_offset(np.datetime64(pd.Timestamp(day).date()), n, roll="forward", holidays=HOLIDAYS)
    return pd.Timestamp(d)


# PROVIDED
def settle_time(day: pd.Timestamp) -> pd.Timestamp:
    """The moment a given day's settlement price becomes public (14:30 ET)."""
    return pd.Timestamp(day).normalize() + pd.Timedelta(hours=SETTLE_HOUR, minutes=SETTLE_MINUTE)


# PROVIDED
def snapshot_time(day: pd.Timestamp, delay_minutes: int = 5, window_minutes: int = 30) -> pd.Timestamp:
    """End of the clean post-settle window: the as_of we use for daily comparisons.

    WHY: with daily data your F is only fresh right after the settle. Comparing
    to Kalshi prices from this window means model and market saw the same F.
    """
    return settle_time(day) + pd.Timedelta(minutes=delay_minutes + window_minutes)


# YOUR TURN
def trading_days_until(as_of: pd.Timestamp, settle_date: pd.Timestamp) -> int:
    """Number of daily settlements still to come, from as_of up to and including settle_date.

    Today's settle counts only if as_of is BEFORE today's 14:30 settle.

    Args:
        as_of: the moment the prediction is made.
        settle_date: the date whose settle decides the contract.

    Returns:
        Integer >= 0.

    Example:
        >>> trading_days_until(pd.Timestamp("2026-10-12 15:05"), pd.Timestamp("2026-10-16"))
        4      # Tue, Wed, Thu, Fri settles remain
        >>> trading_days_until(pd.Timestamp("2026-10-12 10:00"), pd.Timestamp("2026-10-16"))
        5      # Monday's settle hasn't happened yet
    """
    # TODO 1: Find the first calendar day whose settle is still in the future.
    #         That is as_of's date, or the next day if as_of is at/after settle_time(as_of).
    # TODO 2: Count trading days from that day through settle_date, INCLUSIVE.
    # TODO 3: Never return a negative number.



    first = as_of
    if as_of >= settle_time(as_of):
        first = as_of.normalize() + pd.Timedelta(days = 1)
    else:
        first = as_of.normalize()
             # HINT: compare as_of with settle_time(as_of); pd.Timedelta(days=1) moves a day
    n = np.busday_count(first, np.datetime64(settle_date(as_of) + pd.Timedelta(days = 1)), holidays=HOLIDAYS)  # HINT 1: np.busday_count(begin, end, holidays=HOLIDAYS) counts days in [begin, end)
             # HINT 2: that interval is half-open, so to INCLUDE settle_date pass the day after it.
             # SYNTAX: numpy wants dates, not Timestamps: use `.date()` or np.datetime64(...)

    
    raise NotImplementedError


# YOUR TURN
def year_fraction(as_of: pd.Timestamp, settle_date: pd.Timestamp, days_per_year: int = 252) -> float:
    """Time to settlement in years of TRADING time.

    Formula:  T = (trading days remaining) / 252

    Example:
        >>> year_fraction(pd.Timestamp("2026-10-12 15:05"), pd.Timestamp("2026-10-16"))
        0.015873...   # 4 / 252
    """
    # WHY trading days, not calendar days: volatility comes from trading. A weekend
    #     adds 2 calendar days but almost no price movement.
    # SYNTAX: `days_per_year: int = 252` is a default argument; callers may omit it.
    raise NotImplementedError
