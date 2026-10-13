"""Shared interface for every price engine. Fully PROVIDED: read it before Stage 1.

Two ideas drive this file:

1. WHY `as_of` is on every method.
   A backtest replays history. If any function could see "the latest data", it
   would quietly use tomorrow's prices to predict today, and the backtest would
   look brilliant and be worthless (lookahead bias). By forcing every call to say
   "pretend it is now `as_of`", and by making the only door to price data
   (`FuturesCurve`) filter on `available_at < as_of`, lookahead is hard to
   introduce by accident.

2. WHY every engine shares one interface.
   The backtester, the live ledger and the report only ever call `fit(...)` and
   `predict(...)`. They never ask "is this GBM or Heston?". So you can add a new
   engine without touching any of them, and every engine is scored on exactly
   the same contracts with exactly the same code. That's polymorphism.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

ContractType = Literal["above", "below", "between", "max", "min"]
TERMINAL_TYPES = ("above", "below", "between")
BARRIER_TYPES = ("max", "min")


class LookaheadError(RuntimeError):
    """Raised when something tries to use data that wasn't public yet."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

# PROVIDED
# OOP: a dataclass is a class whose job is to HOLD data. Python writes __init__,
#      __repr__ and __eq__ for you. frozen=True makes it immutable, so a contract
#      can't be changed after creation (it's a fact about the world).
#      kw_only=True forces Contract(ticker=..., strike=...), which is much harder
#      to get wrong than positional arguments.
@dataclass(frozen=True, kw_only=True)
class Contract:
    ticker: str
    contract_type: ContractType
    strike: float                      # "above"/"below"/"max"/"min" level, or LOW edge for "between"
    strike_high: float | None = None   # HIGH edge, only for "between"
    settle_date: pd.Timestamp          # date whose settle decides it (end date for max/min)
    futures_month: str                 # e.g. "2026-12": the WTI contract month it settles on
    start_date: pd.Timestamp | None = None  # max/min only: first monitored settle date
    source: str = "kalshi"

    # OOP: __post_init__ runs right after the auto-generated __init__. Validating
    #      here means a bad Contract can never exist anywhere in the program.
    def __post_init__(self) -> None:
        if self.contract_type not in TERMINAL_TYPES + BARRIER_TYPES:
            raise ValueError(f"unknown contract_type {self.contract_type!r}")
        if self.contract_type == "between":
            if self.strike_high is None or self.strike_high <= self.strike:
                raise ValueError("'between' needs strike_high > strike")
        object.__setattr__(self, "settle_date", pd.Timestamp(self.settle_date).normalize())

    # OOP: @property makes a method look like an attribute: contract.is_barrier
    @property
    def is_barrier(self) -> bool:
        return self.contract_type in BARRIER_TYPES


# PROVIDED
@dataclass(frozen=True)
class Prediction:
    ticker: str
    probability: float
    model_name: str
    model_version: str
    as_of: pd.Timestamp
    # SYNTAX: never write `inputs: dict = {}`; a mutable default would be SHARED by
    #         every instance. field(default_factory=dict) makes a fresh dict each time.
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0) or np.isnan(self.probability):
            raise ValueError(f"probability out of range: {self.probability}")


# PROVIDED
@dataclass(frozen=True)
class SettleQuote:
    """One settle price plus WHEN it became public."""
    price: float
    date: pd.Timestamp
    available_at: pd.Timestamp


# PROVIDED
# OOP: FuturesCurve ENCAPSULATES the price table. Nobody touches the DataFrame
#      directly; they ask `curve.settle(month, as_of)`. Because every price request
#      goes through these few methods, the lookahead filter lives in ONE place
#      and can't be forgotten somewhere else.
class FuturesCurve:
    """Daily WTI settles by contract month, point-in-time.

    Expected columns:
        date          trade date the settle belongs to
        futures_month "YYYY-MM"
        settle        $/bbl
        available_at  when the settle became public (normally date 14:30 ET)
    """

    REQUIRED = ("date", "futures_month", "settle", "available_at")

    def __init__(self, df: pd.DataFrame):
        missing = set(self.REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"FuturesCurve missing columns: {missing}")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["available_at"] = pd.to_datetime(df["available_at"])
        self.df = df.sort_values(["futures_month", "available_at"]).reset_index(drop=True)
        # Pre-split by month so lookups are fast (searchsorted on sorted arrays).
        # SYNTAX: dict comprehension -> {key: value for key, value in iterable}
        self._by_month = {m: g.reset_index(drop=True) for m, g in self.df.groupby("futures_month")}

    def __repr__(self) -> str:
        return f"FuturesCurve({len(self.df)} rows, {self.df['futures_month'].nunique()} months)"

    def months(self) -> list[str]:
        return sorted(self._by_month)

    def settle_quote(self, futures_month: str, as_of: pd.Timestamp) -> SettleQuote:
        """Latest settle of `futures_month` that was public strictly before as_of."""
        g = self._by_month.get(futures_month)
        if g is None:
            raise KeyError(f"no data for futures month {futures_month}")
        i = g["available_at"].searchsorted(pd.Timestamp(as_of), side="left") - 1
        if i < 0:
            raise LookaheadError(f"no {futures_month} settle available before {as_of}")
        row = g.iloc[i]
        return SettleQuote(float(row["settle"]), row["date"], row["available_at"])

    def settle(self, futures_month: str, as_of: pd.Timestamp) -> float:
        return self.settle_quote(futures_month, as_of).price

    def history(self, futures_month: str, as_of: pd.Timestamp) -> pd.Series:
        """All settles of one month public before as_of, indexed by date."""
        g = self._by_month.get(futures_month)
        if g is None:
            raise KeyError(f"no data for futures month {futures_month}")
        g = g[g["available_at"] < as_of]
        return g.set_index("date")["settle"]

    def front_month(self, as_of: pd.Timestamp) -> str:
        """Nearest month with a settle on the latest available date."""
        known = self.df[self.df["available_at"] < as_of]
        if known.empty:
            raise LookaheadError(f"no settles available before {as_of}")
        last_day = known["date"].max()
        return known.loc[known["date"] == last_day, "futures_month"].min()

    def nearby_returns(self, as_of: pd.Timestamp) -> pd.Series:
        """Daily log returns of the nearest month, computed WITHIN each contract month.

        WHY: a "continuous front-month" series jumps on roll days because it
        splices two different contracts together. That jump isn't a real price
        move, and it would inflate any volatility estimate. Here each return is
        ln(settle_t / settle_{t-1}) for the SAME month, then for each date we keep
        the nearest month that has such a return.
        """
        known = self.df[self.df["available_at"] < as_of].sort_values(["futures_month", "date"])
        if known.empty:
            return pd.Series(dtype=float, name="ret")
        known = known.assign(ret=known.groupby("futures_month")["settle"].transform(lambda s: np.log(s).diff()))
        known = known.dropna(subset=["ret"]).sort_values(["date", "futures_month"])
        nearest = known.groupby("date").first()
        return nearest["ret"].rename("ret")


# PROVIDED
@dataclass
class MarketData:
    """Everything a model may look at. Stage 1–9 models only use `curve`.

    features: long table with columns [name, date, value, available_at]
              (EIA, CFTC, FRED, GDELT... built in Stage 4, used from Stage 10).
    """
    curve: FuturesCurve
    features: pd.DataFrame | None = None

    def features_asof(self, times: pd.Series | pd.DatetimeIndex, names: list[str]) -> pd.DataFrame:
        """For each timestamp in `times`, the latest value of each feature public BEFORE it.

        Returns a DataFrame indexed like `times` with one column per name (NaN if
        nothing was public yet).
        """
        times = pd.DatetimeIndex(times)
        out = pd.DataFrame(index=times)
        if self.features is None:
            for n in names:
                out[n] = np.nan
            return out
        left = pd.DataFrame({"t": times}).sort_values("t")
        for n in names:
            right = (self.features[self.features["name"] == n][["available_at", "value"]]
                     .sort_values("available_at"))
            # merge_asof finds, for each t, the last row with available_at < t
            merged = pd.merge_asof(left, right, left_on="t", right_on="available_at",
                                   allow_exact_matches=False, direction="backward")
            out[n] = merged.set_index("t")["value"].reindex(times).values
        return out


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

# PROVIDED
# OOP: an ABSTRACT base class is a contract for subclasses: "anything that calls
#      itself a VolatilityEstimator must have .estimate()". You can't create one
#      directly. GBMModel holds ANY estimator and calls .estimate() without
#      caring which kind it is, so you can swap HistoricalVol -> HARVol -> ImpliedVol
#      without editing GBMModel. (This is called the "strategy pattern".)
class VolatilityEstimator(ABC):
    name: str = "vol"

    @abstractmethod
    def estimate(self, data: MarketData, as_of: pd.Timestamp) -> float:
        """Annualized volatility forecast using only data public before as_of."""

    def horizon_variance(self, sigma: float, T: float, as_of: pd.Timestamp,
                         settle_date: pd.Timestamp) -> float:
        """Total variance of ln(F) between now and settlement.

        Default: sigma^2 * T. Subclasses may add to it. In Stage 6 an event-aware
        estimator adds extra variance for scheduled events inside the window.
        """
        return sigma ** 2 * T

    def __repr__(self) -> str:
        # SYNTAX: vars(self) is the instance's attribute dict; f-strings format values inline.
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


# PROVIDED
class PriceModel(ABC):
    """Every engine (GBM, Heston, Bates, neural net) inherits from this.

    Subclasses MUST implement fit, prob_terminal and prob_barrier.
    They INHERIT predict, which the backtester and live system call.
    """
    name: str = "base"
    version: str = "0.0"

    def __init__(self) -> None:
        self.data: MarketData | None = None
        self.fitted_as_of: pd.Timestamp | None = None

    @abstractmethod
    def fit(self, data: MarketData, as_of: pd.Timestamp) -> None:
        """Estimate parameters using ONLY data public before as_of. Store `data` for predict."""

    @abstractmethod
    def prob_terminal(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """P(above / below / between) at contract.settle_date."""

    @abstractmethod
    def prob_barrier(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        """P(settle touches the level at any monitored date) for max / min contracts."""

    # OOP: predict is written ONCE here and inherited by every engine. Subclasses
    #      fill in the pieces (prob_terminal / prob_barrier); the base class
    #      decides the order things happen in. This is the "template method" idea.
    def predict(self, contract: Contract, as_of: pd.Timestamp) -> Prediction:
        if self.data is None or self.fitted_as_of is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        if as_of < self.fitted_as_of:
            raise LookaheadError(f"{self.name}: predicting at {as_of} with a fit from {self.fitted_as_of}")
        if contract.is_barrier:
            return self.prob_barrier(contract, as_of)
        return self.prob_terminal(contract, as_of)

    # PROVIDED helper so every engine builds Predictions the same way.
    def _prediction(self, contract: Contract, probability: float, as_of: pd.Timestamp,
                    **inputs: Any) -> Prediction:
        # SYNTAX: **inputs collects any keyword arguments into a dict:
        #         self._prediction(c, 0.4, t, F=70.1, sigma=0.3) -> inputs={"F": 70.1, "sigma": 0.3}
        p = float(np.clip(probability, 0.0, 1.0))
        return Prediction(contract.ticker, p, self.name, self.version, as_of, dict(inputs))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"
