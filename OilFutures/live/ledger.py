"""Stage 5 · Append-only prediction ledger (SQLite).

Every live prediction gets written here BEFORE the outcome is known, and is
never changed afterwards. That's what makes your track record trustworthy: you
can't "fix" a bad prediction later, even by accident.

Two tables:
    predictions  one row per (model, contract, time): probability, market quote, paper trade
    outcomes     one row per ticker, written once by live/resolve.py after settlement
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

from models.base import Contract, Prediction


# PROVIDED
@dataclass(frozen=True)
class LedgerRow:
    recorded_at: str
    ticker: str
    contract_type: str
    strike: float
    strike_high: float | None
    settle_date: str
    futures_month: str
    model_name: str
    model_version: str
    as_of: str
    probability: float
    yes_bid: float | None
    yes_ask: float | None
    mid: float | None
    trade_side: str | None        # "yes" | "no" | None
    trade_price: float | None
    trade_contracts: float | None
    inputs_json: str

    # YOUR TURN
    @classmethod
    def from_prediction(cls, pred: Prediction, contract: Contract, yes_bid: float | None,
                        yes_ask: float | None, trade: tuple[str, float, float] | None = None) -> "LedgerRow":
        """Build a row from a Prediction + the contract + the current Kalshi quote.

        OOP: a @classmethod receives the CLASS (cls) instead of an instance, so it can
             build and return a new one: LedgerRow.from_prediction(...). It's the standard
             way to offer a second constructor without cluttering __init__.

        Args:
            trade: optional (side, price, contracts) of the hypothetical paper trade.

        Rules:
            recorded_at = current time, ISO string: pd.Timestamp.now().isoformat()
            timestamps/dates stored as ISO strings (SQLite has no datetime type)
            mid = (bid + ask) / 2 if both exist, else None
            inputs_json = json.dumps(pred.inputs, default=str)  (default=str handles Timestamps)
        """
        # TODO: compute mid, unpack trade (or three Nones), then return cls(...) with every field.
        # SYNTAX: tuple unpacking: side, price, n = trade if trade else (None, None, None)
        raise NotImplementedError


# OOP: the Ledger class ENCAPSULATES the database. Callers get exactly four verbs
#      (append, record_outcome, read_*, close) and no way to update or delete.
#      The SQL triggers you write enforce the same rule inside the database, so even
#      a stray manual query can't rewrite history.
class Ledger:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._create_schema()
        self._install_append_only_guards()

    # SYNTAX/OOP: __enter__/__exit__ let you write `with Ledger(path) as led: ...`
    #             and the connection closes automatically, even after an error.
    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # PROVIDED
    def _create_schema(self) -> None:
        cols = ", ".join(f"{f.name} {'REAL' if f.type in ('float', 'float | None') else 'TEXT'}"
                         for f in fields(LedgerRow))
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols});
            CREATE TABLE IF NOT EXISTS outcomes (
                ticker TEXT PRIMARY KEY, outcome INTEGER NOT NULL, resolved_at TEXT NOT NULL);
        """)
        self.conn.commit()

    # YOUR TURN
    def _install_append_only_guards(self) -> None:
        """Create SQLite triggers that ABORT any UPDATE or DELETE on both tables.

        SQL trigger syntax:
            CREATE TRIGGER IF NOT EXISTS <name> BEFORE UPDATE ON <table>
            BEGIN SELECT RAISE(ABORT, '<message>'); END;
        You need four: (predictions, outcomes) × (UPDATE, DELETE).
        """
        # TODO 1: Build the four CREATE TRIGGER statements.
        #         SYNTAX: a nested loop inside one comprehension:
        #         [f"...{t}...{op}..." for t in ("predictions", "outcomes") for op in ("UPDATE", "DELETE")]
        # TODO 2: Run them with self.conn.executescript("\n".join(statements)); then self.conn.commit().
        # WHY in the database and not just in Python: Python rules only protect
        #     Python code. Triggers also stop a quick "fix" typed into a SQL shell.
        raise NotImplementedError

    # PROVIDED
    def append(self, row: LedgerRow) -> None:
        d = asdict(row)
        sql = f"INSERT INTO predictions ({', '.join(d)}) VALUES ({', '.join('?' * len(d))})"
        self.conn.execute(sql, list(d.values()))   # `?` placeholders: never paste values into SQL text
        self.conn.commit()

    # PROVIDED
    def record_outcome(self, ticker: str, outcome: int) -> None:
        """Insert once. A second insert for the same ticker fails (PRIMARY KEY)."""
        self.conn.execute("INSERT INTO outcomes (ticker, outcome, resolved_at) VALUES (?, ?, ?)",
                          (ticker, int(outcome), pd.Timestamp.now().isoformat()))
        self.conn.commit()

    # PROVIDED
    def read_predictions(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM predictions", self.conn)

    # PROVIDED
    def read_outcomes(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM outcomes", self.conn)

    # PROVIDED
    def read_joined(self) -> pd.DataFrame:
        """Predictions with their outcome (NaN while unresolved)."""
        return pd.read_sql("SELECT p.*, o.outcome FROM predictions p LEFT JOIN outcomes o USING (ticker)",
                           self.conn, parse_dates=["as_of", "recorded_at", "settle_date"])

    # PROVIDED
    def unresolved_tickers(self) -> list[str]:
        q = "SELECT DISTINCT ticker FROM predictions WHERE ticker NOT IN (SELECT ticker FROM outcomes)"
        return [r[0] for r in self.conn.execute(q)]

    def close(self) -> None:
        self.conn.close()
