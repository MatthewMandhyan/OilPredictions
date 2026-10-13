# PROVIDED · Stage 5 checks: run with  pytest tests/test_stage05.py -v
import sqlite3

import numpy as np
import pandas as pd
import pytest

from live.ledger import Ledger, LedgerRow
from live.predict import run_predictions
from live.report import naive_trade, paper_pnl, rolling_skill
from live.resolve import outcome_from_market, resolve
from models.base import Contract, Prediction
from models.gbm import GBMModel, HistoricalVol
from tests.helpers import gbm_prices, make_curve

TS = pd.Timestamp


def _contract(ticker="K1", strike=70.0):
    return Contract(ticker=ticker, contract_type="above", strike=strike, settle_date="2026-10-16",
                    futures_month="2026-12")


def _row(ticker="K1", p=0.6, trade=("yes", 0.5, 1.0)):
    pred = Prediction(ticker, p, "gbm", "1.0", TS("2026-10-12 15:05"), {"F": 70.0, "F_available_at": TS("2026-10-12 14:30")})
    return LedgerRow.from_prediction(pred, _contract(ticker), 0.48, 0.52, trade)


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.sqlite") as led:
        yield led


def test_from_prediction_fields():
    r = _row()
    assert r.mid == pytest.approx(0.50)
    assert (r.trade_side, r.trade_price, r.trade_contracts) == ("yes", 0.5, 1.0)
    assert r.settle_date == "2026-10-16" and r.futures_month == "2026-12"
    assert "F_available_at" in r.inputs_json


def test_from_prediction_without_trade_or_quote():
    pred = Prediction("K1", 0.6, "gbm", "1.0", TS("2026-10-12 15:05"))
    r = LedgerRow.from_prediction(pred, _contract(), None, 0.52)
    assert r.mid is None and r.trade_side is None


def test_append_and_read(ledger):
    ledger.append(_row("K1"))
    ledger.append(_row("K2"))
    df = ledger.read_predictions()
    assert len(df) == 2 and set(df["ticker"]) == {"K1", "K2"}


def test_ledger_rejects_update(ledger):
    ledger.append(_row())
    with pytest.raises(sqlite3.DatabaseError):
        ledger.conn.execute("UPDATE predictions SET probability = 0.99")


def test_ledger_rejects_delete(ledger):
    ledger.append(_row())
    with pytest.raises(sqlite3.DatabaseError):
        ledger.conn.execute("DELETE FROM predictions")


def test_outcomes_are_append_only(ledger):
    ledger.record_outcome("K1", 1)
    with pytest.raises(sqlite3.DatabaseError):
        ledger.conn.execute("UPDATE outcomes SET outcome = 0")
    with pytest.raises(sqlite3.DatabaseError):
        ledger.conn.execute("DELETE FROM outcomes")
    with pytest.raises(sqlite3.DatabaseError):
        ledger.record_outcome("K1", 0)


def test_outcome_from_market():
    assert outcome_from_market({"status": "settled", "result": "yes"}) == 1
    assert outcome_from_market({"status": "finalized", "result": "no"}) == 0
    assert outcome_from_market({"status": "open", "result": ""}) is None
    assert outcome_from_market({"status": "settled", "result": ""}) is None
    assert outcome_from_market({"status": "closed", "result": "yes"}) is None


def test_resolver_handles_unsettled_contracts(ledger):
    for t in ("K1", "K2", "K3"):
        ledger.append(_row(t))
    markets = pd.DataFrame({"ticker": ["K1", "K2"], "status": ["settled", "open"], "result": ["yes", ""]})
    assert resolve(ledger, markets) == 1
    assert set(ledger.unresolved_tickers()) == {"K2", "K3"}
    assert resolve(ledger, markets) == 0, "running twice must not record anything new"


def test_naive_trade():
    assert naive_trade(0.70, 0.55, 0.60) == ("yes", 0.60, 1.0)
    side, price, n = naive_trade(0.30, 0.40, 0.45)
    assert side == "no" and price == pytest.approx(0.60)
    assert naive_trade(0.42, 0.40, 0.45) is None


def test_paper_pnl():
    df = pd.DataFrame({"trade_side": ["yes", "yes", "no", None], "trade_price": [0.4, 0.4, 0.6, np.nan],
                       "trade_contracts": [2.0, 1.0, 1.0, np.nan], "outcome": [1, 0, 0, 1]})
    assert list(paper_pnl(df)) == pytest.approx([1.2, -0.4, 0.4, 0.0])


def test_rolling_skill():
    days = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({"as_of": np.repeat(days, 2), "probability": [0.9, 0.1] * 10,
                       "mid": [0.6, 0.4] * 10, "outcome": [1, 0] * 10})
    out = rolling_skill(df, window_days=3)
    assert list(out.columns) == ["date", "n", "skill"]
    assert list(out["n"]) == [2, 4, 6, 6, 6, 6, 6, 6, 6, 6]
    assert out["skill"].iloc[-1] == pytest.approx(0.9375)


def test_run_predictions_logs_every_contract_and_survives_failures(ledger):
    data = make_curve(gbm_prices(60, sigma=0.3, seed=1))
    as_of = data.curve.df["available_at"].iloc[-1] + pd.Timedelta(minutes=35)
    good = [Contract(ticker=f"G{i}", contract_type="above", strike=k, settle_date="2026-05-01",
                     futures_month="2026-12") for i, k in enumerate((65.0, 70.0, 75.0))]
    bad = Contract(ticker="BAD", contract_type="above", strike=70.0, settle_date="2026-05-01", futures_month="1999-01")
    quotes = pd.DataFrame({"contract": good + [bad], "yes_bid": [0.4] * 4, "yes_ask": [0.45] * 4})
    models = {"gbm": GBMModel(vol=HistoricalVol(20)), "gbm60": GBMModel(vol=HistoricalVol(40))}
    n = run_predictions(models, quotes, data, as_of, ledger)
    assert n == 6, "2 models × 3 good contracts; the bad one is skipped, not fatal"
    assert len(ledger.read_predictions()) == 6
