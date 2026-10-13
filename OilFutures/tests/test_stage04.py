# PROVIDED · Stage 4 checks: run with  pytest tests/test_stage04.py -v   (no internet needed)
import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import Backtester
from evaluation.lookahead import assert_features_point_in_time, assert_no_lookahead
from evaluation.simulate import synthetic_backtest_inputs
from models.base import Contract, LookaheadError, MarketData, Prediction, PriceModel
from models.gbm import GBMModel, HistoricalVol
from pipeline.contracts import (ContractMappingError, kalshi_front_month, market_to_contract,
                                parse_futures_month, parse_ticker_date)
from pipeline.fundamentals import cot_available_at, daily_close_available_at, eia_weekly_available_at
from pipeline.futures import build_curve_frame, month_for_position, wti_last_trade_date
from pipeline.kalshi import snapshot_prices
from pipeline.sentiment import gdelt_available_at, trends_available_at

TS = pd.Timestamp


# ---- lookahead --------------------------------------------------------------

def _contract(settle="2026-10-16"):
    return Contract(ticker="X", contract_type="above", strike=70, settle_date=settle, futures_month="2026-12")


def test_lookahead_passes_clean_prediction():
    p = Prediction("X", 0.4, "m", "1", TS("2026-10-12 15:05"), {"F_available_at": TS("2026-10-12 14:30")})
    assert_no_lookahead(p, _contract())


def test_lookahead_catches_future_input():
    p = Prediction("X", 0.4, "m", "1", TS("2026-10-12 15:05"), {"F_available_at": TS("2026-10-13 14:30")})
    with pytest.raises(LookaheadError):
        assert_no_lookahead(p, _contract())


def test_lookahead_catches_input_at_exactly_as_of():
    p = Prediction("X", 0.4, "m", "1", TS("2026-10-12 14:30"), {"F_available_at": TS("2026-10-12 14:30")})
    with pytest.raises(LookaheadError):
        assert_no_lookahead(p, _contract())


def test_lookahead_catches_any_available_at_key():
    p = Prediction("X", 0.4, "m", "1", TS("2026-10-12 15:05"),
                   {"F_available_at": TS("2026-10-12 14:30"), "ovx_available_at": TS("2026-10-13 08:00")})
    with pytest.raises(LookaheadError):
        assert_no_lookahead(p, _contract())


def test_lookahead_catches_prediction_after_settlement():
    p = Prediction("X", 0.4, "m", "1", TS("2026-10-16 15:05"), {})
    with pytest.raises(LookaheadError):
        assert_no_lookahead(p, _contract("2026-10-16"))


def test_feature_table_check():
    f = pd.DataFrame({"available_at": [TS("2026-10-07 10:30"), TS("2026-10-14 10:30")]})
    assert_features_point_in_time(f.iloc[:1], TS("2026-10-12 15:05"))
    with pytest.raises(LookaheadError):
        assert_features_point_in_time(f, TS("2026-10-12 15:05"))


class PeekingModel(PriceModel):
    """A deliberately broken model that reads TOMORROW's settle. The check must catch it."""
    name, version = "peek", "0"

    def fit(self, data, as_of):
        self.data, self.fitted_as_of = data, as_of

    def prob_terminal(self, contract, as_of):
        q = self.data.curve.settle_quote(contract.futures_month, as_of + pd.Timedelta(days=3))
        return self._prediction(contract, 0.5, as_of, F=q.price, F_available_at=q.available_at)

    def prob_barrier(self, contract, as_of):
        raise NotImplementedError


def test_backtest_with_check_catches_peeking_model():
    data, contracts = synthetic_backtest_inputs(n_days=150, seed=2)
    with pytest.raises(LookaheadError):
        Backtester(PeekingModel(), data, checks=[assert_no_lookahead]).run(contracts)


def test_backtest_with_check_passes_gbm():
    data, contracts = synthetic_backtest_inputs(n_days=150, seed=2)
    res = Backtester(GBMModel(vol=HistoricalVol(window=20)), data, checks=[assert_no_lookahead]).run(contracts)
    assert len(res) == len(contracts)


# ---- futures months ---------------------------------------------------------

def test_ice_last_trade_dates():
    assert wti_last_trade_date("2026-12") == TS("2026-11-19")
    assert wti_last_trade_date("2027-01") == TS("2026-12-18")   # Dec 25 holiday -> anchor Dec 24
    assert wti_last_trade_date("2026-08") == TS("2026-07-20")   # Jul 25 is a Saturday
    assert wti_last_trade_date("2026-06") == TS("2026-05-18")   # May 25 is Memorial Day


def test_nymex_rule_is_one_day_later():
    assert wti_last_trade_date("2026-12", business_days_before_25th=3) == TS("2026-11-20")


def test_month_for_position():
    assert month_for_position(TS("2026-11-18"), 1) == "2026-12"
    assert month_for_position(TS("2026-11-19"), 1) == "2026-12"   # last trading day still counts
    assert month_for_position(TS("2026-11-20"), 1) == "2027-01"
    assert month_for_position(TS("2026-11-20"), 2) == "2027-02"


def test_build_curve_frame():
    pos = pd.DataFrame({"date": [TS("2026-11-19")] * 2 + [TS("2026-11-20")] * 2,
                        "position": [1, 2, 1, 2], "settle": [70.0, 70.5, 71.0, 71.4]})
    out = build_curve_frame(pos)
    assert list(out.columns) == ["date", "futures_month", "settle", "available_at"]
    jan = out[out["futures_month"] == "2027-01"].sort_values("date")
    assert list(jan["settle"]) == [70.5, 71.0], "Jan is position 2 then position 1: one continuous series"
    assert (out["available_at"] == out["date"] + pd.Timedelta(hours=14, minutes=30)).all()


# ---- contract mapping -------------------------------------------------------

RULES = ("If the settlement price of the front-month ICE WTI crude oil futures contract (currently the "
         "August 2026 contract) on Jun 26, 2026 is above 75.99, then the market resolves to Yes.")


def test_parse_futures_month():
    assert parse_futures_month(RULES) == "2026-08"
    assert parse_futures_month("the September 2026 contract ... the September 2026 contract") == "2026-09"


@pytest.mark.parametrize("text", ["settles on Jun 26, 2026", "", "August 2026 or September 2026"])
def test_parse_futures_month_fails_loudly(text):
    with pytest.raises(ContractMappingError):
        parse_futures_month(text)


def test_parse_ticker_date():
    assert parse_ticker_date("KXWTI-26JUN2614") == TS("2026-06-26")
    assert parse_ticker_date("KXWTIW-26JUL0214") == TS("2026-07-02")
    with pytest.raises(ContractMappingError):
        parse_ticker_date("NOTATICKER")


def test_kalshi_front_month_roll():
    assert kalshi_front_month(TS("2026-11-16")) == "2026-12"
    assert kalshi_front_month(TS("2026-11-17")) == "2027-01"
    assert kalshi_front_month(TS("2026-06-26")) == "2026-08"   # July expired Jun 19


def _market(**kw):
    m = {"ticker": "KXWTI-26JUN2614-T75.99", "event_ticker": "KXWTI-26JUN2614", "strike_type": "greater",
         "floor_strike": 75.99, "cap_strike": None, "rules_primary": RULES, "rules_secondary": ""}
    m.update(kw)
    return m


def test_market_to_contract_above():
    c = market_to_contract(_market())
    assert (c.contract_type, c.strike, c.futures_month) == ("above", 75.99, "2026-08")
    assert c.settle_date == TS("2026-06-26")


def test_market_to_contract_between_and_below():
    b = market_to_contract(_market(strike_type="between", floor_strike=71.0, cap_strike=71.99))
    assert (b.contract_type, b.strike, b.strike_high) == ("between", 71.0, 71.99)
    lo = market_to_contract(_market(strike_type="less", floor_strike=None, cap_strike=60.0))
    assert (lo.contract_type, lo.strike) == ("below", 60.0)


def test_market_to_contract_falls_back_to_calendar():
    c = market_to_contract(_market(rules_primary="Resolves on the WTI settlement price on Jun 26, 2026."))
    assert c.futures_month == "2026-08"


def test_market_to_contract_rejects_disagreement():
    with pytest.raises(ContractMappingError):
        market_to_contract(_market(rules_primary=RULES.replace("August 2026", "September 2026")))


def test_market_to_contract_rejects_unknown_type():
    with pytest.raises(ContractMappingError):
        market_to_contract(_market(strike_type="functional"))


# ---- available_at rules -----------------------------------------------------

def test_eia_weekly_release():
    assert eia_weekly_available_at(TS("2026-10-02")) == TS("2026-10-07 10:30")
    assert eia_weekly_available_at(TS("2026-10-09")).date() == TS("2026-10-15").date()  # Columbus Day week


def test_cot_release():
    assert cot_available_at(TS("2026-10-06")) == TS("2026-10-09 15:30")


def test_daily_close_is_next_day():
    assert daily_close_available_at(TS("2026-10-09")) == TS("2026-10-12 08:00")
    assert daily_close_available_at(TS("2026-10-12")) > TS("2026-10-12 15:05"), "same-day close is NOT usable at the snapshot"


def test_sentiment_release():
    assert gdelt_available_at(TS("2026-10-09")) == TS("2026-10-10 00:15")
    assert trends_available_at(TS("2026-10-04")) == TS("2026-10-11")


# ---- snapshot prices ---------------------------------------------------------

def test_snapshot_vwap_uses_only_post_settle_window():
    trades = pd.DataFrame({
        "trade_id": list("abcde"), "ticker": ["K"] * 5,
        "time": [TS("2026-10-12 10:00"), TS("2026-10-12 14:40"), TS("2026-10-12 14:50"),
                 TS("2026-10-12 15:20"), TS("2026-10-13 14:36")],
        "yes_price": [0.10, 0.40, 0.50, 0.90, 0.30], "count": [100.0, 10.0, 30.0, 100.0, 5.0],
        "taker_side": ["yes"] * 5,
    })
    out = snapshot_prices(trades, delay_minutes=5, window_minutes=30).sort_values("date").reset_index(drop=True)
    assert len(out) == 2
    assert out.loc[0, "market_price"] == pytest.approx((0.4 * 10 + 0.5 * 30) / 40)
    assert out.loc[0, "n_trades"] == 2 and out.loc[0, "volume"] == 40
    assert out.loc[0, "as_of"] == TS("2026-10-12 15:05")
    assert out.loc[1, "market_price"] == pytest.approx(0.30)
