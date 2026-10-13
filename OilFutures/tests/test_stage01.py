# PROVIDED · Stage 1 checks: run with  pytest tests/test_stage01.py -v
import numpy as np
import pandas as pd
import pytest

from common.calendar import trading_days_until, year_fraction
from models.base import Contract
from models.gbm import GBMModel, HistoricalVol, prob_above, prob_below, prob_between
from tests.helpers import gbm_prices, make_curve

T10 = 10 / 252
T30 = 30 / 252


# ---- calendar ---------------------------------------------------------------

def test_trading_days_after_settle_skips_today():
    assert trading_days_until(pd.Timestamp("2026-10-12 15:05"), pd.Timestamp("2026-10-16")) == 4


def test_trading_days_before_settle_counts_today():
    assert trading_days_until(pd.Timestamp("2026-10-12 10:00"), pd.Timestamp("2026-10-16")) == 5


def test_trading_days_skips_weekend():
    # Friday after settle -> next Monday = 1 settle left
    assert trading_days_until(pd.Timestamp("2026-10-16 15:05"), pd.Timestamp("2026-10-19")) == 1


def test_trading_days_skips_good_friday():
    # Thu 2026-04-02 after settle -> Mon 2026-04-06: Good Friday (Apr 3) has no settle
    assert trading_days_until(pd.Timestamp("2026-04-02 15:05"), pd.Timestamp("2026-04-06")) == 1


def test_trading_days_never_negative():
    assert trading_days_until(pd.Timestamp("2026-10-20 15:05"), pd.Timestamp("2026-10-16")) == 0


def test_year_fraction():
    assert year_fraction(pd.Timestamp("2026-10-12 15:05"), pd.Timestamp("2026-10-16")) == pytest.approx(4 / 252)


# ---- closed form ------------------------------------------------------------

def test_known_value_at_the_money():
    # ATM is slightly BELOW 0.5: the −½σ²T term puts the median under F.
    assert prob_above(70, 70, 0.35, T10) == pytest.approx(0.4861, abs=1e-4)


def test_known_value_out_of_the_money():
    assert prob_above(70, 75, 0.35, T30) == pytest.approx(0.2638, abs=1e-4)


def test_known_value_between():
    assert prob_between(70, 69, 71, 0.35, T10) == pytest.approx(0.1623, abs=1e-4)


def test_above_plus_below_is_one():
    for K in (55, 65, 70, 80):
        assert prob_above(70, K, 0.35, T30) + prob_below(70, K, 0.35, T30) == pytest.approx(1.0)


def test_monotonic_in_strike():
    probs = [prob_above(70, k, 0.35, T30) for k in np.arange(50, 90, 0.5)]
    assert all(a > b for a, b in zip(probs, probs[1:])), "P(above K) must fall as K rises"


def test_between_ladder_sums_to_one():
    edges = np.arange(40, 101, 2.5)
    total = prob_below(70, edges[0], 0.35, T30) + prob_above(70, edges[-1], 0.35, T30)
    total += sum(prob_between(70, lo, hi, 0.35, T30) for lo, hi in zip(edges, edges[1:]))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_more_volatility_raises_out_of_the_money_probability():
    assert prob_above(70, 80, 0.50, T30) > prob_above(70, 80, 0.25, T30)


def test_no_time_left_gives_certain_answer():
    assert prob_above(70, 69, 0.35, 0.0) == 1.0
    assert prob_above(70, 71, 0.35, 0.0) == 0.0


def test_between_rejects_bad_strikes():
    with pytest.raises(ValueError):
        prob_between(70, 71, 69, 0.35, T10)


# ---- historical vol ---------------------------------------------------------

def test_historical_vol_matches_formula():
    prices = gbm_prices(80, sigma=0.4, seed=1)
    data = make_curve(prices)
    as_of = data.curve.df["available_at"].iloc[-1] + pd.Timedelta(minutes=1)
    r = np.diff(np.log(prices))
    expected = np.std(r[-20:], ddof=1) * np.sqrt(252)
    assert HistoricalVol(window=20).estimate(data, as_of) == pytest.approx(expected, rel=1e-9)


def test_historical_vol_ignores_settles_not_yet_public():
    prices = gbm_prices(80, sigma=0.4, seed=1)
    prices[-1] = prices[-2] * 1.5                      # a huge final move...
    data = make_curve(prices)
    before_last = data.curve.df["available_at"].iloc[-1]  # ...published exactly at this moment
    r = np.diff(np.log(prices))[:-1]
    expected = np.std(r[-20:], ddof=1) * np.sqrt(252)
    assert HistoricalVol(window=20).estimate(data, before_last) == pytest.approx(expected, rel=1e-9)


def test_historical_vol_needs_enough_data():
    data = make_curve(gbm_prices(10, sigma=0.4))
    with pytest.raises(ValueError):
        HistoricalVol(window=20).estimate(data, pd.Timestamp("2027-01-01"))


# ---- GBMModel ---------------------------------------------------------------

def _fitted_model():
    prices = gbm_prices(60, sigma=0.35, seed=3)
    data = make_curve(prices)
    last = data.curve.df.iloc[-1]
    as_of = last["available_at"] + pd.Timedelta(minutes=35)
    model = GBMModel(vol=HistoricalVol(window=20))
    model.fit(data, as_of)
    return model, data, as_of, float(last["settle"])


def test_model_matches_closed_form():
    model, data, as_of, F = _fitted_model()
    settle_date = as_of.normalize() + pd.Timedelta(days=7)
    c = Contract(ticker="T1", contract_type="above", strike=F + 2, settle_date=settle_date, futures_month="2026-12")
    pred = model.predict(c, as_of)
    T = year_fraction(as_of, settle_date)
    assert pred.probability == pytest.approx(prob_above(F, F + 2, model.sigma, T), abs=1e-9)
    assert pred.model_name == "gbm"


def test_model_records_inputs_including_available_at():
    model, data, as_of, F = _fitted_model()
    c = Contract(ticker="T1", contract_type="below", strike=F, settle_date=as_of.normalize() + pd.Timedelta(days=7),
                 futures_month="2026-12")
    pred = model.predict(c, as_of)
    for key in ("F", "F_available_at", "sigma", "T"):
        assert key in pred.inputs, f"record {key} in Prediction.inputs"
    assert pred.inputs["F_available_at"] < as_of


def test_model_between_and_below_consistent():
    model, data, as_of, F = _fitted_model()
    sd = as_of.normalize() + pd.Timedelta(days=14)
    mk = lambda t, k, kh=None: Contract(ticker=t, contract_type=t, strike=k, strike_high=kh,
                                        settle_date=sd, futures_month="2026-12")
    lo, hi = F - 2, F + 2
    p_between = model.predict(mk("between", lo, hi), as_of).probability
    p_below = model.predict(mk("below", lo), as_of).probability
    p_above = model.predict(mk("above", hi), as_of).probability
    assert p_between + p_below + p_above == pytest.approx(1.0, abs=1e-9)


def test_predict_before_fit_raises():
    c = Contract(ticker="T", contract_type="above", strike=70, settle_date="2026-12-01", futures_month="2026-12")
    with pytest.raises(RuntimeError):
        GBMModel().predict(c, pd.Timestamp("2026-11-01"))
