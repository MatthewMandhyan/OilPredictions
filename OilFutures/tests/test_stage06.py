# PROVIDED · Stage 6 checks: run with  pytest tests/test_stage06.py -v
import numpy as np
import pandas as pd
import pytest

from common.calendar import settle_time, trading_days
from models.base import Contract, FuturesCurve, MarketData
from models.gbm import GBMModel, HistoricalVol
from models.har import (EventAwareVol, HARVol, ImpliedVol, event_extra_variance, fit_har, har_features,
                        har_target)

TS = pd.Timestamp


def _curve_from_returns(r, start="2018-01-02"):
    days = trading_days(start, TS(start) + pd.Timedelta(days=int(len(r) * 1.6) + 40))[: len(r) + 1]
    prices = 70 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    df = pd.DataFrame({"date": days, "futures_month": "2099-12", "settle": prices,
                       "available_at": [settle_time(d) for d in days]})
    return MarketData(FuturesCurve(df)), days


def _regime_returns(n_blocks=12, block=120, lo=0.20, hi=0.60, seed=0):
    rng = np.random.default_rng(seed)
    sig = np.concatenate([np.full(block, lo if i % 2 == 0 else hi) for i in range(n_blocks)]) / np.sqrt(252)
    return rng.standard_normal(len(sig)) * sig, sig * np.sqrt(252)


def test_har_features():
    rv = pd.Series(np.arange(1, 31, dtype=float))
    X = har_features(rv)
    assert list(X.columns) == ["rv_d", "rv_w", "rv_m"]
    assert X["rv_w"].iloc[4] == pytest.approx(3.0)       # mean(1..5)
    assert np.isnan(X["rv_m"].iloc[20]) and X["rv_m"].iloc[21] == pytest.approx(11.5)


def test_har_features_use_no_future():
    rv = pd.Series(np.random.default_rng(0).random(60))
    a = har_features(rv)
    rv2 = rv.copy()
    rv2.iloc[40:] = 999.0                                # change the future...
    b = har_features(rv2)
    pd.testing.assert_frame_equal(a.iloc[:40], b.iloc[:40])   # ...the past features must not move


def test_har_target():
    t = har_target(pd.Series([1.0, 2.0, 3.0, 4.0]), 2)
    assert t.iloc[0] == 2.5 and t.iloc[1] == 3.5 and t.iloc[2:].isna().all()


def test_fit_har_recovers_known_coefficients():
    # build an rv series that exactly follows a HAR(1-day) recursion
    rng = np.random.default_rng(1)
    rv = pd.Series(np.abs(rng.normal(1.0, 0.3, 400)))
    true = np.array([0.1, 0.3, 0.2, 0.4])
    X = har_features(rv)
    for t in range(22, 399):
        rv.iloc[t + 1] = true @ np.r_[1.0, X.iloc[t].to_numpy()]
        X.iloc[t + 1] = [rv.iloc[t + 1], rv.iloc[t - 3:t + 2].mean(), rv.iloc[t - 20:t + 2].mean()]
    coef = fit_har(rv.iloc[60:].reset_index(drop=True), horizon=1)
    assert coef == pytest.approx(true, abs=0.02)


def test_har_tracks_volatility_regimes():
    r, sig = _regime_returns()
    data, days = _curve_from_returns(r)
    har = HARVol(horizon=5)
    # late in a HIGH block vs late in a LOW block
    hi_day, lo_day = days[12 * 120 - 10], days[11 * 120 - 10]   # blocks count from 0: block 11 high, block 10 low
    s_hi = har.estimate(data, settle_time(hi_day) + pd.Timedelta(minutes=35))
    s_lo = har.estimate(data, settle_time(lo_day) + pd.Timedelta(minutes=35))
    assert s_hi > 0.45 and s_lo < 0.32
    assert har.coef_ is not None and len(har.coef_) == 4


def test_har_needs_history():
    r, _ = _regime_returns(n_blocks=1, block=100)
    data, days = _curve_from_returns(r)
    with pytest.raises(ValueError):
        HARVol(min_obs=250).estimate(data, TS("2030-01-01"))


def test_gbm_accepts_har():
    r, _ = _regime_returns()
    data, days = _curve_from_returns(r)
    as_of = settle_time(days[1000]) + pd.Timedelta(minutes=35)
    m = GBMModel(vol=HARVol())
    m.fit(data, as_of)
    c = Contract(ticker="X", contract_type="above", strike=70.0, settle_date=days[1005], futures_month="2099-12")
    assert 0 < m.predict(c, as_of).probability < 1


# ---- implied vol ---------------------------------------------------------------

def test_implied_vol_point_in_time():
    r, _ = _regime_returns(n_blocks=1, block=50)
    data, days = _curve_from_returns(r)
    data.features = pd.DataFrame({"name": "OVXCLS", "date": [days[10], days[11]], "value": [40.0, 80.0],
                                  "available_at": [TS(days[11]) + pd.Timedelta(hours=8),
                                                   TS(days[12]) + pd.Timedelta(hours=8)]})
    iv = ImpliedVol()
    assert iv.estimate(data, settle_time(days[11])) == pytest.approx(0.40)   # day-11 close not public yet
    assert iv.estimate(data, settle_time(days[12])) == pytest.approx(0.80)
    assert ImpliedVol(scale=0.9).estimate(data, settle_time(days[12])) == pytest.approx(0.72)
    with pytest.raises(ValueError):
        iv.estimate(data, settle_time(days[5]))


# ---- event-aware vol -----------------------------------------------------------

def _event_world(seed=3, n=1200, mult=3.0):
    rng = np.random.default_rng(seed)
    sd = 0.30 / np.sqrt(252)
    r = rng.standard_normal(n) * sd
    data, days = _curve_from_returns(np.zeros(n))
    event_idx = np.arange(5, n, 5)                       # every 5th trading day is an "EIA" day
    r[event_idx - 1] *= mult                             # return INTO the event day's settle
    data, days = _curve_from_returns(r)
    events = pd.DataFrame({"date": days[event_idx], "event_type": "EIA",
                           "announced_at": days[event_idx] - pd.Timedelta(days=60)})
    planted = (mult ** 2 - 1) * sd ** 2
    return data, days, events, planted


def test_event_extra_variance_recovers_planted_value():
    data, days, events, planted = _event_world()
    returns = data.curve.nearby_returns(TS("2040-01-01"))
    est = event_extra_variance(returns, pd.DatetimeIndex(events["date"]))
    assert est == pytest.approx(planted, rel=0.25)
    assert event_extra_variance(returns, pd.DatetimeIndex(events["date"][:3])) == 0.0   # too few events


def test_event_aware_adds_variance_only_for_events_in_window():
    data, days, events, planted = _event_world()
    ev = EventAwareVol(HistoricalVol(window=60), events)
    i = 1000
    as_of = settle_time(days[i]) + pd.Timedelta(minutes=35)
    sigma = ev.estimate(data, as_of)
    assert ev.extra_var_["EIA"] == pytest.approx(planted, rel=0.3)
    nxt = events["date"][events["date"] > days[i]].iloc[0]
    T = 1 / 252
    before = ev.horizon_variance(sigma, T, as_of, nxt - pd.Timedelta(days=1))
    through = ev.horizon_variance(sigma, T, as_of, nxt)
    assert before == pytest.approx(sigma ** 2 * T)
    assert through == pytest.approx(sigma ** 2 * T + ev.extra_var_["EIA"])


def test_event_not_yet_announced_is_ignored():
    data, days, events, planted = _event_world()
    i = 1000
    as_of = settle_time(days[i]) + pd.Timedelta(minutes=35)
    late = events.copy()
    late["announced_at"] = late["date"] + pd.Timedelta(days=1)       # every event announced after it happens
    ev = EventAwareVol(HistoricalVol(window=60), late)
    sigma = ev.estimate(data, as_of)
    nxt = late["date"][late["date"] > days[i]].iloc[0]
    assert ev.horizon_variance(sigma, 1 / 252, as_of, nxt) == pytest.approx(sigma ** 2 / 252)


def test_gbm_with_events_widens_distribution():
    data, days, events, planted = _event_world()
    i = 1000
    as_of = settle_time(days[i]) + pd.Timedelta(minutes=35)
    nxt = events["date"][events["date"] > days[i]].iloc[0]
    F = data.curve.settle("2099-12", as_of)
    c = Contract(ticker="X", contract_type="above", strike=F * 1.03, settle_date=nxt, futures_month="2099-12")
    plain, evt = GBMModel(vol=HistoricalVol(window=60)), GBMModel(vol=EventAwareVol(HistoricalVol(window=60), events))
    plain.fit(data, as_of)
    evt.fit(data, as_of)
    assert evt.predict(c, as_of).probability > plain.predict(c, as_of).probability
