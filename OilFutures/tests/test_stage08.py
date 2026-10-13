# PROVIDED · Stage 8 checks: run with  pytest tests/test_stage08.py -v
import numpy as np
import pandas as pd
import pytest

from common.calendar import settle_time, trading_days
from models.base import Contract, FuturesCurve, MarketData
from models.gbm import GBMModel, HistoricalVol, prob_above
from tests.helpers import data_from_returns as _data_from_returns
from models.heston import (HestonModel, HestonParams, estimate_heston_params, rho_history,
                           simulate_heston_paths, variance_proxy)

rng = np.random.default_rng


def _heston_returns(p, n, seed):
    prices, _ = simulate_heston_paths(70.0, p, n / 252, n, 1, rng(seed), substeps=8)
    return pd.Series(np.diff(np.log(prices[0])))


def test_feller_property():
    assert HestonParams(2.0, 0.09, 0.5, -0.5, 0.09).feller_ok
    assert not HestonParams(1.0, 0.04, 1.0, -0.5, 0.04).feller_ok


def test_variance_proxy():
    r = pd.Series([0.01] * 30)
    assert variance_proxy(r, 21).iloc[-1] == pytest.approx(0.01 ** 2 * 252)


def test_variance_never_negative_even_when_feller_fails():
    p = HestonParams(kappa=1.0, theta=0.04, xi=1.5, rho=-0.7, v0=0.04)
    prices, v = simulate_heston_paths(70.0, p, 0.5, 126, 5000, rng(0))
    assert prices.shape == v.shape == (5000, 127)
    assert (v >= 0).all()
    assert np.isfinite(prices).all() and (prices > 0).all()


def test_heston_paths_are_a_martingale():
    p = HestonParams(kappa=2.0, theta=0.10, xi=0.6, rho=-0.6, v0=0.10)
    prices, _ = simulate_heston_paths(70.0, p, 0.25, 63, 100_000, rng(1))
    se = prices[:, -1].std() / np.sqrt(len(prices))
    assert abs(prices[:, -1].mean() - 70.0) < 4 * se


def test_zero_vol_of_vol_reproduces_gbm():
    sigma = 0.35
    p = HestonParams(kappa=2.0, theta=sigma ** 2, xi=1e-8, rho=-0.5, v0=sigma ** 2)
    prices, _ = simulate_heston_paths(70.0, p, 20 / 252, 20, 100_000, rng(2))
    for K in (64.0, 70.0, 76.0):
        p_mc = (prices[:, -1] > K).mean()
        p_cf = prob_above(70.0, K, sigma, 20 / 252)
        assert abs(p_mc - p_cf) < 4 * np.sqrt(p_cf * (1 - p_cf) / 100_000) + 1e-4


def test_negative_rho_creates_skew():
    base = dict(kappa=2.0, theta=0.12, xi=0.9, v0=0.12)
    neg, _ = simulate_heston_paths(70.0, HestonParams(rho=-0.8, **base), 0.5, 126, 50_000, rng(3))
    pos, _ = simulate_heston_paths(70.0, HestonParams(rho=0.8, **base), 0.5, 126, 50_000, rng(3))
    # negative rho: crashes come with high vol -> fatter LEFT tail
    assert (neg[:, -1] < 50).mean() > (pos[:, -1] < 50).mean()


def test_estimation_recovers_rough_values_and_rho_sign():
    true = HestonParams(kappa=3.0, theta=0.12, xi=0.8, rho=-0.7, v0=0.12)
    r = _heston_returns(true, 5000, seed=4)
    est = estimate_heston_params(r)
    assert est.kappa > 0.3
    assert est.theta == pytest.approx(variance_proxy(r).mean(), rel=0.35)
    assert est.xi > 0
    assert est.rho < -0.03, "the proxy method understates |rho| but must get its sign"
    est_pos = estimate_heston_params(_heston_returns(HestonParams(3.0, 0.12, 0.8, 0.7, 0.12), 5000, seed=5))
    assert est_pos.rho > 0.03


def test_rho_history_detects_sign_flip():
    a = _heston_returns(HestonParams(3.0, 0.12, 0.9, -0.8, 0.12), 2000, seed=6)
    b = _heston_returns(HestonParams(3.0, 0.12, 0.9, 0.8, 0.12), 2000, seed=7)
    r = pd.concat([a, b], ignore_index=True)
    h = rho_history(r, lookback=750, step=125)
    assert h.iloc[0] < 0 < h.iloc[-1]


def test_heston_model_end_to_end_and_rho_log():
    r = _heston_returns(HestonParams(3.0, 0.12, 0.8, -0.6, 0.12), 600, seed=8)
    data, days = _data_from_returns(r)
    m = HestonModel(n_paths=20_000, seed=0)
    for i in (500, 520):
        as_of = settle_time(days[i]) + pd.Timedelta(minutes=35)
        m.fit(data, as_of)
    assert len(m.rho_log) == 2
    F = data.curve.settle("2099-12", as_of)
    sd = days[540]
    up = Contract(ticker="U", contract_type="above", strike=F, settle_date=sd, futures_month="2099-12")
    dn = Contract(ticker="D", contract_type="below", strike=F, settle_date=sd, futures_month="2099-12")
    mx = Contract(ticker="M", contract_type="max", strike=F * 1.05, settle_date=sd, futures_month="2099-12")
    p_up, p_dn = m.predict(up, as_of).probability, m.predict(dn, as_of).probability
    assert p_up + p_dn == pytest.approx(1.0, abs=0.02)
    p_max = m.predict(mx, as_of).probability
    tail = m.predict(Contract(ticker="A", contract_type="above", strike=F * 1.05, settle_date=sd,
                              futures_month="2099-12"), as_of).probability
    assert p_max >= tail - 0.01, "touching a level is at least as likely as finishing above it"


def test_variance_forecaster_sets_v0():
    r = _heston_returns(HestonParams(3.0, 0.12, 0.8, -0.6, 0.12), 600, seed=9)
    data, days = _data_from_returns(r)
    as_of = settle_time(days[500]) + pd.Timedelta(minutes=35)
    m = HestonModel(variance_forecaster=HistoricalVol(window=20))
    m.fit(data, as_of)
    assert m.params.v0 == pytest.approx(HistoricalVol(window=20).estimate(data, as_of) ** 2)
