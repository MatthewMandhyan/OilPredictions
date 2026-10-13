# PROVIDED · Stage 9 checks: run with  pytest tests/test_stage09.py -v
import numpy as np
import pandas as pd
import pytest

from common.calendar import settle_time
from models.base import Contract
from models.bates import (BatesModel, BatesParams, detect_jumps, estimate_jump_params, remove_jumps,
                          simulate_bates_paths)
from models.heston import HestonModel, HestonParams, simulate_heston_paths
from tests.helpers import data_from_returns as _data_from_returns

rng = np.random.default_rng
H = dict(kappa=2.0, theta=0.10, xi=0.5, rho=-0.5, v0=0.10)


def _jumpy_returns(n=2000, n_jumps=10, size=0.15, seed=0):
    g = rng(seed)
    r = pd.Series(g.normal(0, 0.02, n))
    idx = g.choice(np.arange(100, n), n_jumps, replace=False)
    r.iloc[idx] = size * g.choice([-1.0, 1.0], n_jumps)
    return r, set(idx)


def test_bates_params_inherit_heston():
    p = BatesParams(**H, lam=2.0, mu_j=-0.05, sigma_j=0.1)
    assert isinstance(p, HestonParams) and p.feller_ok in (True, False)
    assert p.compensator == pytest.approx(np.exp(-0.05 + 0.005) - 1)


def test_detect_planted_jumps():
    r, planted = _jumpy_returns()
    flags = detect_jumps(r, k=4.0, window=60)
    found = set(np.flatnonzero(flags.to_numpy()))
    assert len(found & planted) >= 9
    assert len(found - planted) <= 3
    assert not flags.iloc[:60].any()


def test_detect_jumps_uses_only_the_past():
    r, _ = _jumpy_returns()
    a = detect_jumps(r)
    r2 = r.copy()
    r2.iloc[1500:] = 5.0
    b = detect_jumps(r2)
    pd.testing.assert_series_equal(a.iloc[:1500], b.iloc[:1500])


def test_estimate_and_remove_jumps():
    r, planted = _jumpy_returns(n=2000, n_jumps=10)
    mask = pd.Series(False, index=r.index)
    mask.iloc[sorted(planted)] = True
    lam, mu, sj = estimate_jump_params(r, mask)
    assert lam == pytest.approx(10 / 2000 * 252)
    assert sj > 0.1
    clean = remove_jumps(r, mask)
    assert len(clean) == 1990 and clean.abs().max() < 0.15
    assert estimate_jump_params(r, mask & False) == (0.0, 0.0, 0.0)


def test_zero_jump_intensity_reproduces_heston():
    T, n = 20 / 252, 20
    b, _ = simulate_bates_paths(70.0, BatesParams(**H), T, n, 100_000, rng(1))
    h, _ = simulate_heston_paths(70.0, HestonParams(**H), T, n, 100_000, rng(2))
    for K in (64.0, 70.0, 76.0):
        pb, ph = (b[:, -1] > K).mean(), (h[:, -1] > K).mean()
        assert abs(pb - ph) < 5 * np.sqrt(2 * ph * (1 - ph) / 100_000) + 1e-4


def test_jumps_keep_the_martingale():
    p = BatesParams(**H, lam=5.0, mu_j=-0.08, sigma_j=0.10)
    prices, v = simulate_bates_paths(70.0, p, 0.5, 126, 100_000, rng(3))
    se = prices[:, -1].std() / np.sqrt(len(prices))
    assert abs(prices[:, -1].mean() - 70.0) < 4 * se, "missing compensator? E[F_T] must equal F0"
    assert (v >= 0).all()


def test_jumps_fatten_tails():
    T, n = 20 / 252, 20
    b, _ = simulate_bates_paths(70.0, BatesParams(**H, lam=8.0, mu_j=-0.05, sigma_j=0.08), T, n, 100_000, rng(4))
    h, _ = simulate_heston_paths(70.0, HestonParams(**H), T, n, 100_000, rng(5))
    assert (b[:, -1] < 56).mean() > 1.5 * (h[:, -1] < 56).mean()


def test_bates_model_end_to_end():
    r, planted = _jumpy_returns(n=800, n_jumps=8, size=0.12, seed=6)
    data, days = _data_from_returns(r.to_numpy())
    as_of = settle_time(days[790]) + pd.Timedelta(minutes=35)
    bates, heston = BatesModel(n_paths=20_000, seed=0), HestonModel(n_paths=20_000, seed=0)
    bates.fit(data, as_of)
    heston.fit(data, as_of)
    assert isinstance(bates.params, BatesParams) and bates.params.lam > 0
    assert bates.params.theta < heston.params.theta, "removing jumps should lower the diffusive variance"
    F = data.curve.settle("2099-12", as_of)
    c = Contract(ticker="X", contract_type="below", strike=F * 0.85, settle_date=days[799], futures_month="2099-12")
    p = bates.predict(c, as_of)
    assert 0.0 <= p.probability <= 1.0 and p.model_name == "bates"
