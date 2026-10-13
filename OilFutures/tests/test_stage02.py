# PROVIDED · Stage 2 checks: run with  pytest tests/test_stage02.py -v
import numpy as np
import pandas as pd
import pytest

from models.base import Contract
from models.gbm import (GBMModel, HistoricalVol, prob_above, prob_from_barrier, prob_from_terminal,
                        prob_touch_continuous, prob_touch_discrete_bgk, simulate_gbm_paths)
from tests.helpers import gbm_prices, make_curve


def rng(seed=0):
    return np.random.default_rng(seed)


def test_paths_shape_and_start():
    p = simulate_gbm_paths(70.0, 0.3, 20 / 252, 20, 500, rng())
    assert p.shape == (500, 21)
    assert np.allclose(p[:, 0], 70.0)
    assert (p > 0).all(), "GBM prices can never be negative"


def test_paths_are_a_martingale():
    # zero drift: the AVERAGE final price equals today's price
    p = simulate_gbm_paths(70.0, 0.4, 1.0, 50, 200_000, rng(1))
    se = p[:, -1].std() / np.sqrt(len(p))
    assert abs(p[:, -1].mean() - 70.0) < 4 * se


def test_paths_have_the_right_volatility():
    p = simulate_gbm_paths(70.0, 0.4, 1.0, 50, 100_000, rng(2))
    assert np.log(p[:, -1] / 70.0).std() == pytest.approx(0.4, rel=0.02)


def test_prob_from_terminal_examples():
    f = np.array([60.0, 70.0, 80.0, 90.0])
    assert prob_from_terminal(f, "above", 75.0) == 0.5
    assert prob_from_terminal(f, "below", 75.0) == 0.5
    assert prob_from_terminal(f, "between", 65.0, 85.0) == 0.5


@pytest.mark.parametrize("K", [60.0, 67.0, 70.0, 74.0, 85.0])
def test_monte_carlo_matches_closed_form(K):
    n, T, sigma = 100_000, 20 / 252, 0.35
    final = simulate_gbm_paths(70.0, sigma, T, 20, n, rng(3))[:, -1]
    p_mc = prob_from_terminal(final, "above", K)
    p_cf = prob_above(70.0, K, sigma, T)
    se = np.sqrt(max(p_cf * (1 - p_cf), 1e-12) / n)
    assert abs(p_mc - p_cf) < 4 * se + 1e-4, "MC should agree with Stage 1 within sampling error"


def test_prob_from_barrier_example():
    paths = np.array([[70.0, 71.0, 76.0], [70.0, 69.0, 68.0]])
    assert prob_from_barrier(paths, "max", 75.0) == 0.5
    assert prob_from_barrier(paths, "min", 68.5) == 0.5
    # column 0 is today's known price, not a future settle
    assert prob_from_barrier(np.array([[80.0, 70.0, 70.0]]), "max", 75.0) == 0.0


@pytest.mark.parametrize("K,direction", [(75.0, "max"), (66.0, "min")])
def test_fine_monitoring_matches_reflection_formula(K, direction):
    T, sigma = 20 / 252, 0.35
    paths = simulate_gbm_paths(70.0, sigma, T, 2000, 20_000, rng(4))
    p_mc = prob_from_barrier(paths, direction, K)
    p_cont = prob_touch_continuous(70.0, K, sigma, T, direction)
    assert p_mc == pytest.approx(p_cont, abs=0.015)


def test_daily_monitoring_is_lower_than_continuous():
    # The key Stage 2 lesson: Kalshi checks 20 daily settles, not every instant.
    T, sigma = 20 / 252, 0.35
    daily = prob_from_barrier(simulate_gbm_paths(70.0, sigma, T, 20, 100_000, rng(5)), "max", 75.0)
    cont = prob_touch_continuous(70.0, 75.0, sigma, T, "max")
    assert cont - daily > 0.03, "continuous formula should clearly overstate the daily-settle probability"


def test_continuous_formula_example_and_edges():
    assert prob_touch_continuous(70.0, 75.0, 0.35, 20 / 252, "max") == pytest.approx(0.4675, abs=1e-3)
    assert prob_touch_continuous(70.0, 65.0, 0.35, 20 / 252, "max") == 1.0


def test_optional_bgk_correction_matches_daily_monitoring():
    try:
        p_bgk = prob_touch_discrete_bgk(70.0, 75.0, 0.35, 20 / 252, 20, "max")
    except NotImplementedError:
        pytest.skip("optional: prob_touch_discrete_bgk not implemented")
    daily = prob_from_barrier(simulate_gbm_paths(70.0, 0.35, 20 / 252, 20, 200_000, rng(6)), "max", 75.0)
    assert p_bgk == pytest.approx(daily, abs=0.01)


# ---- GBMModel.prob_barrier --------------------------------------------------

def _model():
    data = make_curve(gbm_prices(60, sigma=0.35, seed=7))
    last = data.curve.df.iloc[-1]
    as_of = last["available_at"] + pd.Timedelta(minutes=35)
    m = GBMModel(vol=HistoricalVol(window=20), n_paths=50_000, seed=0)
    m.fit(data, as_of)
    return m, data, as_of, float(last["settle"])


def test_model_barrier_close_to_daily_monte_carlo():
    m, data, as_of, F = _model()
    sd = pd.Timestamp("2026-04-30")   # 20+ trading days after the last settle in the helper data
    c = Contract(ticker="MAX1", contract_type="max", strike=F * 1.05, settle_date=sd, futures_month="2026-12")
    p = m.predict(c, as_of).probability
    from common.calendar import trading_days_until
    n = trading_days_until(as_of, sd)
    ref = prob_from_barrier(simulate_gbm_paths(F, m.sigma, n / 252, n, 100_000, rng(8)), "max", F * 1.05)
    assert p == pytest.approx(ref, abs=0.01)
    assert p < prob_touch_continuous(F, F * 1.05, m.sigma, n / 252, "max")


def test_model_barrier_already_hit_is_certain():
    m, data, as_of, F = _model()
    hist = data.curve.history("2026-12", as_of)
    c = Contract(ticker="MAX2", contract_type="max", strike=float(hist.max()) - 0.01,
                 settle_date=pd.Timestamp("2026-04-30"), futures_month="2026-12", start_date=hist.index[0])
    assert m.predict(c, as_of).probability == 1.0


def test_model_min_contract_in_range():
    m, data, as_of, F = _model()
    c = Contract(ticker="MIN1", contract_type="min", strike=F * 0.95,
                 settle_date=pd.Timestamp("2026-04-30"), futures_month="2026-12")
    p = m.predict(c, as_of).probability
    assert 0.05 < p < 0.95
