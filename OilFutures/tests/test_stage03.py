# PROVIDED · Stage 3 checks: run with  pytest tests/test_stage03.py -v
import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import Backtester, walk_forward_windows
from evaluation.scoring import (add_buckets, bootstrap_brier_skill, brier_score, brier_skill,
                                calibration_table, log_loss, strike_distance_sd)
from evaluation.simulate import add_logit_noise, synthetic_backtest_inputs
from models.gbm import GBMModel, HistoricalVol


# ---- metrics ----------------------------------------------------------------

def test_brier_examples():
    assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0


def test_log_loss_example_and_clipping():
    assert log_loss(np.array([0.9, 0.2]), np.array([1, 0])) == pytest.approx(0.1643, abs=1e-4)
    assert np.isfinite(log_loss(np.array([0.0]), np.array([1])))


def test_brier_skill_example():
    assert brier_skill(np.array([0.9, 0.1]), np.array([0.6, 0.4]), np.array([1, 0])) == pytest.approx(0.9375)
    p = np.array([0.3, 0.7])
    assert brier_skill(p, p, np.array([0, 1])) == pytest.approx(0.0)


def test_calibration_table():
    t = calibration_table(np.array([0.05, 0.08, 0.92, 1.0]), np.array([0, 0, 1, 1]), n_bins=10)
    assert list(t.columns) == ["bin", "p_mean", "y_mean", "count"]
    assert list(t["bin"]) == [0, 9]
    assert list(t["count"]) == [2, 2]
    assert t["p_mean"].iloc[0] == pytest.approx(0.065)


def test_strike_distance():
    d = strike_distance_sd(np.array([77.0, 70.0]), np.array([70.0, 70.0]), np.array([0.35, 0.35]),
                           np.array([0.25, 0.0]))
    assert d[0] == pytest.approx(np.log(1.1) / (0.35 * 0.5))
    assert np.isinf(d[1])


def test_buckets():
    df = pd.DataFrame({"horizon_days": [1, 7, 8, 30, 200], "strike_distance_sd": [0.0, 0.3, 0.7, 1.5, 3.0]})
    out = add_buckets(df, [0, 7, 28, 91, 100000], ["1-7d", "1-4w", "1-3m", "3m+"], [0, 0.5, 1, 2, 100])
    assert list(out["horizon_bucket"].astype(str)) == ["1-7d", "1-7d", "1-4w", "1-3m", "3m+"]
    assert out["distance_bucket"].notna().all()
    assert "horizon_bucket" not in df.columns, "add_buckets must not modify its input"


def _bootstrap_frame(seed=0, n_days=200, per_day=10):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        y = int(rng.random() < 0.5)               # one outcome per day: strikes move together
        for _ in range(per_day):
            rows.append({"date": d, "outcome": y, "probability": 0.5 + 0.1 * (2 * y - 1) * rng.random(),
                         "market_price": 0.5})
    return pd.DataFrame(rows)


def test_bootstrap_contains_point_and_is_reproducible():
    df = _bootstrap_frame()
    a = bootstrap_brier_skill(df, n_boot=500, seed=1)
    b = bootstrap_brier_skill(df, n_boot=500, seed=1)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]


def test_block_bootstrap_is_wider_than_row_bootstrap():
    rng = np.random.default_rng(3)
    rows = []
    for d in range(150):
        shock = rng.normal()                      # one shared surprise per day
        for _ in range(10):
            y = int(shock > 0)
            rows.append({"date": d, "outcome": y, "probability": float(np.clip(0.5 + 0.3 * shock, 0.01, 0.99)),
                         "market_price": 0.5})
    df = pd.DataFrame(rows)
    blk = bootstrap_brier_skill(df, n_boot=800, block_col="date", seed=0)
    iid = bootstrap_brier_skill(df, n_boot=800, block_col=None, seed=0)
    assert (blk["hi"] - blk["lo"]) > 1.5 * (iid["hi"] - iid["lo"]), \
        "resampling whole days should give a much wider (honest) interval"


def test_logit_noise():
    rng = np.random.default_rng(0)
    assert np.allclose(add_logit_noise(np.array([0.2, 0.5]), 0.0, rng), [0.2, 0.5])
    noisy = add_logit_noise(np.full(10_000, 0.3), 1.0, rng)
    assert ((noisy > 0) & (noisy < 1)).all()
    assert np.median(noisy) == pytest.approx(0.3, abs=0.02)


# ---- walk-forward -----------------------------------------------------------

def test_walk_forward_windows():
    assert walk_forward_windows(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]


@pytest.fixture(scope="module")
def synthetic_result():
    data, contracts = synthetic_backtest_inputs(n_days=2000, sigma=0.35, noise_sd=0.6, seed=4)
    bt = Backtester(GBMModel(vol=HistoricalVol(window=60)), data, refit_every=20)
    return bt, contracts, bt.run(contracts)


def test_backtest_output_shape(synthetic_result):
    bt, contracts, res = synthetic_result
    assert len(res) == len(contracts)
    for col in ("probability", "market_price", "outcome", "strike_distance_sd", "horizon_days"):
        assert col in res.columns
    assert len(bt.fit_log) == int(np.ceil(contracts["date"].nunique() / 20))


def test_backtest_never_fits_on_the_future(synthetic_result):
    bt, contracts, res = synthetic_result
    fits = pd.Series(sorted(bt.fit_log))
    # every prediction's as_of must be at or after the most recent fit before it
    idx = fits.searchsorted(res["as_of"], side="right") - 1
    assert (idx >= 0).all(), "a prediction was made before any fit"


def test_gbm_is_calibrated_on_gbm_world(synthetic_result):
    # NOTE: even the TRUE probabilities miss by ~0.02 in some bins here, because all
    # strikes on a day share one price path (the Stage 3 "correlated luck" lesson).
    _, _, res = synthetic_result
    assert res["probability"].mean() == pytest.approx(res["true_prob"].mean(), abs=0.01)
    t = calibration_table(res["probability"], res["outcome"], n_bins=10)
    big = t[t["count"] >= 300]
    assert (big["p_mean"] - big["y_mean"]).abs().max() < 0.06, f"miscalibrated bins:\n{big}"


def test_gbm_beats_noisy_market(synthetic_result):
    _, _, res = synthetic_result
    s = bootstrap_brier_skill(res, n_boot=500, seed=0)
    assert s["point"] > 0.03
    assert s["lo"] > 0, "on its own synthetic world GBM must beat a noisy market; if not, the harness has a bug"


def test_gbm_close_to_truth(synthetic_result):
    _, _, res = synthetic_result
    # vs the TRUE probabilities the model can't be much better or worse (it only mis-estimates sigma)
    assert abs(brier_skill(res["probability"], res["true_prob"], res["outcome"])) < 0.02
