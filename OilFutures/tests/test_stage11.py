# PROVIDED · Stage 11 checks: run with  pytest tests/test_stage11.py -v
import numpy as np
import pandas as pd
import pytest

from evaluation.scoring import brier_score
from layers.market_layer import (FEATURES, MarketAdjustedModel, MarketLayer, build_features,
                                 cross_strike_inconsistency, gap_by_tercile, gap_slices, logit, taker_imbalance,
                                 volume_spike)
from models.base import Contract
from models.gbm import GBMModel, HistoricalVol
from tests.helpers import gbm_prices, make_curve

TS = pd.Timestamp
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def test_logit_is_finite_at_edges():
    v = logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(v).all() and v[1] == 0.0 and v[0] < -5 and v[2] > 5


def test_taker_imbalance():
    trades = pd.DataFrame({"time": [TS("2026-10-12 09:00"), TS("2026-10-12 12:00"), TS("2026-10-12 14:00"),
                                    TS("2026-10-12 16:00")],
                           "count": [10.0, 30.0, 10.0, 100.0], "taker_side": ["no", "yes", "no", "yes"]})
    assert taker_imbalance(trades, TS("2026-10-12 15:05"), 24) == pytest.approx((30 - 20) / 50)
    assert taker_imbalance(trades, TS("2026-10-12 15:05"), 2) == pytest.approx(-1.0)
    assert taker_imbalance(trades.iloc[:0], TS("2026-10-12 15:05")) == 0.0


def test_cross_strike_inconsistency():
    assert cross_strike_inconsistency(np.array([70, 71, 72]), np.array([0.6, 0.4, 0.2])) == pytest.approx([0, 0, 0])
    out = cross_strike_inconsistency(np.array([72, 70, 71]), np.array([0.45, 0.60, 0.40]))
    assert out == pytest.approx([0.025, 0.0, -0.025])


def test_volume_spike():
    assert volume_spike(99.0, np.array([9.0, 9.0, 9.0])) == pytest.approx(np.log(10))
    assert volume_spike(5.0, np.array([])) == 0.0


def _raw(n, seed=0, informed=0.0):
    """Synthetic engine/market rows. Cheap contracts (market < 15c) are overpriced; engine is noisy truth."""
    g = np.random.default_rng(seed)
    market = sigmoid(g.normal(0, 1.8, n))
    p = sigmoid(logit(market) - 0.7 * (market < 0.15) + g.normal(0, 0.4, n))   # truth sits BELOW cheap prices
    y = (g.random(n) < p).astype(int)
    engine = sigmoid(logit(p) + g.normal(0, 0.5, n))
    imb = np.clip(informed * (y - p) + g.normal(0, 0.5, n), -1, 1)
    return pd.DataFrame({"engine_p": engine, "market_p": market, "prev_engine_p": np.nan, "prev_market_p": np.nan,
                         "taker_imbalance": imb, "hours_to_close": g.uniform(1, 100, n),
                         "cross_strike_gap": 0.0, "volume_spike": g.normal(0, 1, n), "outcome": y, "true_p": p})


def test_build_features():
    df = _raw(5)
    df.loc[0, ["prev_engine_p", "prev_market_p"]] = [df.loc[0, "engine_p"], 0.5]
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert X.loc[0, "overreaction"] == pytest.approx(logit(df.loc[0, "market_p"]) - 0.0)
    assert (X.loc[1:, "overreaction"] == 0).all()
    assert set(X["longshot"]) <= {0, 1}


def test_layer_coefficients_have_expected_signs():
    layer = MarketLayer().fit(_raw(20_000, seed=1, informed=0.3))
    c = layer.coef_
    assert c["logit_engine"] > 0 and c["logit_market"] > 0
    assert c["longshot"] < 0, "the market overprices longshots, so the layer should mark them down"
    assert c["taker_imbalance"] > 0, "informed taker flow should push the probability its way"


def test_layer_improves_on_both_inputs_out_of_sample():
    layer = MarketLayer().fit(_raw(20_000, seed=2))
    test = _raw(20_000, seed=3)
    adj = layer.adjust(test)
    assert brier_score(adj, test["outcome"]) < brier_score(test["market_p"], test["outcome"])
    assert brier_score(adj, test["outcome"]) < brier_score(test["engine_p"], test["outcome"])


def test_adjustment_never_leaves_open_interval():
    layer = MarketLayer().fit(_raw(5000, seed=4))
    extreme = _raw(4, seed=5)
    extreme["engine_p"] = [0.0, 1.0, 0.0, 1.0]
    extreme["market_p"] = [0.0, 1.0, 1.0, 0.0]
    extreme["taker_imbalance"] = [-1, 1, 1, -1]
    p = layer.adjust(extreme)
    assert ((p > 0) & (p < 1)).all()


def test_market_adjusted_model_wraps_engine():
    data = make_curve(gbm_prices(60, sigma=0.3, seed=1))
    as_of = data.curve.df["available_at"].iloc[-1] + pd.Timedelta(minutes=35)
    layer = MarketLayer().fit(_raw(5000, seed=6))
    lookup = lambda c, t: {"market_p": 0.4, "prev_engine_p": np.nan, "prev_market_p": np.nan, "taker_imbalance": 0.0,
                           "hours_to_close": 24.0, "cross_strike_gap": 0.0, "volume_spike": 0.0}
    m = MarketAdjustedModel(GBMModel(vol=HistoricalVol(20)), layer, lookup)
    m.fit(data, as_of)
    c = Contract(ticker="X", contract_type="above", strike=70.0, settle_date="2026-05-01", futures_month="2026-12")
    pred = m.predict(c, as_of)
    assert pred.model_name == "gbm+market" and 0 < pred.probability < 1
    assert "engine_p" in pred.inputs and "F_available_at" in pred.inputs


def test_gap_tables():
    df = _raw(3000, seed=7).assign(sigma=np.random.default_rng(0).uniform(0.2, 0.6, 3000),
                                   prev_return=np.random.default_rng(1).normal(0, 0.02, 3000))
    t = gap_by_tercile(df)
    assert list(t.index.astype(str)) == ["low", "mid", "high"]
    assert {"n", "engine_mean", "market_mean", "outcome_mean", "gap_mean"} <= set(t.columns)
    assert t.loc["low", "gap_mean"] > 0, "market overprices the low (longshot) tercile in this world"
    s = gap_slices(df)
    assert list(s.index) == ["tail", "high_vol", "post_shock", "all"]
    assert s.loc["all", "n"] == 3000
