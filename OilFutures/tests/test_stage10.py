# PROVIDED · Stage 10 checks: run with  pytest tests/test_stage10.py -v
import numpy as np
import pandas as pd
import pytest

from common.calendar import settle_time
from layers.factor_layer import FactorHARVol, directional_test, event_counts, factor_matrix
from models.bates import BatesModel
from models.har import HARVol, har_target
from tests.helpers import data_from_returns

TS = pd.Timestamp


def _factor_world(n=1600, seed=0):
    """Daily vol is driven by a public factor 'signal' published at NOON, before that day's snapshot.

    Plus three pure-noise factors. A correct LASSO-HAR should pick 'signal' and beat plain HAR.
    """
    g = np.random.default_rng(seed)
    persistent = np.zeros(n)
    for t in range(1, n):
        persistent[t] = 0.97 * persistent[t - 1] + g.normal(0, 0.05)
    f = g.uniform(0, 1, n)                              # fresh news each day, linear effect on variance
    sig_daily = 0.30 / np.sqrt(252) * np.sqrt((0.4 + 1.6 * f) * np.exp(persistent))
    r = g.standard_normal(n) * sig_daily               # r[t] is the return INTO day t+1's settle
    data, days = data_from_returns(r)
    rows = []
    for t in range(n):                                  # f[t] drives r[t] (day t -> t+1), published day t 12:00
        rows.append({"name": "signal", "date": days[t], "value": f[t], "available_at": days[t] + pd.Timedelta(hours=12)})
        for k in range(3):
            rows.append({"name": f"noise{k}", "date": days[t], "value": g.normal(),
                         "available_at": days[t] + pd.Timedelta(hours=12)})
    data.features = pd.DataFrame(rows)
    data.true_daily_var = sig_daily ** 2               # test-only attribute: the variance we planted
    return data, days


def test_factor_matrix_is_point_in_time():
    data, days = _factor_world(n=300)
    m = factor_matrix(data, pd.DatetimeIndex(days[100:103]), ["signal"])
    vals = data.features[data.features["name"] == "signal"].set_index("date")["value"]
    # at day d's 15:05 snapshot, the value published at noon that day is the latest known
    assert m.loc[days[101], "signal"] == pytest.approx(vals[days[101]])
    late = data.features.assign(available_at=data.features["available_at"] + pd.Timedelta(hours=6))  # 18:00
    data2 = type(data)(data.curve, late)
    assert factor_matrix(data2, pd.DatetimeIndex([days[101]]), ["signal"]).iloc[0, 0] == pytest.approx(vals[days[100]])
    assert list(m.index) == list(days[100:103])


def test_event_counts():
    dates = pd.DatetimeIndex([TS("2026-10-05"), TS("2026-10-12")])
    events = pd.DataFrame({"date": [TS("2026-10-07"), TS("2026-10-14"), TS("2026-10-08")],
                           "event_type": ["EIA", "EIA", "OPEC"],
                           "announced_at": [TS("2026-09-01"), TS("2026-09-01"), TS("2026-10-06")]})
    out = event_counts(dates, events, horizon=5)
    assert list(out.columns) == ["evt_EIA", "evt_OPEC"]
    assert list(out["evt_EIA"]) == [1, 1]
    assert list(out["evt_OPEC"]) == [0, 0], "OPEC date announced after the 10-05 snapshot must not count"


@pytest.fixture(scope="module")
def world():
    return _factor_world()


def test_lasso_selects_the_planted_factor(world):
    data, days = world
    est = FactorHARVol(["signal", "noise0", "noise1", "noise2"], horizon=1)
    est.estimate(data, settle_time(days[1500]) + pd.Timedelta(minutes=35))
    assert "signal" in est.selected_
    assert len(est.selected_) <= 3


def test_factor_har_beats_plain_har_out_of_sample(world):
    data, days = world
    # Score against the TRUE planted variance (a single squared return is far too noisy a target)
    names = ["signal", "noise0", "noise1", "noise2"]
    err_f, err_h = [], []
    for i in range(1100, 1590, 10):
        as_of = settle_time(days[i]) + pd.Timedelta(minutes=35)
        target = data.true_daily_var[i]
        f = FactorHARVol(names, horizon=1).estimate(data, as_of) ** 2 / 252
        h = HARVol(horizon=1).estimate(data, as_of) ** 2 / 252
        err_f.append((f - target) ** 2)
        err_h.append((h - target) ** 2)
    assert np.mean(err_f) < 0.8 * np.mean(err_h)


def test_factor_har_feeds_bates(world):
    data, days = world
    as_of = settle_time(days[1500]) + pd.Timedelta(minutes=35)
    fv = FactorHARVol(["signal"], horizon=5)
    m = BatesModel(n_paths=2000, variance_forecaster=fv)
    m.fit(data, as_of)
    assert m.params.v0 == pytest.approx(fv.estimate(data, as_of) ** 2)


# ---- direction test ---------------------------------------------------------------

def _direction_data(signal_strength, n=2000, seed=1):
    g = np.random.default_rng(seed)
    X = pd.DataFrame({"a": g.normal(size=n), "b": g.normal(size=n)})
    y = pd.Series(signal_strength * X["a"] + g.normal(size=n))
    return y, X


def test_direction_test_finds_planted_signal():
    y, X = _direction_data(0.3)
    out = directional_test(y, X, n_boot=500)
    assert out["r2_oos"] > 0.03 and out["lo"] > 0
    assert out["n"] == 1500


def test_direction_test_finds_nothing_in_noise():
    y, X = _direction_data(0.0)
    out = directional_test(y, X, n_boot=500)
    assert out["r2_oos"] < 0.01
    assert out["lo"] < 0 < out["hi"] or out["hi"] <= 0
