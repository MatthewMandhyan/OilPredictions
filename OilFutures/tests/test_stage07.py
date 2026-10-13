# PROVIDED · Stage 7 checks: run with  pytest tests/test_stage07.py -v
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from evaluation.fills import maker_filled, order_pnl
from layers.trading_rules import Order, TradingRules, evaluate_rules, walk_forward_tune

RULES = TradingRules(bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25, max_position_frac=0.02,
                     max_daily_exposure_frac=0.10, cheap_threshold=0.15, cheap_extra_edge=0.03,
                     maker_default=True, taker_rate=0.07, maker_rate=0.0175)


def test_fee_formula_and_rounding():
    assert RULES.fee(0.50, 10, maker=False) == pytest.approx(0.18)    # 0.175 rounds UP
    assert RULES.fee(0.50, 4, maker=False) == pytest.approx(0.07)     # exactly 0.07, no extra cent
    assert RULES.fee(0.10, 1, maker=True) == pytest.approx(0.01)


def test_fee_placeholder_raises():
    with pytest.raises(ValueError):
        TradingRules().fee(0.5, 1, maker=False)


def test_required_edge():
    assert RULES.required_edge(0.40) == pytest.approx(0.03)
    assert RULES.required_edge(0.10) == pytest.approx(0.06)


def test_kelly():
    assert TradingRules.kelly_binary(0.60, 0.50) == pytest.approx(0.20)
    assert TradingRules.kelly_binary(0.40, 0.50) == 0.0


def test_no_trade_when_edge_below_threshold():
    assert RULES.decide("K", 0.52, 0.49, 0.51) is None          # edge ~0.03 minus fee < 0.03


def test_yes_trade_as_maker_uses_bid():
    o = RULES.decide("K", 0.60, 0.45, 0.48)
    assert o is not None and o.side == "yes" and o.maker and o.price == pytest.approx(0.45)
    assert o.edge == pytest.approx(0.60 - 0.45 - 0.0175 * 0.45 * 0.55)


def test_no_trade_side_when_model_is_low():
    o = RULES.decide("K", 0.30, 0.45, 0.48, maker=False)
    assert o.side == "no" and not o.maker and o.price == pytest.approx(0.55)
    assert o.prob_win == pytest.approx(0.70)


def test_sizing_respects_position_cap():
    o = RULES.decide("K", 0.90, 0.40, 0.42)
    assert o.price * o.contracts <= RULES.max_position_frac * RULES.bankroll + 1e-9
    assert o.contracts == 50                                  # $20 cap / $0.40


def test_sizing_uses_fractional_kelly_when_smaller():
    r = replace(RULES, max_position_frac=1.0)
    o = r.decide("K", 0.50, 0.44, 0.46)
    expected_budget = 0.25 * TradingRules.kelly_binary(0.50, 0.44) * 1000
    assert o.contracts == int(np.floor(expected_budget / 0.44))


def test_cheap_contract_needs_extra_edge():
    # 5-point raw edge: enough at 40¢, not enough at 10¢ (needs 6 points there)
    assert RULES.decide("K", 0.45, 0.40, 0.42) is not None
    assert RULES.decide("K", 0.15, 0.10, 0.12) is None


def test_daily_cap():
    orders = [Order("A", "yes", 0.40, 50, True, 0.1, 0.5), Order("B", "yes", 0.50, 40, True, 0.1, 0.6),
              Order("C", "no", 0.60, 100, True, 0.1, 0.7)]
    kept = RULES.apply_daily_cap(orders)
    assert sum(o.price * o.contracts for o in kept) <= 100.0 + 1e-9
    assert {o.ticker for o in kept} == {"A", "B", "C"}
    assert RULES.apply_daily_cap(orders[:1]) == orders[:1]


def test_maker_fill_requires_trade_through():
    yes = Order("K", "yes", 0.40, 1, True, 0.05, 0.5)
    assert not maker_filled(yes, pd.DataFrame({"yes_price": [0.41, 0.40]}))
    assert maker_filled(yes, pd.DataFrame({"yes_price": [0.41, 0.39]}))
    no = Order("K", "no", 0.55, 1, True, 0.05, 0.6)
    assert maker_filled(no, pd.DataFrame({"yes_price": [0.46]}))
    assert not maker_filled(no, pd.DataFrame({"yes_price": [0.45, 0.30]}))


def test_order_pnl():
    o = Order("K", "yes", 0.40, 3, False, 0.05, 0.5)
    assert order_pnl(o, 0, 0.06) == pytest.approx(-1.26)
    assert order_pnl(o, 1, 0.06) == pytest.approx(1.74)
    n = Order("K", "no", 0.70, 2, False, 0.05, 0.8)
    assert order_pnl(n, 0, 0.0) == pytest.approx(0.60)


def _rows(seed=0, n_days=240, per_day=5):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2025-01-01", periods=n_days):
        for j in range(per_day):
            p_true = rng.uniform(0.1, 0.9)
            mid = np.clip(p_true + rng.normal(0, 0.08), 0.03, 0.97)
            rows.append({"date": d, "ticker": f"{d:%Y%m%d}-{j}", "probability": p_true,
                         "yes_bid": mid - 0.01, "yes_ask": mid + 0.01, "outcome": int(rng.random() < p_true)})
    return pd.DataFrame(rows)


def test_evaluate_rules_runs():
    tr = evaluate_rules(_rows(n_days=20), RULES)
    assert {"pnl", "contracts", "side"} <= set(tr.columns) and len(tr) > 0


def test_walk_forward_tune_uses_only_the_past():
    rows = _rows()
    grid = [0.02, 0.05, 0.10]
    a = walk_forward_tune(rows, RULES, grid, block_days=60)
    assert len(a) == 4 and a["chosen_min_edge"].iloc[0] == RULES.min_edge
    assert set(a["chosen_min_edge"].iloc[1:]) <= set(grid)
    # flip every outcome in the LAST block: earlier choices must not change
    last_dates = sorted(rows["date"].unique())[180:]
    changed = rows.copy()
    m = changed["date"].isin(last_dates)
    changed.loc[m, "outcome"] = 1 - changed.loc[m, "outcome"]
    b = walk_forward_tune(changed, RULES, grid, block_days=60)
    assert list(a["chosen_min_edge"]) == list(b["chosen_min_edge"])
    assert a["oos_pnl"].iloc[-1] != b["oos_pnl"].iloc[-1]
