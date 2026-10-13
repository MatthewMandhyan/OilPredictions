# OilFutures: Kalshi WTI probability model (learning build)

Estimate calibrated probabilities for Kalshi WTI contracts, compare them with Kalshi
prices, and **paper-trade** the differences. One pipeline, swappable parts:

```
data (point-in-time) → volatility (Hist / HAR / OVX / events / factors)
      → engine (GBM | Heston | Bates | neural net) → market layer → trading rules
      → ledger → resolver → report
```

**Paper trading only.** Nothing in this project places real orders.

---

## Start here

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_stage01.py -v        # everything fails with NotImplementedError: that's the starting line
```

Open `models/gbm.py`, read `models/base.py` (provided), and start on Stage 1. A stage is
done when its test file is all green. Run `pytest -x tests/test_stage01.py` to stop at
the first failure.

## The stages

| # | Stage | Files you write in | Test |
|---|---|---|---|
| **Part 1: Foundations** | | | |
| 1 | GBM closed form, trading-day time, historical vol | `common/calendar.py`, `models/gbm.py` | `test_stage01.py` |
| 2 | GBM Monte Carlo, barrier (max/min) with daily monitoring | `models/gbm.py` | `test_stage02.py` |
| 3 | Scoring (Brier, log loss, calibration, block bootstrap) + walk-forward backtester | `evaluation/scoring.py`, `simulate.py`, `backtest.py` | `test_stage03.py` |
| **Part 2: Real data and live testing** | | | |
| 4 | Data pipeline, contract → futures month, `available_at`, lookahead guard | `pipeline/*.py`, `evaluation/lookahead.py` | `test_stage04.py` |
| 5 | Live ledger (append-only), resolver, report | `live/*.py` | `test_stage05.py` |
| **Part 3: Better volatility, then trade** | | | |
| 6 | HAR volatility, OVX implied vol, **event-aware volatility** | `models/har.py` | `test_stage06.py` |
| 7 | **Trading rules** (fees, edge, Kelly, caps) + fill model: *first paper-tradable system* | `layers/trading_rules.py`, `evaluation/fills.py` | `test_stage07.py` |
| 8 | Heston (stochastic variance) | `models/heston.py` | `test_stage08.py` |
| 9 | Bates (Heston + jumps) | `models/bates.py` | `test_stage09.py` |
| **Part 4: Research layers** | | | |
| 10 | Factor layer (HAR + LASSO on outside data) + **direction test** | `layers/factor_layer.py` | `test_stage10.py` |
| 11 | Market layer (crowd behavior, gap analysis) | `layers/market_layer.py` | `test_stage11.py` |
| 12 | *(optional)* Neural-net challenger (mixture density network) | `models/neural_net.py` | `test_stage12.py` |

### Running things with real data (from Stage 4)

```bash
export EIA_API_KEY=...   FRED_API_KEY=...         # free keys, links in config.yaml
python -m pipeline.futures --start 2015-01-01      # WTI settles by contract month
python -m pipeline.kalshi --series KXWTI --status settled --trades
python -m pipeline.fundamentals --start 2015-01-01 # Stage 10 inputs
python -m pipeline.sentiment --start 2018-01-01
python -m evaluation.real_backtest --series KXWTI  # GBM on real contracts, sliced report + calibration plot

# Stage 5: live paper loop, once per trading day after ~15:10 ET
python -m live.predict
python -m live.resolve
python -m live.report
```

Before trusting any P&L, fill `fees.taker_rate` / `fees.maker_rate` in `config.yaml` from
Kalshi's current fee schedule, and check `kalshi.*_series` and `futures.kalshi_roll_business_days`
against the contract terms.

---

## How to read the code

| Tag | Meaning |
|---|---|
| `# PROVIDED` | Finished plumbing (API calls, I/O, plotting). Read it, don't rewrite it. |
| `# YOUR TURN` | You implement this. The body is TODO steps ending in `raise NotImplementedError`. |
| `# HINT` / `# HINT 1` / `# HINT 2` | A nudge, then a more specific nudge. Try without them first. |
| `# WHY:` | The reasoning behind a design choice. |
| `# PITFALL:` | A mistake that's easy to make and hard to notice. |
| `# OOP:` | Why something is (or isn't) a class, and which OOP idea it shows. |
| `# SYNTAX:` | A Python pattern worth knowing, shown where it's useful. |

Formulas are in docstrings. Hints describe the idea and point to a reference; they
don't hand you the line of code.

## Conventions you must keep

* **Time:** every timestamp is naive `America/New_York` wall-clock time. WTI settles at 14:30 ET.
* **Point-in-time:** every data row has `date` (what it describes) and `available_at` (when it
  became public). A prediction at `as_of` may only use rows with `available_at < as_of`.
* **Clean daily comparison:** with free daily data, compare against Kalshi prices from the window
  just after the settle (`timing.snapshot_*` in config). That's when your F is fresh.
* **Walk-forward only:** fit on the past, predict the next window, roll.
* **Same code path:** backtest and live both call `model.predict(contract, as_of)`.
* **Append-only ledger:** never edit or delete a row.
* **Settings in `config.yaml`:** fees, edge, Kelly fraction, caps. Never hard-code them.

---

## OOP map: why the classes are where they are

| Class | OOP idea | Why it's a class |
|---|---|---|
| `Contract`, `Prediction`, `Order`, `HestonParams` | dataclass | Pure data with validation; frozen = can't be changed by accident. |
| `PriceModel` (abstract) → `GBMModel`, `HestonModel`, `BatesModel`, `NeuralNetModel` | inheritance + polymorphism | The backtester calls `.fit()`/`.predict()` on any engine without knowing which one it is. |
| `PriceModel.predict` | template method | Written once in the base; subclasses fill in `prob_terminal`/`prob_barrier`. |
| `VolatilityEstimator` → `HistoricalVol`, `HARVol`, `ImpliedVol`, `FactorHARVol` | strategy pattern | `GBMModel(vol=...)` accepts any of them, so you change sigma without touching GBM. |
| `EventAwareVol` | composition / decorator | Wraps *any* estimator and adds event variance on top. |
| `HestonModel` → `BatesModel` | inheritance + `super()` + hooks | Bates overrides two hook methods and reuses everything else. |
| `FuturesCurve`, `Ledger` | encapsulation | The only doors to the data are methods that enforce the rules (no lookahead; no edits). |
| `MarketAdjustedModel` | composition | HAS-an engine and HAS-a layer, and IS a `PriceModel` itself. |
| `KalshiClient`, `TradingRules` | state + `@classmethod` constructors | Settings stored once; `from_config(cfg)` builds them. |
| `evaluation/scoring.py` | **no classes** | Pure math functions. No state, so no class. |

## Python patterns you'll meet

| Pattern | Example | Where |
|---|---|---|
| List comprehension | `[x * 2 for x in xs]` | `walk_forward_windows`, `build_curve_frame` |
| Dict / set comprehension | `{m: g for m, g in df.groupby("month")}` | `FuturesCurve`, `parse_futures_month` |
| Lambda | `key=lambda o: o.edge`, `"gbm": lambda: GBMModel()` | `TradingRules.decide`, `MODEL_REGISTRY` |
| Generator (`yield`) | `for m in client.paginate(...)` | `KalshiClient.paginate` |
| `*args` / `**kwargs` | `self._prediction(c, p, t, F=70, sigma=0.3)` | `PriceModel._prediction`, `BatesModel.__init__` |
| Context manager | `with Ledger(path) as led:` | `live/*` |
| Vectorized numpy | `(paths[:, 1:].max(axis=1) > K).mean()` | Monte Carlo everywhere |
| Boolean masks | `df[df["available_at"] < as_of]` | everywhere point-in-time |
| `groupby` + named aggregation | `.agg(p_mean=("p", "mean"))` | `calibration_table` |
| Regular expressions | `re.findall(r"(August)\s+(\d{4})", text)` | `pipeline/contracts.py` |
| `try` / `except` | skip one bad market, keep the daily run alive | `live/predict.py` |

---

## Data sources

* **Kalshi API**: markets, trades (with taker side), settlements; live and `/historical/` endpoints.
* **WTI by contract month**: EIA RCLC1–4 (free, NYMEX, delayed) for history; Yahoo month symbols
  (e.g. `CLZ26.NYM`, unofficial) for same-day live prices. Paid (ICE, Databento, Nasdaq Data Link)
  for serious backtests. Kalshi's rules name **ICE** WTI.
* **EIA** weekly inventories · **CFTC** COT positioning · **FRED** (OVX, dollar, rates)
* **GDELT** news tone · **Google Trends** (manual CSV; see the rescaling PITFALL)
* **Event calendar** (`data/raw/events.csv`, columns `date,event_type,announced_at`): OPEC+
  meetings, FOMC decisions, EIA release dates. You maintain it.

## Papers

* Corsi (2009), HAR realized volatility (Stage 6)
* Bürgi, Deng & Whelan (2025), *Makers and Takers: The Economics of the Kalshi Prediction Market*, CEPR DP20631 (Stages 7, 11)
* Luo, Klein, Walther & Ji (2024), *Forecasting realized volatility of crude oil futures prices based on machine learning*, J. Forecasting 43(5) (Stage 10)
* Portnaya (2026), *Do Prediction Markets Match Option Prices?*, arXiv 2606.19517 (Stage 11)
* Akey et al. (2026), *Who Wins and Who Loses in Prediction Markets? Evidence from Polymarket* (Stage 11)
* *How uncertain are oil prices? Bayesian regularized distribution neural network*, Applied Economics (2025) (Stage 12)
* Optional: Broadie, Glasserman & Kou (1997), discrete barrier correction (Stage 2)
