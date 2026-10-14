# ERCOT Renewable Forecast-Error Pricing Model

> ## ⚠️ This repo has TWO datasets — read both result sections before citing anything
> Most of the numbers below (R²=0.87, 73% hit rate, etc.) come from a
> **fabricated dataset built to have known statistical properties**
> ([`src/generate_synthetic_data.py`](src/generate_synthetic_data.py)),
> built to validate that the modeling pipeline itself — leakage handling,
> walk-forward splits, HAC SEs, quantile calibration — works correctly.
> Real ERCOT data was later pulled and run through the same pipeline (see
> [Results (real data)](#results-real-data--read-this-before-the-synthetic-section-above)),
> but public retention windows limit the real, fully-overlapping sample to
> **166 hours**, not 90 days — too small to validate or refute the
> underlying relationship. Treat the synthetic section as "the pipeline
> works," the real section as "here's what happens when you point that
> same pipeline at real, messy, 7-day data," and neither as "this strategy
> makes money." See
> [Honest limitations](#honest-limitations-read-this-before-citing-any-number)
> before repeating any number from this README out loud.

## The question

Does a wind-generation forecast miss in ERCOT predictably move the gap
between the day-ahead and real-time price (the "DART spread"), and is that
relationship nonlinear enough to justify a model more complex than linear
regression?

## Why this design

- **Target:** `DART_spread(h) = DA_price(h) − RT_price(h)`, not absolute
  price — this isolates the piece of price movement caused by forecasts
  being wrong, which is the actual thing a trading desk can act on.
- **Core predictor:** `forecast_error(h) = actual_wind(h) − forecast_wind(h)`.
- **Leakage distinction (the thing to get right, real-data or not):** using
  the *same-hour actual* wind output is explanatory, not tradeable — you
  can't know actual output before real-time settles. A genuinely tradeable
  version would need ERCOT's rolling forecast *updates* (known before
  settlement), not the actual. This repo's synthetic version only builds
  the explanatory case; the real-data version (see below) needs to make
  this distinction explicit and should build both.
- **Three models, increasing complexity, each justified by what the last
  one couldn't do:**
  1. **OLS** — interpretable, signed coefficient, the benchmark everything
     else has to beat.
  2. **LightGBM** — only worth the added complexity if it meaningfully
     beats OLS out-of-sample. (It does here — see results — but that's
     expected, since the synthetic generator has a real nonlinear kink
     built into it on purpose, to test whether the pipeline can recover a
     known-true nonlinearity. On real data this comparison could easily
     go the other way, and that would be a fine result to report.)
  3. **Quantile regression (LightGBM, τ=0.1/0.5/0.9)** — tail risk, not
     just the mean.

## What this demonstrates (for resumes/interviews)

| Area | Where it shows up |
|---|---|
| **Markets / finance** | DART spread as the target (the actual tradeable gap, not raw price); explicit day-ahead vs. real-time market structure; the explanatory-vs-tradeable distinction (what's knowable *before* settlement vs. after — the real constraint any desk operates under); a toy directional backtest with hit rate, PnL, and an honest list of what it's missing (costs, position sizing, real market mechanics) |
| **Math / statistics** | ADF stationarity test, Ljung-Box autocorrelation test, Newey-West (HAC) robust standard errors with a quantified 282% SE inflation, GARCH(1,1) + Engle's ARCH-LM prerequisite test (including catching and correcting my own misread of a model-degeneracy result), block bootstrap confidence intervals under autocorrelation |
| **Data science** | Cleaning a real, messy, rolling-republish ERCOT report (dedup across thousands of overlapping snapshots), building a leakage-safe join on *publish time*, not just event time, aggregating 15-minute real-time prices to hourly, merging four independent data sources on overlapping real-world windows with honest handling of a much-smaller-than-planned sample |
| **Machine learning** | LightGBM vs. OLS with a justified complexity escalation, quantile regression (τ=0.1/0.5/0.9) with a calibration check, SHAP values as an independent cross-check on a learned nonlinearity, walk-forward (expanding-window) validation with explicit leakage assertions |

**One line for a resume:** *Built an ERCOT wind-forecast-error pricing
model (OLS → LightGBM → quantile regression) with walk-forward validation,
HAC/GARCH robustness checks, and a leakage-safe tradeable-signal
construction from real ERCOT data; validated the full pipeline on a
synthetic dataset with known ground truth before running it on real,
production data pulled via the ERCOT API.*

## Results (synthetic data)

| Model | OOS R² | OOS RMSE ($/MWh) | vs. naive baseline |
|---|---|---|---|
| Naive (predict train mean) | — | 13.57 | — |
| OLS | 0.691 | 7.53 | 44% better |
| LightGBM | 0.874 | 4.81 | 65% better, 36% better than OLS |

- **Walk-forward (12 monthly folds, expanding window, strict no-leakage
  assertion on every fold):** R² = 0.830, RMSE = $5.69/MWh across 8,160
  out-of-sample hours.
- **LightGBM correctly recovers the built-in nonlinear kink:** predicted
  slope is ~3x steeper above the 1,200 MW kink threshold than below it,
  matching the generator's ground truth (see `outputs/charts/01_nonlinear_kink.png`).
- **Quantile regression is well-calibrated out-of-sample:** 10th/50th/90th
  percentile coverage lands within 1.6 percentage points of target, and
  zero monotonicity violations across 1,920 test hours.
- **Toy directional backtest** (73% hit rate, Sharpe-like ratio ~39,
  no transaction costs): see [Honest limitations](#honest-limitations-read-this-before-citing-any-number)
  below before reading anything into these numbers.

Charts: `outputs/charts/01_nonlinear_kink.png`,
`02_actual_vs_predicted.png`, `03_quantile_band.png`,
`04_cumulative_pnl.png`, `05_shap_dependence.png`.

## Results (real data — read this before the synthetic section above)

Real ERCOT data was pulled successfully (see [Getting real data](#getting-real-data-next-step-not-yet-done)
below — this is now done, not a next step). After leakage-safe cleaning
(dedup to one settled actual + one pre-settlement forecast per hour) and
joining wind + day-ahead price + real-time price + load on `HB_HOUSTON`,
the fully-overlapping real window is **166 hours (Oct 4–10, 2026)** —
not 90 days. ERCOT's public real-time price report only retains 7 days of
history; that, intersected with settled wind hours, is the real
constraint, not a choice. See `src/clean_real_data.py` for the exact dedup
logic and `src/model_real.py` for the model run.

| Model | OOS R² (n_test=42) | OOS RMSE ($/MWh) |
|---|---|---|
| Naive (predict train mean) | — | 13.51 |
| OLS (HAC-robust SE) | **-4.53** | 18.35 |
| LightGBM (small-n settings) | **-5.76** | 20.30 |

Both real models did **worse** than just predicting the training mean.
Before concluding "the signal doesn't exist on real data," look at *why*:
the 124-hour train window includes one -$226/MWh outlier hour (train std
≈ $28), while the 42-hour test window is a calm 1.5-day stretch with no
outliers at all (test std ≈ $8, mean $8.34 vs train mean -$2.69). A model
trained on a volatile window and evaluated on a calm one will produce a
negative R² almost mechanically — this is a **train/test regime shift
from having only 166 hours to split**, not evidence the OLS coefficient
is garbage. For what it's worth, the OLS coefficient on `forecast_error_mw`
is still the correct sign (+0.0041, i.e. more wind than forecast → wider
DART spread, matching the synthetic model's direction) but is not
significant at 166 observations (HAC p=0.113) — which is itself an honest,
correctly-reported null result: **166 hours is not enough data to
establish this relationship, full stop.**

**What this real run is good for in an interview:** proof that the
end-to-end pipeline — leakage-safe forecast/actual dedup on a rolling
republish report, hub-matched price joins, 15-min→hourly aggregation,
HAC-robust inference, bootstrap CIs on small samples — runs correctly on
real, messy ERCOT data, not just a clean synthetic generator. **What it is
NOT good for:** claiming the forecast-error → DART-spread relationship is
proven or disproven. That needs months of real data, which the public
retention windows don't expose without a paid data subscription or an
automated daily collector run over time (see Limitations below).

Quantile regression, GARCH/ARCH-LM, and block-bootstrap backtest
significance were **not rerun** on real data — all three need a few
hundred+ points to produce a meaningful result, and running them on 166
rows (42 in the test fold) would itself be the kind of overclaim this
README has been explicit about avoiding throughout. That's a stated scope
limit, not an oversight.

### Explanatory vs. tradeable, head-to-head on real data

This was the one gap left unbuilt earlier in the project, and it's the
part worth leading with in an interview. `forecast_error_mw` needs
same-hour *actual* wind output — unknowable before real-time settles, so
no desk could ever observe it before the decision point. `src/clean_real_data.py`
also builds `forecast_revision_mw`: the change in ERCOT's published
forecast between (interval start − 6h) and (interval start − 1h), using
`pd.merge_asof` to pull only forecast snapshots that existed at those
times — a number any desk genuinely could compute and act on before
settlement.

| Feature | Needs the actual? | In-sample HAC p-value | OOS R² (n_test≈40) |
|---|---|---|---|
| `forecast_error_mw` (explanatory) | Yes — not tradeable | 0.113 (not significant) | -4.53 |
| `forecast_revision_mw` (tradeable) | No — genuinely tradeable | **0.040 (significant)** | -3.51 |

The forecast-*revision* coefficient is actually significant in-sample
(HAC-robust, 166 hours) — an upward revision to the wind forecast
associates with a wider DART spread. That's a real, mechanistically
sensible real-data finding: more wind than the market had priced in tends
to push real-time price down relative to day-ahead. It did NOT hold up
out-of-sample here, for the same train/test regime-shift reason described
above — 42 test hours is not enough to confirm an in-sample p-value
survives, and that gap between "significant in-sample" and "doesn't hold
OOS" is itself a textbook small-sample-overfitting lesson worth explaining
if asked.

## Advanced statistics layer

Added on top of the base pipeline — these are the checks that separate
"ran a regression" from "understands what the regression assumes":

- **Stationarity (ADF test):** spread series is stationary (ADF stat
  -48.4, p≈0) — safe to model directly, no differencing needed.
- **Autocorrelation (Ljung-Box, 24h lag):** statistically significant
  (p=0.046) — hourly spread values aren't independent, which breaks a
  core OLS assumption.
- **Newey-West (HAC) robust standard errors:** correcting for that
  autocorrelation **inflates the forecast-error coefficient's standard
  error by 282%** — plain OLS was understating uncertainty by a wide
  margin. The coefficient itself is unchanged and remains significant
  after correction (p≈3×10⁻²⁴⁹), but the 282% number is the one worth
  leading with in an interview — it shows the naive result was
  overconfident, not wrong.
- **GARCH(1,1) volatility clustering — an honest null result:** ran
  Engle's ARCH-LM test first (p=0.59, no significant ARCH effects), as
  the correct prerequisite before trusting any GARCH fit. The GARCH(1,1)
  model itself converged to alpha≈0 (p=0.999) with beta≈0.996 — a known
  estimation degeneracy on data with no real volatility clustering, not
  evidence of clustering. Reporting beta's p-value alone here would have
  been a methodological error; the full writeup in `src/model_garch.py`
  catches and explains it. This null result is *expected and correct* —
  the synthetic generator draws noise independently each hour, so there's
  genuinely nothing to find. Real ERCOT data, where weather events span
  multiple hours, would likely show real clustering on the same test.
- **Block bootstrap on the backtest (24h blocks, 2000 resamples):** 73%
  hit rate has a 95% CI of [72.1%, 74.0%]. Notably, the block-bootstrap CI
  is **barely wider than a naive i.i.d. confidence interval** (-1.4%
  change) — unlike the continuous spread, the binary direction-correct
  indicator doesn't inherit much extra uncertainty from autocorrelation.
  This is a real, reported nuance, not a forced parallel to the HAC
  result.
- **SHAP values as a cross-check, not just nicer feature importance:**
  decomposed real predictions on held-out test data (independent of the
  synthetic probe-grid method used in the base pipeline) and found the
  same nonlinear kink — slope roughly 4x steeper above the 1,200 MW
  threshold. Two independent methods agreeing on the same nonlinearity is
  a stronger claim than either alone.

## Honest limitations (read this before citing any number)

1. **The synthetic results are not realistic.** The ~0.83 walk-forward R²
   and the Sharpe-like ratio of ~39 are artifacts of a synthetic generator
   with a much cleaner signal-to-noise ratio than any real market. **None
   of these specific numbers belong on a resume or in an interview
   answer** — the pipeline design and methodology do.
1b. **The real-data results (166 hours) don't settle anything either, in
   the other direction.** A negative OOS R² on 166 hours, most of it driven
   by one train-window outlier hour against a calm test window, is not
   evidence the forecast-error → DART-spread relationship is false any
   more than the synthetic 0.83 is evidence it's that strong. The honest
   conclusion from the real run is narrower: **the pipeline runs correctly
   on real data, and 166 hours is nowhere near enough to say anything
   about the underlying relationship.** Getting a defensible real-data
   answer needs a data collector that runs daily for months (ERCOT's
   public reports only expose a rolling few-days-to-weeks window — there's
   no way to pull 90 days of real history retroactively through the free
   public API), not a one-time pull.
2. **The toy backtest is not a trading strategy.** It ignores transaction
   costs, bid-ask spread, position sizing, and ERCOT's actual market
   mechanics (you submit MW quantities into the day-ahead market, not a
   directional bet). It's a check on directional skill, nothing more.
3. **Explanatory vs. tradeable is not yet built.** This version uses
   same-hour actual wind output, which isn't known before real-time
   settles. The real-data version needs a second model using ERCOT's
   hour-ahead forecast *revisions* instead, and both should be reported
   side by side with the distinction stated plainly.
4. **Separately-fit quantile models can cross in principle**, even though
   this run had zero violations — that's a known limitation of the
   approach, not a guarantee.
5. **Single hub, hourly only.** No attempt yet to generalize across ERCOT
   hubs or down to 5-minute real-time granularity.

## Getting real data (done)

Real ERCOT data access was blocked in the cloud environment this was
first built in (ERCOT's site rejects that network's egress and blocks
WebFetch via robots.txt). Resolved by running the puller on a local
Windows machine instead:

```
pip install gridstatus pandas
python pull_ercot_data.py
```

Needs a Python version with prebuilt `lxml` wheels (3.11–3.13; 3.14 did
not have one at build time — a Miniconda environment pinned to Python
3.12 with `conda install -c conda-forge lxml` is the reliable fix on
Windows if `pip install lxml` tries to compile from source and fails).
See `pull_ercot_data.py` for the exact `gridstatus` calls — wind/solar
actual+forecast, day-ahead and real-time settlement point prices, load by
weather zone — verified against the installed library source, not just
documentation.

The raw pulled CSVs land in `data/real_ercot_raw/`. From there:

```
python src/clean_real_data.py   # dedup, leakage-safe join -> data/REAL_ercot_hourly.csv
python src/model_real.py        # OLS + LightGBM on the real 166-hour window
```

`clean_real_data.py` is the part worth walking through in an interview:
the raw wind/solar report reposts hourly with a 48h-trailing/168h-forward
forecast window, so `GEN` (actual) is null until ~1-2h after the interval
starts and then locks constant, while `STWPF` (forecast) revises across
publishes. The script keeps only the last *settled* actual and the last
forecast published *strictly before* the interval started — getting that
ordering backwards is exactly the kind of leakage this whole project has
been built to avoid.

## Repo structure

```
src/
  generate_synthetic_data.py   # SYNTHETIC data generator (documented ground truth)
  model_ols.py                 # OLS baseline
  model_lgbm.py                # LightGBM, nonlinearity probe
  model_quantile.py            # Quantile regression, calibration check
  model_backtest.py            # Walk-forward validation, leakage assertion, toy signal
  make_charts.py               # 4 base charts
  stat_tests.py                # ADF stationarity + Ljung-Box autocorrelation tests
  model_ols_hac.py             # Newey-West HAC-robust standard errors
  model_garch.py               # ARCH-LM test + GARCH(1,1), honest null result
  backtest_significance.py     # Block bootstrap CI on hit rate / PnL
  model_shap.py                # SHAP values, independent cross-check on the kink
  clean_real_data.py           # Dedups raw ERCOT republish snapshots, leakage-safe join -> REAL_ercot_hourly.csv
  model_real.py                 # OLS + LightGBM on the real 166-hour window, honest small-n framing
data/
  SYNTHETIC_ercot_hourly.csv   # Generated data (regenerate via generate_synthetic_data.py)
  real_ercot_raw/               # Raw CSVs from pull_ercot_data.py (wind, solar, DAM, RTM, load)
  REAL_ercot_hourly.csv         # Cleaned/joined real data (166 hours, Oct 4-10 2026)
outputs/
  *.csv                         # Model results, predictions, backtest, stat test output
  real_data_results.csv         # OLS/LightGBM OOS results on real data
  charts/                       # 5 PNG charts, all titled "SYNTHETIC DATA"
pull_ercot_data.py              # Real ERCOT data puller (gridstatus-based) — run successfully on Windows/conda
```

## Reproducing this

```bash
pip install pandas numpy scikit-learn statsmodels lightgbm matplotlib arch shap gridstatus

# Synthetic pipeline
python src/generate_synthetic_data.py
python src/model_ols.py
python src/model_lgbm.py
python src/model_quantile.py
python src/model_backtest.py
python src/make_charts.py
python src/stat_tests.py
python src/model_ols_hac.py
python src/model_garch.py
python src/backtest_significance.py
python src/model_shap.py

# Real-data pipeline (needs pull_ercot_data.py's CSVs in data/real_ercot_raw/ first)
python src/clean_real_data.py
python src/model_real.py
```

Random seed is fixed (`np.random.seed(42)`) in the data generator for
reproducibility.
