# ERCOT wind forecast error and the DART spread

I wanted to know whether a miss in ERCOT's wind forecast moves the gap between
the day-ahead and real-time price (the DART spread) at HB_HOUSTON, and whether
that relationship is nonlinear enough to need more than linear regression.

There are two datasets here. Most of the pipeline was built and tested on
synthetic data with a known relationship (`src/generate_synthetic_data.py`).
I then ran OLS and LightGBM on real ERCOT data, but ERCOT's public real-time
price report only keeps about 7 days, so the real sample is 166 hours.

## Running it

```bash
pip install -r requirements.txt

# synthetic pipeline
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

# real data (raw pulls are already in data/real_ercot_raw/)
python src/clean_real_data.py
python src/model_real.py
```

`pull_ercot_data.py` re-pulls the raw data with `gridstatus`. ERCOT blocks a lot
of cloud IPs, so it needs to run on a normal machine.

## Results

Synthetic data, chronological 80/20 split:

```
                 OOS R²   RMSE ($/MWh)
Naive (mean)       -        13.57
OLS              0.691       7.53
LightGBM         0.874       4.81
```

Walk-forward over 12 monthly folds gave R² 0.830. The 10/50/90 quantile models
landed within 1.6 points of their targets. SHAP shows the forecast-error slope
is about 4x steeper above the 1,200 MW kink, which is what the generator built
in. Newey-West standard errors were 282% larger than plain OLS ones.

Real data (Oct 4-10, 2026, 124 train / 42 test hours): OLS R² was -4.53 and
LightGBM -5.76, both worse than predicting the training mean. The training
window has a -$226/MWh hour (std $27.9) and the test window is calm (std $7.9),
so this mostly says 166 hours is too few. The forecast-error coefficient was
not significant (HAC p = 0.113). A tradeable version that only uses forecast
revisions published 6h and 1h before each hour was significant in sample
(p = 0.040) but still had negative OOS R² (-3.51).

## Limits

The synthetic numbers come from a generator I wrote, so they show the pipeline
works, not that the relationship is this strong in real markets. The 73% hit
rate in the toy backtest has no costs or real bidding mechanics. A real answer
needs months of data, which means collecting it daily over time.
