"""
Block bootstrap confidence intervals for the toy backtest's hit rate and
mean PnL. Hourly results are autocorrelated, so resample 24-hour blocks
instead of single hours. 2000 resamples.
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)

bt = pd.read_csv(os.path.join(ROOT, "outputs", "walk_forward_backtest.csv"), parse_dates=["timestamp"])
bt = bt.sort_values("timestamp").reset_index(drop=True)

hit = bt["correct_direction"].values.astype(float)
pnl = bt["pnl"].values

BLOCK_SIZE = 24  # same lag as the HAC / Ljung-Box tests
N_BOOTSTRAP = 2000
N = len(bt)
n_blocks_needed = int(np.ceil(N / BLOCK_SIZE))


def block_bootstrap_sample(series, block_size, n_blocks, total_len):
    """Moving block bootstrap, trimmed to the original length."""
    max_start = len(series) - block_size
    starts = np.random.randint(0, max_start, size=n_blocks)
    blocks = [series[s:s+block_size] for s in starts]
    return np.concatenate(blocks)[:total_len]


boot_hit_rates = np.empty(N_BOOTSTRAP)
boot_mean_pnl = np.empty(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    sample_hit = block_bootstrap_sample(hit, BLOCK_SIZE, n_blocks_needed, N)
    sample_pnl = block_bootstrap_sample(pnl, BLOCK_SIZE, n_blocks_needed, N)
    boot_hit_rates[i] = sample_hit.mean()
    boot_mean_pnl[i] = sample_pnl.mean()

hit_ci = np.percentile(boot_hit_rates, [2.5, 97.5])
pnl_ci = np.percentile(boot_mean_pnl, [2.5, 97.5])

actual_hit_rate = hit.mean()
actual_mean_pnl = pnl.mean()

print("=== Block bootstrap (24-hour blocks, 2000 resamples) ===")
print(f"Actual hit rate: {actual_hit_rate*100:.1f}%")
print(f"95% CI (block bootstrap): [{hit_ci[0]*100:.1f}%, {hit_ci[1]*100:.1f}%]")
print(f"\nActual mean PnL/hour: ${actual_mean_pnl:.3f}")
print(f"95% CI (block bootstrap): [${pnl_ci[0]:.3f}, ${pnl_ci[1]:.3f}]")

# share of resamples at or below "no skill"
p_value_hit = (boot_hit_rates <= 0.5).mean()
p_value_pnl = (boot_mean_pnl <= 0).mean()
print(f"\nBootstrap p-value (hit rate <= 50%): {p_value_hit:.4f}")
print(f"Bootstrap p-value (mean PnL <= 0):    {p_value_pnl:.4f}")

# compare with the iid binomial CI
naive_se = np.sqrt(actual_hit_rate * (1 - actual_hit_rate) / N)
naive_ci = [actual_hit_rate - 1.96*naive_se, actual_hit_rate + 1.96*naive_se]
block_se = (hit_ci[1] - hit_ci[0]) / (2 * 1.96)

inflation_pct = (block_se/naive_se - 1) * 100
print(f"\n=== iid vs block-bootstrap CI width, hit rate ===")
print(f"iid 95% CI: [{naive_ci[0]*100:.2f}%, {naive_ci[1]*100:.2f}%]")
print(f"Block bootstrap 95% CI: [{hit_ci[0]*100:.1f}%, {hit_ci[1]*100:.1f}%]")
print(f"CI width change: {inflation_pct:+.1f}%")

results = pd.DataFrame({
    "metric": ["hit_rate", "mean_pnl_per_hour"],
    "actual": [actual_hit_rate, actual_mean_pnl],
    "ci_low": [hit_ci[0], pnl_ci[0]],
    "ci_high": [hit_ci[1], pnl_ci[1]],
    "bootstrap_p_value": [p_value_hit, p_value_pnl],
})
results.to_csv(os.path.join(ROOT, "outputs", "backtest_significance.csv"), index=False)
print("\nSaved to outputs/backtest_significance.csv")
