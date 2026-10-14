"""
Is the 73% hit rate / toy backtest PnL actually statistically significant,
or could it plausibly be noise? A single point estimate with no confidence
interval isn't a finished result — and because hourly PnL is autocorrelated
(same issue as the OLS standard errors), a naive binomial test on the hit
rate would understate uncertainty just like naive OLS SEs did.

Method: block bootstrap. Resample contiguous 24-hour blocks (not individual
hours) with replacement, rebuild a same-length synthetic backtest each time,
recompute hit rate and mean PnL, repeat 2000 times. The spread of those 2000
outcomes IS the honest confidence interval — it respects whatever
autocorrelation is actually in the data instead of assuming independence.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

bt = pd.read_csv("/home/claude/ercot_project/outputs/walk_forward_backtest.csv", parse_dates=["timestamp"])
bt = bt.sort_values("timestamp").reset_index(drop=True)

hit = bt["correct_direction"].values.astype(float)
pnl = bt["pnl"].values

BLOCK_SIZE = 24  # hours — matches the lag used in the HAC/Ljung-Box tests
N_BOOTSTRAP = 2000
N = len(bt)
n_blocks_needed = int(np.ceil(N / BLOCK_SIZE))

def block_bootstrap_sample(series, block_size, n_blocks, total_len):
    """Moving block bootstrap: resample starting indices for contiguous
    blocks with replacement, concatenate, trim to original length."""
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

# Bootstrap p-value: proportion of resamples where hit rate <= 50% (null:
# no directional skill) — this is the honest, autocorrelation-aware version
# of "is this better than a coin flip"
p_value_hit = (boot_hit_rates <= 0.5).mean()
p_value_pnl = (boot_mean_pnl <= 0).mean()
print(f"\nBootstrap p-value (hit rate <= 50%): {p_value_hit:.4f}")
print(f"Bootstrap p-value (mean PnL <= 0):    {p_value_pnl:.4f}")

# --- Compare to naive (iid-assumption) binomial CI, to show the same
# "autocorrelation inflates uncertainty" lesson as the HAC section ---
from scipy.stats import norm
naive_se = np.sqrt(actual_hit_rate * (1 - actual_hit_rate) / N)
naive_ci = [actual_hit_rate - 1.96*naive_se, actual_hit_rate + 1.96*naive_se]
block_se = (hit_ci[1] - hit_ci[0]) / (2 * 1.96)

inflation_pct = (block_se/naive_se - 1) * 100
print(f"\n=== Naive (iid) vs block-bootstrap CI width, hit rate ===")
print(f"Naive 95% CI (assumes independent hours): [{naive_ci[0]*100:.2f}%, {naive_ci[1]*100:.2f}%]")
print(f"Block bootstrap 95% CI (respects autocorrelation): [{hit_ci[0]*100:.1f}%, {hit_ci[1]*100:.1f}%]")
print(f"CI width change: {inflation_pct:+.1f}%")

if inflation_pct > 10:
    print("\nSame lesson as the HAC-corrected OLS: ignoring autocorrelation made")
    print("this result look more certain than it is. The block bootstrap number")
    print("is the one that's actually defensible.")
else:
    print("\nUnlike the continuous spread (which showed real autocorrelation),")
    print("the BINARY direction-correct indicator shows little to no extra")
    print("uncertainty from autocorrelation here. This is a genuinely")
    print("different, honest finding — not every quantity in this pipeline")
    print("inherits the same autocorrelation problem, and reporting that")
    print("nuance correctly matters more than forcing a consistent story.")

results = pd.DataFrame({
    "metric": ["hit_rate", "mean_pnl_per_hour"],
    "actual": [actual_hit_rate, actual_mean_pnl],
    "ci_low": [hit_ci[0], pnl_ci[0]],
    "ci_high": [hit_ci[1], pnl_ci[1]],
    "bootstrap_p_value": [p_value_hit, p_value_pnl],
})
results.to_csv("/home/claude/ercot_project/outputs/backtest_significance.csv", index=False)
print("\nSaved to outputs/backtest_significance.csv")
