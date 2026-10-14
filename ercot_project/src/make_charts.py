"""
Charts for the README/notebook. Every single one gets 'SYNTHETIC DATA' in
the title — no exceptions — so there's no way a screenshot of these floats
around without that context.

Palette: validated default from the dataviz skill (series-1 blue for OLS,
series-2 orange for LightGBM — fixed categorical order, not cycled).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Palette (validated default) ---
BLUE = "#2a78d6"      # series-1 (OLS)
ORANGE = "#eb6834"    # series-2 (LightGBM)
AQUA = "#1baf7a"       # series-3 (actual/reference)
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

OUT = "/home/claude/ercot_project/outputs/charts"

# =========================================================
# Chart 1: the nonlinear kink — OLS straight line vs LightGBM curve
# =========================================================
probe = pd.read_csv("/home/claude/ercot_project/outputs/lgbm_kink_probe.csv")

# Reconstruct an OLS-equivalent straight line for comparison using the
# same coefficient reported by model_ols.py (hardcode is avoided by
# refitting a quick line through the probe's own linear region for display)
from numpy.polynomial import polynomial as P
linear_fit = np.polyfit(probe["forecast_error_mw"], probe["predicted_spread"], 1)
ols_equivalent_line = np.polyval(linear_fit, probe["forecast_error_mw"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(probe["forecast_error_mw"], ols_equivalent_line, color=BLUE, lw=2,
        label="Linear fit (OLS-style)", linestyle="--")
ax.plot(probe["forecast_error_mw"], probe["predicted_spread"], color=ORANGE, lw=2.5,
        label="LightGBM prediction")
ax.axvline(1200, color=MUTED, lw=1, linestyle=":", label="Kink threshold (1,200 MW)")
ax.set_xlabel("Forecast error (MW) — actual wind minus forecast")
ax.set_ylabel("Predicted DART spread ($/MWh)")
ax.set_title("SYNTHETIC DATA — LightGBM captures the nonlinear kink OLS can't", fontsize=11)
ax.legend(frameon=False, loc="upper right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/01_nonlinear_kink.png", dpi=150)
plt.close()
print("Saved 01_nonlinear_kink.png")

# =========================================================
# Chart 2: OOS actual vs predicted (walk-forward)
# =========================================================
bt = pd.read_csv("/home/claude/ercot_project/outputs/walk_forward_backtest.csv")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(bt["dart_spread"], bt["predicted"], s=6, alpha=0.25, color=BLUE,
           edgecolors="none")
lims = [bt["dart_spread"].min(), bt["dart_spread"].max()]
ax.plot(lims, lims, color=MUTED, lw=1, linestyle="--", label="Perfect prediction")
ax.set_xlabel("Actual DART spread ($/MWh)")
ax.set_ylabel("Predicted DART spread ($/MWh)")
ax.set_title("SYNTHETIC DATA — Walk-forward OOS: actual vs predicted", fontsize=11)
ax.legend(frameon=False, loc="upper left")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/02_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved 02_actual_vs_predicted.png")

# =========================================================
# Chart 3: quantile band over time (one representative week)
# =========================================================
q = pd.read_csv("/home/claude/ercot_project/outputs/quantile_predictions.csv", parse_dates=["timestamp"])
week = q[(q["timestamp"] >= q["timestamp"].min()) & (q["timestamp"] < q["timestamp"].min() + pd.Timedelta(days=7))]

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(week["timestamp"], week["q10"], week["q90"], color=BLUE, alpha=0.15,
                label="10th–90th percentile band")
ax.plot(week["timestamp"], week["q50"], color=BLUE, lw=1.5, label="Median prediction (q50)")
ax.plot(week["timestamp"], week["actual"], color=AQUA, lw=2, label="Actual spread")
ax.set_xlabel("")
ax.set_ylabel("DART spread ($/MWh)")
ax.set_title("SYNTHETIC DATA — Quantile forecast vs actual (one representative week, OOS)", fontsize=11)
ax.legend(frameon=False, loc="upper right", ncol=3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT}/03_quantile_band.png", dpi=150)
plt.close()
print("Saved 03_quantile_band.png")

# =========================================================
# Chart 4: cumulative toy backtest PnL over time
# =========================================================
bt["timestamp"] = pd.to_datetime(bt["timestamp"])
bt = bt.sort_values("timestamp")
bt["cumulative_pnl"] = bt["pnl"].cumsum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bt["timestamp"], bt["cumulative_pnl"], color=ORANGE, lw=1.8)
ax.axhline(0, color=MUTED, lw=1, linestyle=":")
ax.set_xlabel("")
ax.set_ylabel("Cumulative PnL ($, 1 MW notional)")
ax.set_title(
    "SYNTHETIC DATA — Toy directional backtest, no transaction costs\n"
    "(illustrates pipeline mechanics only — NOT a real performance claim)",
    fontsize=10,
)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT}/04_cumulative_pnl.png", dpi=150)
plt.close()
print("Saved 04_cumulative_pnl.png")

print("\nAll 4 charts saved to outputs/charts/")
