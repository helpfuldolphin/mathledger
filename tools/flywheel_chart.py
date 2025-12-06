import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("artifacts")/"flywheel_timeseries.csv"
df = pd.read_csv(csv_path, parse_dates=["date"])
df = df.sort_values("date")

fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

# left axis
ax1.plot(df["date"], df["proofs_total"], linewidth=2, label="Cumulative proofs")
ax1.bar(df["date"], df["proofs_7d"], alpha=0.25, label="7-day proofs")

ax1.set_ylabel("Proofs (total / 7-day)")
ax1.set_xlabel("Date")

# right axis (percentages)
plotted_right = False
if "success_rate" in df.columns and df["success_rate"].notnull().any():
    ax2.plot(df["date"], df["success_rate"].astype(float)*100.0, color="tab:orange", linewidth=2, label="Success rate %")
    plotted_right = True
if "coverage" in df.columns and df["coverage"].notnull().any():
    ax2.plot(df["date"], df["coverage"].astype(float)*100.0, color="tab:green", linewidth=2, linestyle="--", label="Coverage %")
    plotted_right = True

if plotted_right:
    ax2.set_ylabel("Percent (%)")

ax1.grid(alpha=0.25)
lines, labels = ax1.get_legend_handles_labels()
if plotted_right:
    l2, lb2 = ax2.get_legend_handles_labels()
    lines += l2; labels += lb2
ax1.legend(lines, labels, loc="upper left")

out = Path("artifacts")/"flywheel_live_curve.png"
fig.tight_layout()
fig.savefig(out, dpi=180)
print(f"[OK] wrote {out}")
