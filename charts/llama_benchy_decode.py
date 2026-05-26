#!/usr/bin/env python3
"""Generate dark-themed bar chart: Token generation speed at depth=0 for c=1 and c=4."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "axes.edgecolor": "#444466",
})

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")

gen_lengths = [32, 128, 512]
x = np.arange(len(gen_lengths))
width = 0.22

# c=1
c1 = [20.3, 17.0, 16.3]
# c=4 total
c4_total = [57.3, 63.5, 60.6]
# c=4 per-req
c4_per_req = [18.2, 18.1, 16.5]

bars1 = ax.bar(x - width, c1, width, label="c=1 (single)", color="#4fc3f7", zorder=3)
bars2 = ax.bar(x, c4_total, width, label="c=4 total", color="#ffb74d", zorder=3)
bars3 = ax.bar(x + width, c4_per_req, width, label="c=4 per-request", color="#81c784", zorder=3)

ax.set_xlabel("Generation Length (tokens)", fontsize=12, fontweight="bold")
ax.set_ylabel("Token Generation Speed (t/s)", fontsize=12, fontweight="bold")
ax.set_title("Decode Throughput at Context Depth = 0", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(["32", "128", "512"])
ax.set_ylim(0, 78)
ax.grid(axis="y", color="#2a2a4a", linestyle="--", linewidth=0.7, zorder=0)
ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="#2a2a4e", edgecolor="#444466")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     fontsize=9, color="#e0e0e0")

plt.tight_layout()
plt.savefig("/tmp/Qwen3.6-27B-on-Spark/charts/llama_benchy_decode.svg", format="svg", facecolor="#1a1a2e")
print("Saved llama_benchy_decode.svg")
