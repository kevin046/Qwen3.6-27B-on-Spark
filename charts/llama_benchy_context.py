#!/usr/bin/env python3
"""Generate dark-themed line chart: Single-stream decode speed across context depths."""

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

depths = [0, 4096, 16384]
depth_labels = ["0", "4,096", "16,384"]

# tg=32
tg32 = [20.3, 22.5, 23.2]
# tg=128
tg128 = [17.0, 21.4, 20.9]
# tg=512
tg512 = [16.3, 16.7, 17.4]

ax.plot(depths, tg32, "o-", color="#4fc3f7", linewidth=2.5, markersize=9, label="tg=32", zorder=3)
ax.plot(depths, tg128, "s-", color="#ffb74d", linewidth=2.5, markersize=9, label="tg=128", zorder=3)
ax.plot(depths, tg512, "^-", color="#81c784", linewidth=2.5, markersize=9, label="tg=512", zorder=3)

ax.set_xlabel("Context Depth (tokens)", fontsize=12, fontweight="bold")
ax.set_ylabel("Decode Speed (t/s)", fontsize=12, fontweight="bold")
ax.set_title("Single-Stream Decode Speed Across Context Depths (c=1)", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(depths)
ax.set_xticklabels(depth_labels)
ax.set_ylim(12, 28)
ax.grid(True, color="#2a2a4a", linestyle="--", linewidth=0.7, zorder=0)
ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="#2a2a4e", edgecolor="#444466")

for data, color in [(tg32, "#4fc3f7"), (tg128, "#ffb74d"), (tg512, "#81c784")]:
    for i, v in enumerate(data):
        ax.annotate(f"{v:.1f}", xy=(depths[i], v), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=9, color=color)

plt.tight_layout()
plt.savefig("/tmp/Qwen3.6-27B-on-Spark/charts/llama_benchy_context.svg", format="svg", facecolor="#1a1a2e")
print("Saved llama_benchy_context.svg")
