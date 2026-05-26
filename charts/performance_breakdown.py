#!/usr/bin/env python3
"""Generate horizontal bar chart: custom benchmark vs official profiles."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

labels = [
    "LiveBench (generation)",
    "Custom Chat (40 req, 8 workers)",
    "decode_only (0→512)",
    "few_shot (512→512)",
    "code (2048→256)",
    "long_output (512→4096)",
    "long_context (8192→128)",
    "mixed (1024→512)",
]
values = [90, 88.15, 152, 102, 72, 61, 52, 35]

# Color by category
colors = []
for lbl in labels:
    if "Custom" in lbl:
        colors.append('#FFD700')  # Gold for custom benchmark
    elif "LiveBench" in lbl:
        colors.append('#00BCD4')  # Cyan for LiveBench
    elif "decode" in lbl:
        colors.append('#4CAF50')
    elif "few_shot" in lbl:
        colors.append('#66BB6A')
    elif "code" in lbl:
        colors.append('#FFA726')
    elif "long_output" in lbl:
        colors.append('#FF7043')
    elif "long_context" in lbl:
        colors.append('#EF5350')
    else:
        colors.append('#AB47BC')

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor='#555555', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2.,
            f'{val} t/s', ha='left', va='center', fontweight='bold',
            fontsize=12, color='white')

ax.set_xlabel('Throughput (tokens/second)', fontsize=13, fontweight='bold', color='#cccccc')
ax.set_title('Performance Breakdown — All Benchmarks\nQwen3.6-27B-FP8 on NVIDIA DGX Spark (GB10)',
             fontsize=15, fontweight='bold', pad=15, color='white')
ax.set_xlim(0, max(values) * 1.25)
ax.invert_yaxis()
ax.tick_params(axis='y', labelsize=10, colors='#aaaaaa')
ax.tick_params(axis='x', labelsize=11, colors='#aaaaaa')
ax.grid(axis='x', alpha=0.15, color='#888888')
ax.set_axisbelow(True)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FFD700', label='Custom benchmark'),
    Patch(facecolor='#00BCD4', label='LiveBench'),
    Patch(facecolor='#4CAF50', label='Official sglang profiles'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.3, edgecolor='#555555')

fig.tight_layout()
fig.savefig('/tmp/Qwen3.6-27B-on-Spark/charts/performance_breakdown.svg',
            format='svg', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("✓ performance_breakdown.svg generated")
