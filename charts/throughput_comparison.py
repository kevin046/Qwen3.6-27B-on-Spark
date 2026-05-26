#!/usr/bin/env python3
"""Generate throughput comparison bar chart for sglang benchmark profiles."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use('dark_background')

profiles = [
    "decode_only\n(0→512)",
    "few_shot\n(512→512)",
    "code\n(2048→256)",
    "long_output\n(512→4096)",
    "long_context\n(8192→128)",
    "mixed\n(1024→512)",
]
throughput = [152, 102, 72, 61, 52, 35]

fig, ax = plt.subplots(figsize=(12, 6.5))

colors = ['#4CAF50', '#66BB6A', '#FFA726', '#FF7043', '#EF5350', '#AB47BC']
bars = ax.bar(profiles, throughput, color=colors, width=0.65, edgecolor='#555555', linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val} t/s', ha='center', va='bottom', fontweight='bold',
            fontsize=13, color='white')

ax.set_ylabel('Throughput (tokens/second)', fontsize=13, fontweight='bold', color='#cccccc')
ax.set_title('sglang bench_serving — Throughput by Test Profile\nQwen3.6-27B-FP8 on NVIDIA DGX Spark (GB10)',
             fontsize=15, fontweight='bold', pad=15, color='white')
ax.set_ylim(0, max(throughput) * 1.2)
ax.tick_params(axis='x', labelsize=10, colors='#aaaaaa')
ax.tick_params(axis='y', labelsize=11, colors='#aaaaaa')
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.grid(axis='y', alpha=0.15, color='#888888')
ax.set_axisbelow(True)

# Subtitle annotation
ax.annotate('Speculative Decoding: NEXTN (5 steps, 9 draft tokens, ~55% acceptance)',
            xy=(0.5, -0.18), xycoords='axes fraction', ha='center',
            fontsize=10, color='#999999', style='italic')

fig.tight_layout()
fig.savefig('/tmp/Qwen3.6-27B-on-Spark/charts/throughput_comparison.svg',
            format='svg', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("✓ throughput_comparison.svg generated")
