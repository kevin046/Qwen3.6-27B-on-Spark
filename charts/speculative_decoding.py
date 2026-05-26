#!/usr/bin/env python3
"""Generate NEXTN speculative decoding explanation diagram."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'NEXTN Speculative Decoding — How It Works',
        ha='center', va='center', fontsize=16, fontweight='bold', color='white')
ax.text(7, 7.0, 'Lossless speedup via n-gram draft generation + parallel verification',
        ha='center', va='center', fontsize=11, color='#aaaaaa', style='italic')

# Step 1: Draft Generation
box1 = FancyBboxPatch((0.5, 4.8), 4, 1.6, boxstyle="round,pad=0.15",
                       facecolor='#1a3a5c', edgecolor='#4FC3F7', linewidth=2)
ax.add_patch(box1)
ax.text(2.5, 6.05, '① Draft Generation', ha='center', fontsize=12,
        fontweight='bold', color='#4FC3F7')
ax.text(2.5, 5.55, 'NEXTN generates 9 draft tokens\nper step using n-gram lookup\n(5 steps total per request)',
        ha='center', fontsize=9, color='#cccccc', linespacing=1.4)

# Step 2: Parallel Verification
box2 = FancyBboxPatch((5, 4.8), 4, 1.6, boxstyle="round,pad=0.15",
                       facecolor='#1a3a1a', edgecolor='#66BB6A', linewidth=2)
ax.add_patch(box2)
ax.text(7, 6.05, '② Parallel Verification', ha='center', fontsize=12,
        fontweight='bold', color='#66BB6A')
ax.text(7, 5.55, 'All draft tokens verified in\na single forward pass against\nthe target model',
        ha='center', fontsize=9, color='#cccccc', linespacing=1.4)

# Step 3: Accept/Reject
box3 = FancyBboxPatch((9.5, 4.8), 4, 1.6, boxstyle="round,pad=0.15",
                       facecolor='#3a1a1a', edgecolor='#FF7043', linewidth=2)
ax.add_patch(box3)
ax.text(11.5, 6.05, '③ Accept / Reject', ha='center', fontsize=12,
        fontweight='bold', color='#FF7043')
ax.text(11.5, 5.55, '~55% acceptance rate\n3.8 tokens accepted per step\nRejected tokens are discarded',
        ha='center', fontsize=9, color='#cccccc', linespacing=1.4)

# Arrows between steps
ax.annotate('', xy=(5.0, 5.6), xytext=(4.5, 5.6),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=2))
ax.annotate('', xy=(9.5, 5.6), xytext=(9.0, 5.6),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

# Bottom section: Token flow example
ax.text(7, 3.8, 'Token Flow Example (per step)', ha='center', fontsize=12,
        fontweight='bold', color='white')

# Draft tokens (blue boxes)
draft_tokens = ['D₁', 'D₂', 'D₃', 'D₄', 'D₅', 'D₆', 'D₇', 'D₈', 'D₉']
draft_status = ['✓', '✓', '✓', '✗', '✓', '✓', '✗', '✓', '✗']  # ~55% acceptance
accepted_colors = ['#2E7D32', '#2E7D32', '#2E7D32', '#C62828',
                   '#2E7D32', '#2E7D32', '#C62828', '#2E7D32', '#C62828']

for i, (tok, status, col) in enumerate(zip(draft_tokens, draft_status, accepted_colors)):
    x = 1.2 + i * 1.35
    box = FancyBboxPatch((x - 0.45, 2.5), 0.9, 0.9, boxstyle="round,pad=0.08",
                          facecolor='#1a2a3a', edgecolor=col, linewidth=2)
    ax.add_patch(box)
    ax.text(x, 3.05, tok, ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(x, 2.7, status, ha='center', fontsize=12, fontweight='bold', color=col)

# Labels
ax.text(0.3, 3.0, 'Draft →', ha='right', fontsize=10, color='#4FC3F7', fontweight='bold')

# Result arrow
ax.annotate('', xy=(7, 1.7), xytext=(7, 2.4),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

# Result
result_box = FancyBboxPatch((2, 0.6), 10, 1.0, boxstyle="round,pad=0.12",
                             facecolor='#1a1a3a', edgecolor='#FFD700', linewidth=2)
ax.add_patch(result_box)
ax.text(7, 1.35, 'Result: 5 tokens accepted in 1 forward pass (vs. 1 token without speculation)',
        ha='center', fontsize=11, fontweight='bold', color='#FFD700')
ax.text(7, 0.9, 'Effective speedup: ~3.8× per step  |  Net throughput gain: up to 2.4× vs. autoregressive baseline',
        ha='center', fontsize=9, color='#aaaaaa')

# Stats box at bottom
stats_box = FancyBboxPatch((0.5, -0.5), 13, 0.7, boxstyle="round,pad=0.1",
                            facecolor='#0d0d1a', edgecolor='#444444', linewidth=1)
ax.add_patch(stats_box)
ax.text(7, -0.15,
        'Algorithm: NEXTN (n-gram)  |  Steps: 5  |  Draft tokens/step: 9  |  '
        'Acceptance: ~55%  |  Avg accepted: 3.8/step  |  Quality: Lossless',
        ha='center', fontsize=9, color='#888888')

fig.savefig('/tmp/Qwen3.6-27B-on-Spark/charts/speculative_decoding.svg',
            format='svg', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("✓ speculative_decoding.svg generated")
