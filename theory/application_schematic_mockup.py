"""Mockup: detailed full-window schematic (edits 1-7). Throwaway."""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAPPH = '#bcdcee'
BLACK = '#1a1a1a'
GREEN = '#2ca02c'
RED = '#c1272d'

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis('off')

# --- media ---
ax.add_patch(Rectangle((2.5, 0), 4.0, 5, facecolor=SAPPH, edgecolor='none', zorder=1))
coat_cmap = LinearSegmentedColormap.from_list('coat', [SAPPH, 'white'])
grad = np.linspace(0, 1, 200).reshape(1, -1)
ax.imshow(grad, extent=[6.5, 7.6, 0, 5], aspect='auto', cmap=coat_cmap, zorder=1)
ax.plot([2.5, 2.5], [0, 5], color=BLACK, lw=1.5, zorder=3)        # front surface
ax.plot([7.6, 7.6], [0, 5], color=BLACK, lw=0.8, zorder=3)        # coating/air edge
ax.text(1.2, 4.6, 'air', fontsize=12, ha='center', style='italic')
ax.text(4.3, 4.6, r'sapphire  $n_b$', fontsize=12, ha='center')
ax.text(9.4, 4.6, 'air', fontsize=12, ha='center', style='italic')
ax.text(7.05, 4.55, r'$\epsilon(x)$', fontsize=12, ha='center')   # upright, above coating

# --- rays ---
front = (2.5, 3.2); coat = (6.5, 2.45)
# incident (black)
ax.add_patch(FancyArrowPatch((0.4, 4.6), front, arrowstyle='-|>', mutation_scale=18, lw=2.2, color=BLACK, zorder=4))
# front reflection R_f (green)
ax.add_patch(FancyArrowPatch(front, (0.55, 2.1), arrowstyle='-|>', mutation_scale=16, lw=2.0, color=GREEN, zorder=4))
ax.text(0.35, 1.75, r'$R_{\rm f}$', fontsize=12, color=GREEN)
# refracted (black) -> coating
ax.add_patch(FancyArrowPatch(front, coat, arrowstyle='-|>', mutation_scale=16, lw=2.2, color=BLACK, zorder=4))
# transmitted through coating, exits right (black)
ax.add_patch(FancyArrowPatch(coat, (7.6, 2.25), arrowstyle='-', lw=2.0, color=BLACK, zorder=4))
ax.add_patch(FancyArrowPatch((7.6, 2.25), (9.7, 1.55), arrowstyle='-|>', mutation_scale=16, lw=2.0, color=BLACK, zorder=4))
ax.text(9.5, 1.2, r'$T$', fontsize=12, color=BLACK)
# back reflection R (~0, the minimized quantity) — thin dashed
ax.add_patch(FancyArrowPatch(coat, (2.5, 1.5), arrowstyle='-|>', linestyle='--', mutation_scale=12, lw=1.4, color=BLACK, zorder=4))
ax.add_patch(FancyArrowPatch((2.5, 1.5), (0.55, 0.8), arrowstyle='-|>', linestyle='--', mutation_scale=12, lw=1.4, color=BLACK, zorder=4))
ax.text(0.35, 0.45, r'$R\approx0$', fontsize=12, color=BLACK)

# --- normal + angle arcs at front ---
ax.plot([1.7, 3.3], [3.2, 3.2], color=BLACK, lw=1, ls=':', zorder=3)
ax.add_patch(Arc(front, 1.3, 1.3, theta1=27, theta2=66, color=BLACK, lw=1.3))
ax.text(3.0, 4.0, r'$\theta_{\rm air}$', fontsize=12)
ax.add_patch(Arc(front, 2.1, 2.1, theta1=-12, theta2=0, color=BLACK, lw=1.3))
ax.text(3.75, 2.85, r'$\theta_{n_b}$', fontsize=11)

# --- theory-region box (red dashed) ---
ax.add_patch(FancyBboxPatch((4.9, 0.35), 4.5, 4.3, boxstyle='round,pad=0.02',
                            fill=False, edgecolor=RED, lw=2.0, ls=(0, (6, 3)), zorder=5))
ax.text(7.15, 4.95, r'modeled in theory:  semi-$\infty\ n_b$ | coating | air',
        fontsize=10.5, color=RED, ha='center', va='bottom')

fig.tight_layout()
out = os.path.join(_PROJECT_ROOT, 'theory', '_mockup_schematic.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)
