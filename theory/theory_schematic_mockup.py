"""Theory schematic (Horsley Fig 1a style): isolated nb | coating | air.
Compact square. NOTE: schematics are faster in Inkscape/Illustrator — this is a
precise spec (angles 30/44/58 deg, colors, layout)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
from matplotlib.colors import LinearSegmentedColormap
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NB = '#bcdcee'; BLACK = '#1a1a1a'; GREEN = '#2ca02c'; BLUE = '#1f77b4'; GRAY = '#888888'


def arrow(ax, p0, p1, color, lw=2.0, head=True, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=('-|>' if head else '-'),
                                 mutation_scale=15, lw=lw, color=color, ls=ls,
                                 zorder=4, shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(4.8, 4.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 3.6); ax.set_ylim(0, 3.6); ax.set_aspect('equal'); ax.axis('off')

ax.add_patch(Rectangle((0, 0), 1.4, 3.6, facecolor=NB, edgecolor='none', zorder=1))
coat_cmap = LinearSegmentedColormap.from_list('coat', [NB, 'white'])
ax.imshow(np.linspace(0, 1, 200).reshape(1, -1), extent=[1.4, 2.4, 0, 3.6],
          aspect='auto', cmap=coat_cmap, zorder=1)
ax.plot([1.4, 1.4], [0, 3.6], color=BLACK, lw=1.0, zorder=3)
ax.plot([2.4, 2.4], [0, 3.6], color=BLACK, lw=1.0, zorder=3)
ax.text(0.7, 3.30, r'$n_b$', fontsize=14, ha='center')
ax.text(1.9, 3.30, r'$\epsilon(x)$', fontsize=12, ha='center')
ax.text(3.0, 3.30, 'air', fontsize=12, ha='center', style='italic')

hit = (1.4, 2.5); coat_exit = (2.4, 1.534)
arrow(ax, (0.274, 3.15), hit, BLACK)                  # incident
arrow(ax, hit, coat_exit, GRAY, head=False, ls='--')  # inner (unknown bending)
arrow(ax, coat_exit, (3.09, 0.434), BLUE)             # transmitted
ax.text(2.84, 0.40, r'$T$', fontsize=13, color=BLUE, ha='center')
arrow(ax, hit, (0.274, 1.85), GREEN)                  # reflected
ax.text(0.274, 1.60, r'$R$', fontsize=13, color=GREEN, ha='center')

ax.plot([0.9, 1.4], [2.5, 2.5], color=BLACK, lw=1, ls=':', zorder=3)
ax.add_patch(Arc(hit, 0.9, 0.9, theta1=150, theta2=180, color=BLACK, lw=1.4))
ax.text(0.80, 2.62, r'$\theta_b$', fontsize=13, ha='center')

# x-y axis L: moved right (under reflected ray) and up
arrow(ax, (0.65, 1.1), (1.0, 1.1), BLACK, lw=1.3)
arrow(ax, (0.65, 1.1), (0.65, 1.45), BLACK, lw=1.3)
ax.text(1.06, 1.05, r'$x$', fontsize=11); ax.text(0.65, 1.57, r'$y$', fontsize=11, ha='center')

out = os.path.join(_PROJECT_ROOT, 'theory', 'theory_schematic_mockup.png')
fig.savefig(out, dpi=150); print('saved', out)
