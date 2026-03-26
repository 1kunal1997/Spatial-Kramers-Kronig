"""Plot the Lorentzian dielectric profile used in the forward/backward FoM figure."""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tmm_helper as tmm_h

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

BLUE = '#1f77b4'
RED = '#8B0000'

a, gam, nb = 1.0, 0.1, 1.5
dx = gam / 100
xmin = -gam * 200
xmax = gam * 200
xx = np.linspace(xmin, xmax, 1 + int(np.floor((xmax - xmin) / dx)))
ee = tmm_h.eps(xx, a, gam, nb)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax2 = ax.twinx()
ax.plot(xx, ee.real, color=BLUE, lw=2.5)
ax2.plot(xx, ee.imag, color=RED, lw=2.5)
ax.set_xlabel(r'$x$ ($\mu$m)')
ax.set_ylabel(r"$\epsilon'$", color=BLUE)
ax2.set_ylabel(r"$\epsilon''$", color=RED)
ax.tick_params(axis='y', labelcolor=BLUE)
ax2.tick_params(axis='y', labelcolor=RED)
plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.savefig(os.path.join(FIGDIR, 'fig_lorentzian_profile.png'))
plt.close()
print('Saved fig_lorentzian_profile.png')
