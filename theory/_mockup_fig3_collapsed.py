"""Mockup: collapse Fig 3's 8 colorplots (4 thicknesses x 2 pol) into 2
master colorplots ratio(angle, d/lambda), one per pol. Throwaway.

Scale invariance: R depends only on (angle, d/lambda) for the shape-preserving
non-dispersive coating. So fix one 5um coating (k=8, matching the existing
5um panel) and sweep lambda such that d/lambda = 5/lambda spans [0.1, 100].
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tmm_helper as tmm_h
from skk_analysis_consolidated import R_nobulk_2D

nb = 1.7
delta = 0.01
T = 5.0                       # reference coating thickness (um)
k = 40.0 / T                  # = 8, matches existing 5um colorplot panel

# Shape-preserving coating, colorplot convention (domain +/- 20/k = +/- 2.5)
dx = 1 / (100 * k); xmin = -20 / k; xmax = -xmin
nx = 1 + int(np.floor((xmax - xmin) / dx))
xx = np.linspace(xmin, xmax, nx)
e_re = tmm_h.logistic(xx, k, nb)
e_im = tmm_h.ht_derivative(xx, e_re)
nc_skk, dc_skk = tmm_h.discretize_profile(xx, e_re + 1j * e_im, delta=delta)
nc_grin, dc_grin = tmm_h.discretize_profile(xx, e_re + 0j, delta=delta)

# d/lambda axis -> wavelength axis
dlam = np.logspace(np.log10(0.1), np.log10(100), 110)
lam = T / dlam                # lambda for each d/lambda
angles = np.arange(0, 90, 2.0)

pols = ['s', 'p']
ratio = {}
for pol in pols:
    print(f"computing {pol}-pol ...")
    R_s = R_nobulk_2D(nc_skk, dc_skk, nb, lam, angles, pol)    # [angle, lam]
    R_g = R_nobulk_2D(nc_grin, dc_grin, nb, lam, angles, pol)
    ratio[pol] = R_g / np.clip(R_s, 1e-15, None)

# ---- Plot: 1 row x 2 cols, shared d/lambda (y, log), shared colorbar ----
plt.rcParams.update({'font.size': 12})
norm = matplotlib.colors.LogNorm(vmin=1, vmax=1e5)
cmap = matplotlib.cm.inferno.copy(); cmap.set_over('white')
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
titles = {'s': r'$s$-polarization', 'p': r'$p$-polarization'}
for ax, pol in zip(axes, pols):
    im = ax.pcolormesh(angles, dlam, ratio[pol].T, norm=norm, cmap=cmap, shading='auto')
    ax.set_yscale('log')
    ax.axhline(1.0, color='cyan', lw=1.0, ls=':', alpha=0.7)
    ax.set_xlabel('Angle in Air (degrees)')
    ax.set_title(titles[pol], fontsize=13, fontweight='bold')
    ax.set_xlim(0, 88)
axes[0].set_ylabel(r'$d/\lambda$')
cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.9, pad=0.02, extend='max')
cbar.set_label(r'$R_{\mathrm{GRIN}} / R_{\mathrm{sKK}}$', fontsize=12)
out = os.path.join(_PROJECT_ROOT, 'theory', '_mockup_fig3_collapsed.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)

# Quick regime readout at normal incidence (angle index 0)
print('\nratio at normal incidence (s-pol) vs d/lambda:')
for target in (0.2, 0.5, 1, 2, 5, 20, 80):
    j = np.argmin(np.abs(dlam - target))
    print(f'  d/lam={dlam[j]:6.2f}: ratio={ratio["s"][0, j]:.2e}')
