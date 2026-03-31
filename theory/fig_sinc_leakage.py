"""
fig_sinc_leakage.py
====================
Shows the spectral one-sidedness of the Lorentzian profile.

The Lorentzian ε(x) = nb² - A·γ/(x + iγ) has a pole at x = -iγ (lower
half-plane), so by Paley-Wiener its FT is supported on k ≥ 0.

Normalization: by max(pwr[k > 0.1]) to exclude the DC spike.
DC spike (k=0) is off-scale and annotated.
"""

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
    'legend.fontsize': 10, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

GREEN = '#2ca02c'
RED   = '#d62728'

# ── Lorentzian parameters ────────────────────────────────────────────────────
a, gam, nb = 1.0, 0.1, 1.5

dx   = gam / 100
xmin = -gam * 200
xmax =  gam * 200
xx   = np.linspace(xmin, xmax, 1 + int(np.floor((xmax - xmin) / dx)))
ee   = tmm_h.eps(xx, a, gam, nb)
u    = np.real(ee)   # full u — includes the nb² DC offset
v    = np.imag(ee)

# ── Compute FoM and spectrum ─────────────────────────────────────────────────
fom, k_arr, Z_arr = tmm_h.skk_spectral_fom(
    xx, u, v, allowed_side='positive', derivative=False)

print(f"Spectral FoM = {fom:.2f}%")


def normalize_spectrum(k, Z, k_thresh=0.1):
    """Normalize by max power at |k| > k_thresh (exclude DC spike)."""
    pwr = np.abs(Z)**2
    mask = np.abs(k) > k_thresh
    return pwr / pwr[mask].max()


pwr = normalize_spectrum(k_arr, Z_arr)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

ax.fill_between(k_arr[k_arr >= 0], pwr[k_arr >= 0],
                alpha=0.35, color=GREEN, label=r'$k > 0$ (allowed)')
ax.fill_between(k_arr[k_arr <= 0], pwr[k_arr <= 0],
                alpha=0.35, color=RED, label=r'$k < 0$ (forbidden)')
ax.plot(k_arr, pwr, 'k-', linewidth=0.5, alpha=0.5)
ax.set_yscale('log')
ax.set_ylim(1e-8, 10)
ax.set_xlim(-50, 50)
ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
ax.set_ylabel(r'$|Z(k)|^2$ (normalized)')
ax.set_title(f'Lorentzian spectral one-sidedness\nSpectral FoM = {fom:.2f}%')
ax.legend(loc='upper right', fontsize=9)

# DC spike at k=0 is off-scale — annotate with downward arrow from top
ax.annotate(
    r'DC spike ($k=0$, off-scale)',
    xy=(0, 10), xytext=(12, 4),
    fontsize=8, ha='left',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    bbox=dict(facecolor='wheat', alpha=0.8, boxstyle='round'),
)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_sinc_leakage.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
