"""
fig_sinc_leakage.py
====================
Shows why pad_factor matters when the signal has a constant nb² offset.

Panel (a): pad_factor=8, full u — sinc leakage from the nb² → 0 step
           puts equal power on k>0 and k<0. FoM ≈ 0.16%.
Panel (b): pad_factor=0, full u — periodic extension has only a tiny
           boundary jump. k<0 is ~0.7% of k>0 power. FoM ≈ 99.91%.

Normalization: each panel by max(pwr[k > 0.1]) to exclude the DC spike.
DC spike (k=0) is off-scale and annotated.
"""

import os
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

# ── Compute FoM and spectra ──────────────────────────────────────────────────
# pad_factor=8: zero-pad 8x → big step nb²→0 → sinc leakage
fom_pad8, k_pad8, Z_pad8 = tmm_h.skk_spectral_fom(
    xx, u, v, pad_factor=8, allowed_side='positive', derivative=False)

# pad_factor=0: periodic extension → tiny boundary jump → FoM ~100%
fom_pad0, k_pad0, Z_pad0 = tmm_h.skk_spectral_fom(
    xx, u, v, pad_factor=0, allowed_side='positive', derivative=False)

print(f"pad_factor=8 : FoM = {fom_pad8:.2f}%")
print(f"pad_factor=0 : FoM = {fom_pad0:.2f}%")


def normalize_spectrum(k, Z, k_thresh=0.1):
    """Normalize by max power at |k| > k_thresh (exclude DC spike)."""
    pwr = np.abs(Z)**2
    mask = np.abs(k) > k_thresh
    return pwr / pwr[mask].max()


pwr_pad8 = normalize_spectrum(k_pad8, Z_pad8)
pwr_pad0 = normalize_spectrum(k_pad0, Z_pad0)

# ── 2-panel figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

configs = [
    (ax1, k_pad8, pwr_pad8, fom_pad8,
     r'(a) pad\_factor=8 — sinc leakage from $n_b^2 \to 0$ step'),
    (ax2, k_pad0, pwr_pad0, fom_pad0,
     r'(b) pad\_factor=0 — periodic extension, tiny boundary jump'),
]

for ax, k, pwr, fom_val, title_prefix in configs:
    ax.fill_between(k[k >= 0], pwr[k >= 0],
                    alpha=0.35, color=GREEN, label=r'$k > 0$ (allowed)')
    ax.fill_between(k[k <= 0], pwr[k <= 0],
                    alpha=0.35, color=RED, label=r'$k < 0$ (forbidden)')
    ax.plot(k, pwr, 'k-', linewidth=0.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-8, 10)
    ax.set_xlim(-50, 50)
    ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$|Z(k)|^2$ (normalized)')
    ax.set_title(f'{title_prefix}\nSpectral FoM = {fom_val:.2f}%')
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
