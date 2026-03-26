"""
fig_forward_backward.py
========================
Shows the sKK asymmetry when the profile is spatially reversed.

Panel (a): Forward — ε(x),  allowed_side='positive', FoM ≈ 99.91%.
           Power concentrated at k > 0 (green).
Panel (b): Reversed — ε(−x), allowed_side='positive', FoM ≈ 0%.
           Power concentrated at k < 0 (red/forbidden) — profile is reflective.

Physics:
  ε(x) = nb² − A·γ/(x + iγ) has a pole at x = −iγ (lower half-plane).
  By Paley–Wiener its FT is supported on k ≥ 0.

  Flipping x → −x gives a pole at x = +iγ (upper half-plane) and moves
  FT support to k ≤ 0. For the centered profile u_c = Re(ε) − nb² (which
  is odd), this is equivalent to u_c → −u_c while v is unchanged (even).

FoM computed with pad_factor=0 (validates the k=0 exclusion fix).
Display spectrum computed from centered profiles with pad_factor=8 for
visual smoothness.
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
u    = np.real(ee)
v    = np.imag(ee)

# ── FoM: pad_factor=0 validates the k=0 exclusion fix ───────────────────────
# Forward: u + iv, pole at -iγ → FT support k>0
fom_fwd, _, _ = tmm_h.skk_spectral_fom(
    xx, u, v, pad_factor=0, allowed_side='positive', derivative=False)

# Backward (reversed profile): same allowed_side → FoM ≈ 0%
fom_bwd, _, _ = tmm_h.skk_spectral_fom(
    xx, u[::-1], v[::-1], pad_factor=0, allowed_side='positive', derivative=False)

print(f"Forward  (allowed_side='positive') FoM = {fom_fwd:.2f}%")
print(f"Reversed (allowed_side='positive') FoM = {fom_bwd:.2f}%")

# ── Display spectrum: centered profiles + pad_factor=8 for smoothness ────────
# u_c = u − nb²   is odd → FT supported on k>0
# −u_c = −u + nb² is odd → FT supported on k<0
u_c   =  u - nb**2    # forward centered
u_c_b = -u + nb**2    # backward centered = -u_c

dx_val = xx[1] - xx[0]
pad_n  = 8 * len(xx)


def compute_spectrum(u_centered, v_signal):
    z     = u_centered + 1j * v_signal
    z_pad = np.pad(z, (pad_n, pad_n), mode='constant')
    Z     = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_pad)))
    k     = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z_pad), d=dx_val))
    pwr   = np.abs(Z)**2
    pwr  /= pwr.max()
    return k, pwr


k_fwd, pwr_fwd = compute_spectrum(u_c,   v)
k_bwd, pwr_bwd = compute_spectrum(u_c_b, v)

# ── 2-panel figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

configs = [
    (ax1, k_fwd, pwr_fwd, fom_fwd,
     r'(a) Forward $\varepsilon(x)$'),
    (ax2, k_bwd, pwr_bwd, fom_bwd,
     r'(b) Reversed $\varepsilon(-x)$'),
]

for ax, k, pwr, fom_val, title_prefix in configs:
    ax.fill_between(k[k >= 0], pwr[k >= 0],
                    alpha=0.35, color=GREEN, label=r'$k > 0$ (allowed)')
    ax.fill_between(k[k <= 0], pwr[k <= 0],
                    alpha=0.35, color=RED,   label=r'$k < 0$ (forbidden)')
    ax.plot(k, pwr, 'k-', linewidth=0.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-8, 10)
    ax.set_xlim(-50, 50)
    ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$|Z(k)|^2$ (normalized)')
    ax.set_title(f'{title_prefix}\nSpectral FoM = {fom_val:.2f}%')
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_forward_backward.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
