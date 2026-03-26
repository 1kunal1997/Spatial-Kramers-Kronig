"""
Verify spectral_fom fix: Lorentzian forward vs backward.

After the DC-bias fix (k=0 excluded from both sides), the Lorentzian
eps(x) = nb^2 - a*gam/(x + i*gam) returns ~100% FoM on the correct
allowed side and ~0% on the wrong side.

Physics:
  eps_centered(x) = -a*gam/(x+i*gam) has a pole at x=-i*gam (lower
  half-plane), so it is analytic in the upper half-plane.  By
  Paley-Wiener, its Fourier transform is supported on k >= 0.
  => allowed_side='positive' (forward, +x propagation) => FoM ~100%
  => allowed_side='negative' (backward, -x propagation) => FoM ~0%

Two ways the DC spike bug manifested with the old ~mask code:
  - With pad_factor=0 (periodic extension): Z[k=0] ~ nb^2 * N ~ 90000,
    which landed in E_forbidden via `~mask`, driving FoM to ~0%.
  - With pad_factor>0 (zero-padding): the constant nb^2 creates a large
    step-to-zero at the boundaries, generating sinc leakage on all k.

The fix (exclude k=0 via strict inequality) resolves the pad_factor=0
case cleanly, giving ~100% FoM.  We use pad_factor=0 here so the test
directly validates the fix.

For the plot, we subtract nb^2 from u so the Lorentzian spectral
structure is visible (at k!=0 this is identical to what skk_spectral_fom
sees, since the DC only lives at k=0 which is excluded from both sides).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tmm_helper as tmm_h

# ── Plot style (identical to skk_analysis_consolidated.py) ─────────────────
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

# ── Lorentzian parameters (same grid as generate_n_and_d_v5_avg_over_cell) ─
a, gam, nb = 1.0, 0.1, 1.5

dx   = gam / 100
xmin = -gam * 200
xmax =  gam * 200
xx   = np.linspace(xmin, xmax, 1 + int(np.floor((xmax - xmin) / dx)))
ee   = tmm_h.eps(xx, a, gam, nb)
u    = np.real(ee)
v    = np.imag(ee)

# ── FoM: use pad_factor=0 (periodic extension avoids step-to-zero artifact)
# k=0 is excluded from both sides by the fixed mask logic.
fom_fwd, k_fwd, Z_fwd = tmm_h.skk_spectral_fom(
    xx, u, v, pad_factor=0, allowed_side='positive', derivative=False)
fom_bwd, k_bwd, Z_bwd = tmm_h.skk_spectral_fom(
    xx, u, v, pad_factor=0, allowed_side='negative', derivative=False)

print(f"Forward  (allowed_side='positive') FoM = {fom_fwd:.2f}%")
print(f"Backward (allowed_side='negative') FoM = {fom_bwd:.2f}%")

# ── Plot spectrum: subtract nb^2 DC so the Lorentzian structure is visible.
# At k != 0 this equals exactly what skk_spectral_fom uses (DC at k=0 is
# excluded from both energy sums by the fix anyway).
u_c  = u - nb**2                    # = Re(eps) - nb^2 = -a*gam*x/(x^2+gam^2)
z_plt = u_c + 1j * v               # = -a*gam / (x + i*gam)

# Use pad_factor=8 here only for a smoother-looking spectrum in the plot.
pad_n  = 8 * len(xx)
z_pad  = np.pad(z_plt, (pad_n, pad_n), mode='constant')
dx_val = xx[1] - xx[0]
Z_plt  = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_pad)))
k_plt  = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z_pad), d=dx_val))
pwr    = np.abs(Z_plt)**2
pwr   /= pwr.max()              # max is now at small positive k (Lorentzian peak)

# ── 2-panel figure (format identical to fig_new5_fom_explanation.png) ───────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

configs = [
    (ax1, fom_fwd, 'positive', r'(a) Forward — allowed $k>0$'),
    (ax2, fom_bwd, 'negative', r'(b) Backward — allowed $k<0$'),
]

for ax, fom_val, allowed, title_prefix in configs:
    if allowed == 'positive':
        ax.fill_between(k_plt[k_plt >= 0], pwr[k_plt >= 0],
                        alpha=0.35, color=GREEN,
                        label=r'Positive $k$ (allowed)')
        ax.fill_between(k_plt[k_plt <= 0], pwr[k_plt <= 0],
                        alpha=0.35, color=RED,
                        label=r'Negative $k$ (forbidden)')
    else:
        ax.fill_between(k_plt[k_plt <= 0], pwr[k_plt <= 0],
                        alpha=0.35, color=GREEN,
                        label=r'Negative $k$ (allowed)')
        ax.fill_between(k_plt[k_plt >= 0], pwr[k_plt >= 0],
                        alpha=0.35, color=RED,
                        label=r'Positive $k$ (forbidden)')

    ax.plot(k_plt, pwr, 'k-', linewidth=0.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-10, 2)
    ax.set_xlim(-50, 50)        # Lorentzian decays on scale 1/gam = 10 µm⁻¹
    ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$|Z(k)|^2$ (arb. units)')
    ax.set_title(f'{title_prefix} — Spectral FoM = {fom_val:.1f}%')
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_lorentzian_spectral_fom.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
