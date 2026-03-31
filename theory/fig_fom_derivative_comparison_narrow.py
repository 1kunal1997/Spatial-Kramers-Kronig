"""
fig_fom_derivative_comparison_narrow.py
========================================
Same as fig_fom_derivative_comparison.py, but with the Lorentzian width
(gam = 0.01 um) matched to the logistic transition width (1/k_steep = 0.01 um).
Both profiles now vary on the same spatial scale, so both rows share the
same k-axis range (±400 um⁻¹).

2×2 comparison:
  (a) Lorentzian, no derivative  → FoM high  (endpoints match: ε'→nb² at both ends)
  (b) Lorentzian, with derivative → FoM high  (derivative also works)
  (c) Logistic, no derivative → FoM LOW  (endpoints DIFFER: nb²≠1 → spectral leakage)
  (d) Logistic, with derivative → FoM high (dε'/dx→0 at both ends → no leakage)
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tmm_helper as tmm_h
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 9, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

GREEN = '#2ca02c'
RED = '#d62728'

# === Lorentzian profile — narrow (gam = 0.01 um, matching logistic width) ===
a, gam, nb_l = 1.0, 0.01, 1.5
dx_l = gam / 100
xx_l = np.linspace(-gam * 200, gam * 200, 1 + int(np.floor(2 * gam * 200 / dx_l)))
ee_l = tmm_h.eps(xx_l, a, gam, nb_l)
u_l = np.real(ee_l)
v_l = np.imag(ee_l)

# === Logistic profile (same as before: k_steep = 100, width = 1/k_steep = 0.01 um) ===
k_steep = 100; nb_g = 1.7
dx_g = 1 / (100 * k_steep)
xmin_g = -20 / k_steep; xmax_g = -xmin_g
xx_g = np.linspace(xmin_g, xmax_g, 1 + int(np.floor((xmax_g - xmin_g) / dx_g)))
u_g = (nb_g**2 - 1) / (1 + np.exp(k_steep * xx_g)) + 1

# Compute ε'' via derivative-then-integrate HT (no padding)
N_g = len(xx_g)
ud_g = np.gradient(u_g, xx_g)
vd_g = np.imag(hilbert(ud_g))
v_g = cumulative_trapezoid(vd_g, xx_g, initial=0)
v_g -= np.linspace(v_g[0], v_g[-1], N_g)

# === Compute all 4 FoMs ===
fom_a, k_a, Z_a = tmm_h.skk_spectral_fom(xx_l, u_l, v_l, derivative=False)
fom_b, k_b, Z_b = tmm_h.skk_spectral_fom(xx_l, u_l, v_l, derivative=True)
fom_c, k_c, Z_c = tmm_h.skk_spectral_fom(xx_g, u_g, v_g, derivative=False)
fom_d, k_d, Z_d = tmm_h.skk_spectral_fom(xx_g, u_g, v_g, derivative=True)

print(f"(a) Lorentzian (gam=0.01), no deriv:   FoM = {fom_a:.2f}%")
print(f"(b) Lorentzian (gam=0.01), with deriv: FoM = {fom_b:.2f}%")
print(f"(c) Logistic,              no deriv:   FoM = {fom_c:.2f}%")
print(f"(d) Logistic,              with deriv: FoM = {fom_d:.2f}%")

# Both rows use the same k-axis now that widths are matched
KLIM = 400

# === 2×2 Plot ===
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

configs = [
    (axes[0, 0], k_a, Z_a, fom_a,
     r'(a) Lorentzian ($\gamma=0.01\ \mu$m) — direct FT of $\varepsilon(x)$'
     + f'\nFoM = {fom_a:.2f}%'),
    (axes[0, 1], k_b, Z_b, fom_b,
     r'(b) Lorentzian ($\gamma=0.01\ \mu$m) — FT of $d\varepsilon/dx$'
     + f'\nFoM = {fom_b:.2f}%'),
    (axes[1, 0], k_c, Z_c, fom_c,
     r'(c) Logistic — direct FT of $\varepsilon(x)$'
     + f'\nFoM = {fom_c:.2f}%'),
    (axes[1, 1], k_d, Z_d, fom_d,
     r'(d) Logistic — FT of $d\varepsilon/dx$'
     + f'\nFoM = {fom_d:.2f}%'),
]

for ax, k, Z, fom, title in configs:
    pwr = np.abs(Z)**2
    mask = np.abs(k) > 1.0
    pmax = pwr[mask].max() if pwr[mask].max() > 0 else 1.0
    pwr = pwr / pmax

    ax.fill_between(k[k >= 0], pwr[k >= 0], alpha=0.35, color=GREEN,
                    label=r'$k > 0$ (allowed)')
    ax.fill_between(k[k <= 0], pwr[k <= 0], alpha=0.35, color=RED,
                    label=r'$k < 0$ (forbidden)')
    ax.plot(k, pwr, 'k-', lw=0.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-8, 10)
    ax.set_xlim(-KLIM, KLIM)
    ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$ (normalized)')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

# Row labels
fig.text(0.02, 0.75,
         r'Lorentzian ($\gamma=0.01\ \mu$m)' + '\n'
         + r'$\varepsilon(\pm\infty) = n_b^2$' + '\n(same endpoints)',
         fontsize=11, ha='center', va='center', rotation=90,
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round'))
fig.text(0.02, 0.28,
         'Logistic\n' + r'$\varepsilon(-\infty) = n_b^2 \neq 1 = \varepsilon(+\infty)$'
         + '\n(different endpoints)',
         fontsize=11, ha='center', va='center', rotation=90,
         bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round'))

plt.tight_layout(rect=[0.06, 0, 1, 1])

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_fom_derivative_comparison_narrow.png')
plt.savefig(outpath)
plt.close()
print(f"\nSaved: {outpath}")
