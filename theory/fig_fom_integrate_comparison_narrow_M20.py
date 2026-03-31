"""
fig_fom_integrate_comparison_narrow_M20.py
==========================================
Same as fig_fom_integrate_comparison_narrow.py (gam=0.01 um, klim=400),
but with domain_factor M=20 for the Lorentzian instead of 200.

This matches the Lorentzian domain (±20*gam = ±0.20 um) to the logistic
domain (±20/k_steep = ±0.20 um), giving both the same domain-to-width ratio
of 20. The DC spike should be reduced since the nb² pedestal integral is 10x
smaller, but cannot fully vanish since the Lorentzian sits on nb² everywhere.

  Top left:     Lorentzian (gam=0.01, M=20) — direct FT of ε(x)
  Top right:    Lorentzian (gam=0.01, M=20) — derivative → FT → ÷ik
  Bottom left:  Logistic — direct FT of ε(x)
  Bottom right: Logistic — derivative → FT → ÷ik
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


def compute_fom_from_spectrum(k, pwr):
    E_pos = np.sum(pwr[k > 0])
    E_neg = np.sum(pwr[k < 0])
    return 100 * max(0.0, (E_pos - E_neg) / (E_pos + E_neg))


def direct_ft(x, u, v):
    """Plain FT of ε(x) = u + iv → |ε̂(k)|²."""
    x = np.asarray(x, float)
    z = np.asarray(u, float) + 1j * np.asarray(v, float)
    dx = x[1] - x[0]
    Z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=dx))
    pwr = np.abs(Z)**2
    fom = compute_fom_from_spectrum(k, pwr)
    return fom, k, pwr


def deriv_ft_integrate(x, u, v):
    """Derivative → FT → ÷ik to recover ε̂(k) → |ε̂(k)|²."""
    x = np.asarray(x, float)
    ud = np.gradient(np.asarray(u, float), x)
    vd = np.gradient(np.asarray(v, float), x)
    z_d = ud + 1j * vd
    dx = x[1] - x[0]
    Z_d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_d)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z_d), d=dx))
    eps_hat = np.zeros_like(Z_d)
    nonzero = k != 0
    eps_hat[nonzero] = Z_d[nonzero] / (1j * k[nonzero])
    eps_hat[~nonzero] = 0
    pwr = np.abs(eps_hat)**2
    fom = compute_fom_from_spectrum(k, pwr)
    return fom, k, pwr


# === Lorentzian profile — M=20 domain (±20*gam = ±0.20 um) ===
M = 20
a, gam, nb_l = 1.0, 0.01, 1.5
dx_l = gam / 100
xx_l = np.linspace(-gam * M, gam * M, 1 + int(np.floor(2 * gam * M / dx_l)))
ee_l = tmm_h.eps(xx_l, a, gam, nb_l)
u_l = np.real(ee_l)
v_l = np.imag(ee_l)

# === Logistic profile (same as before — domain ±0.20 um) ===
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

# === Compute all 4 cases ===
fom_a, k_a, pwr_a = direct_ft(xx_l, u_l, v_l)
fom_b, k_b, pwr_b = deriv_ft_integrate(xx_l, u_l, v_l)
fom_c, k_c, pwr_c = direct_ft(xx_g, u_g, v_g)
fom_d, k_d, pwr_d = deriv_ft_integrate(xx_g, u_g, v_g)

print(f"(a) Lorentzian (M=20), direct FT:          FoM = {fom_a:.2f}%")
print(f"(b) Lorentzian (M=20), deriv-FT-integrate: FoM = {fom_b:.2f}%")
print(f"(c) Logistic,          direct FT:          FoM = {fom_c:.2f}%")
print(f"(d) Logistic,          deriv-FT-integrate: FoM = {fom_d:.2f}%")

KLIM = 400

# === 2x2 Plot ===
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

configs = [
    (axes[0, 0], k_a, pwr_a, fom_a,
     r'(a) Lorentzian ($\gamma=0.01\ \mu$m, $M=20$) — direct FT of $\varepsilon(x)$'),
    (axes[0, 1], k_b, pwr_b, fom_b,
     r'(b) Lorentzian ($\gamma=0.01\ \mu$m, $M=20$) — $d\varepsilon/dx \to$ FT $\to \div\, ik$'),
    (axes[1, 0], k_c, pwr_c, fom_c,
     r'(c) Logistic — direct FT of $\varepsilon(x)$'),
    (axes[1, 1], k_d, pwr_d, fom_d,
     r'(d) Logistic — $d\varepsilon/dx \to$ FT $\to \div\, ik$'),
]

for ax, k, pwr, fom, title in configs:
    mask_norm = np.abs(k) > 1.0
    pmax = pwr[mask_norm].max() if pwr[mask_norm].any() else 1.0
    pwr_n = pwr / pmax

    ax.fill_between(k[k >= 0], pwr_n[k >= 0], alpha=0.35, color=GREEN,
                    label=r'$k > 0$ (allowed)')
    ax.fill_between(k[k <= 0], pwr_n[k <= 0], alpha=0.35, color=RED,
                    label=r'$k < 0$ (forbidden)')
    ax.plot(k, pwr_n, 'k-', lw=0.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_ylim(1e-8, 1e4)
    ax.set_xlim(-KLIM, KLIM)
    ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$ (normalized)')
    ax.set_title(f'{title}\nFoM = {fom:.2f}%', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

# Column labels
fig.text(0.28, 0.98, r'Direct FT of $\varepsilon(x)$',
         fontsize=14, ha='center', va='top', fontweight='bold')
fig.text(0.74, 0.98, r'Derivative $\to$ FT $\to$ integrate',
         fontsize=14, ha='center', va='top', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_fom_integrate_comparison_narrow_M20.png')
plt.savefig(outpath)
plt.close()
print(f"\nSaved: {outpath}")
