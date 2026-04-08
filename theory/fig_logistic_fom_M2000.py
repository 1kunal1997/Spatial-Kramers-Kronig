"""
fig_logistic_fom_M2000.py
=========================
Left: logistic dielectric profile ε'(x) and ε''(x), zoomed to ±0.05 μm.
Right: spectral FoM plot — |ε̂(k)|² vs k with green/red fill showing
       one-sidedness. Uses M=2000 (domain ±20 μm) for fine dk.
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

GREEN  = '#2ca02c'
RED    = '#d62728'
BLUE_P = '#1f77b4'
DKRED  = '#8B0000'


def compute_fom_from_spectrum(k, pwr):
    E_pos = np.sum(pwr[k > 0])
    E_neg = np.sum(pwr[k < 0])
    return 100 * max(0.0, (E_pos - E_neg) / (E_pos + E_neg))


def deriv_ft_integrate(x, u, v):
    """Derivative -> FT -> div ik to recover eps_hat(k) -> |eps_hat(k)|^2."""
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


# === Logistic profile with M=2000 ===
k_steep = 100
nb_g = 1.7
M = 2000
dx_g = 1 / (100 * k_steep)
xmin_g = -M / k_steep
xmax_g = -xmin_g
xx_g = np.linspace(xmin_g, xmax_g, 1 + int(np.floor((xmax_g - xmin_g) / dx_g)))
u_g = (nb_g**2 - 1) / (1 + np.exp(np.clip(k_steep * xx_g, -500, 500))) + 1

N_g = len(xx_g)
ud_g = np.gradient(u_g, xx_g)
vd_g = np.imag(hilbert(ud_g))
v_g = cumulative_trapezoid(vd_g, xx_g, initial=0)
v_g -= np.linspace(v_g[0], v_g[-1], N_g)

fom, k, pwr = deriv_ft_integrate(xx_g, u_g, v_g)
print(f"Logistic (M={M}), deriv-FT-integrate: FoM = {fom:.2f}%")

KLIM = 400

# === Figure: 1×2 (profile left, FoM right) ===
fig, (ax_prof, ax_fom) = plt.subplots(1, 2, figsize=(15, 7))

# --- Left: profile ---
zoom = (xx_g >= -0.05) & (xx_g <= 0.05)
ax_prof2 = ax_prof.twinx()
ax_prof.plot(xx_g[zoom], u_g[zoom], color=BLUE_P, lw=2.0)
ax_prof2.plot(xx_g[zoom], v_g[zoom], color=DKRED, lw=2.0)
ax_prof.set_xlabel(r'$x$ ($\mu$m)')
ax_prof.set_ylabel(r"$\varepsilon'(x)$", color=BLUE_P)
ax_prof2.set_ylabel(r"$\varepsilon''(x)$", color=DKRED)
ax_prof.tick_params(axis='y', labelcolor=BLUE_P)
ax_prof2.tick_params(axis='y', labelcolor=DKRED)
ax_prof.set_title(r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$)')
ax_prof.set_xlim(-0.05, 0.05)

# --- Right: FoM spectrum ---
ax_fom.fill_between(k[k >= 0], pwr[k >= 0], alpha=0.35, color=GREEN,
                    label=r'$k > 0$ (allowed)')
ax_fom.fill_between(k[k <= 0], pwr[k <= 0], alpha=0.35, color=RED,
                    label=r'$k < 0$ (forbidden)')
ax_fom.plot(k, pwr, 'k-', lw=0.5, alpha=0.5)
ax_fom.set_yscale('log')
ax_fom.set_xlim(-KLIM, KLIM)
ax_fom.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
ax_fom.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$')
ax_fom.set_title(r'Logistic ($M=2000$) — $d\varepsilon/dx \to$ FT $\to \div\, ik$'
                 + f'\nFoM = {fom:.2f}%', fontsize=12)
ax_fom.legend(loc='upper right', fontsize=10)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_logistic_fom_M2000_unnorm.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
