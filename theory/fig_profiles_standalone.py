"""
fig_profiles_standalone.py
==========================
Four standalone dielectric profile figures (eps' and eps'', dual y-axis).
No zooming — each figure shows the full computation domain.

1. fig_profile_lorentzian_M200.png
   Lorentzian used in fig_fom_integrate_comparison_narrow:
   a=1, gam=0.01 um, nb=1.5, M=200 (domain +-2 um)

2. fig_profile_logistic_M20.png
   Logistic used in fig_fom_integrate_comparison_narrow:
   k_steep=100, nb=1.7, M=20 (domain +-0.20 um)

3. fig_profile_logistic_M2000.png
   Logistic used in fig_logistic_fom_M2000:
   k_steep=100, nb=1.7, M=2000 (domain +-20 um)

4. fig_profile_lorentzian_M2000.png
   Lorentzian used in fig_lorentzian_fom_M2000:
   a=1, gam=0.01 um, nb=1.5, M=2000 (domain +-20 um)
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
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

BLUE  = '#1f77b4'
DKRED = '#8B0000'

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def save_profile(xx, u, v, title, outname):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    ax.plot(xx, u, color=BLUE,  lw=2.0)
    ax2.plot(xx, v, color=DKRED, lw=2.0)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\varepsilon'(x)$",  color=BLUE)
    ax2.set_ylabel(r"$\varepsilon''(x)$", color=DKRED)
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=DKRED)
    ax.set_title(title)
    ax.set_xlim(xx[0], xx[-1])
    plt.tight_layout()
    path = os.path.join(FIGDIR, outname)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── 1. Lorentzian M=200 (from fig_fom_integrate_comparison_narrow) ────────────
a, gam, nb_l = 1.0, 0.01, 1.5
dx_l = gam / 100
xx_l200 = np.linspace(-gam * 200, gam * 200,
                       1 + int(np.floor(2 * gam * 200 / dx_l)))
ee_l200 = tmm_h.eps(xx_l200, a, gam, nb_l)
save_profile(xx_l200, np.real(ee_l200), np.imag(ee_l200),
             r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $M=200$, domain $\pm 2\ \mu$m)',
             'fig_profile_lorentzian_M200.png')

# ── 2. Logistic M=20 (from fig_fom_integrate_comparison_narrow) ───────────────
k_steep, nb_g = 100, 1.7
dx_g = 1 / (100 * k_steep)
xx_g20 = np.linspace(-20 / k_steep, 20 / k_steep,
                      1 + int(np.floor(40 / k_steep / dx_g)))
u_g20 = (nb_g**2 - 1) / (1 + np.exp(k_steep * xx_g20)) + 1
N20 = len(xx_g20)
vd20 = np.imag(hilbert(np.gradient(u_g20, xx_g20)))
v_g20 = cumulative_trapezoid(vd20, xx_g20, initial=0)
v_g20 -= np.linspace(v_g20[0], v_g20[-1], N20)
save_profile(xx_g20, u_g20, v_g20,
             r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$, $M=20$, domain $\pm 0.20\ \mu$m)',
             'fig_profile_logistic_M20.png')

# ── 3. Logistic M=2000 (from fig_logistic_fom_M2000) ─────────────────────────
M = 2000
xx_g2000 = np.linspace(-M / k_steep, M / k_steep,
                        1 + int(np.floor(2 * M / k_steep / dx_g)))
u_g2000 = (nb_g**2 - 1) / (1 + np.exp(np.clip(k_steep * xx_g2000, -500, 500))) + 1
N2000 = len(xx_g2000)
vd2000 = np.imag(hilbert(np.gradient(u_g2000, xx_g2000)))
v_g2000 = cumulative_trapezoid(vd2000, xx_g2000, initial=0)
v_g2000 -= np.linspace(v_g2000[0], v_g2000[-1], N2000)
save_profile(xx_g2000, u_g2000, v_g2000,
             r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$, $M=2000$, domain $\pm 20\ \mu$m)',
             'fig_profile_logistic_M2000.png')

# ── 4. Lorentzian M=2000 (from fig_lorentzian_fom_M2000) ─────────────────────
xx_l2000 = np.linspace(-M * gam, M * gam,
                        1 + int(np.floor(2 * M * gam / dx_l)))
ee_l2000 = tmm_h.eps(xx_l2000, a, gam, nb_l)
save_profile(xx_l2000, np.real(ee_l2000), np.imag(ee_l2000),
             r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $M=2000$, domain $\pm 20\ \mu$m)',
             'fig_profile_lorentzian_M2000.png')
