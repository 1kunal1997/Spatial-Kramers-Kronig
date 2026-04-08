"""
fig_lorentzian_fom_M2000.py
============================
Left: Lorentzian dielectric profile ε'(x) and ε''(x), zoomed to ±0.05 μm.
Right: spectral FoM — |ε̂(k)|² vs k with green/red fill.
       The Lorentzian is analytic in the upper half x-plane (pole only at
       x = −iγ), so all spectral weight is at k > 0 and FoM → 100%.

Uses gam = 0.01 μm, M = 2000 (domain ±20 μm, dk ≈ 0.157 μm⁻¹).
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


# === Lorentzian profile: gam=0.01 um, M=2000 ===
a, gam, nb = 1.0, 0.01, 1.5
M = 2000
dx_l = gam / 100
xmin_l = -M * gam
xmax_l =  M * gam
xx_l = np.linspace(xmin_l, xmax_l, 1 + int(np.floor((xmax_l - xmin_l) / dx_l)))
ee_l = tmm_h.eps(xx_l, a, gam, nb)
u_l = np.real(ee_l)
v_l = np.imag(ee_l)

fom, k, pwr = deriv_ft_integrate(xx_l, u_l, v_l)
print(f"Lorentzian (gam={gam}, M={M}), deriv-FT-integrate: FoM = {fom:.2f}%")

KLIM = 400

# === Figure: 1×2 (profile left, FoM right) ===
fig, (ax_prof, ax_fom) = plt.subplots(1, 2, figsize=(15, 7))

# --- Left: profile (zoomed to ±0.05 um = ±5*gam) ---
zoom = (xx_l >= -0.05) & (xx_l <= 0.05)
ax_prof2 = ax_prof.twinx()
ax_prof.plot(xx_l[zoom], u_l[zoom], color=BLUE_P, lw=2.0)
ax_prof2.plot(xx_l[zoom], v_l[zoom], color=DKRED, lw=2.0)
ax_prof.set_xlabel(r'$x$ ($\mu$m)')
ax_prof.set_ylabel(r"$\varepsilon'(x)$", color=BLUE_P)
ax_prof2.set_ylabel(r"$\varepsilon''(x)$", color=DKRED)
ax_prof.tick_params(axis='y', labelcolor=BLUE_P)
ax_prof2.tick_params(axis='y', labelcolor=DKRED)
ax_prof.set_title(r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $a=1$)')
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
ax_fom.set_title(r'Lorentzian ($\gamma=0.01\ \mu$m, $M=2000$) — $d\varepsilon/dx \to$ FT $\to \div\, ik$'
                 + f'\nFoM = {fom:.2f}%', fontsize=12)
ax_fom.legend(loc='upper right', fontsize=10)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_lorentzian_fom_M2000_unnorm.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
