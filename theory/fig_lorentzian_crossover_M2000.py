"""
fig_lorentzian_crossover_M2000.py
==================================
Left: Lorentzian dielectric profile ε'(x) and ε''(x), zoomed to ±0.05 μm.
      ε' has a dispersive (antisymmetric) shape; ε'' is a Lorentzian bell.
Right: power spectrum |ε̂(k)|² vs k (k > 0 only) with analytic C·exp(−2γk)
       overlay. The Lorentzian has a single pole at x = −iγ in the lower
       half-plane, giving a pure exponential with no crossover — in contrast
       to the logistic (1/k² → exp crossover at k ≈ k_s/π).

Uses gam = 0.01 μm (matching logistic transition width 1/k_steep = 0.01 μm)
and M = 2000 (domain ±20 μm, dk ≈ 0.157 μm⁻¹).
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

BLUE_P = '#1f77b4'
DKRED  = '#8B0000'


def deriv_ft_integrate(x, u, v):
    """Derivative -> FT -> div ik to recover eps_hat(k) -> |eps_hat(k)|^2."""
    x = np.asarray(x, float)
    ud = np.gradient(np.asarray(u, float), x)
    vd = np.gradient(np.asarray(v, float), x)
    z_d = ud + 1j * vd
    dx = x[1] - x[0]
    Z_d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_d))) * dx
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z_d), d=dx))
    eps_hat = np.zeros_like(Z_d)
    nonzero = k != 0
    eps_hat[nonzero] = Z_d[nonzero] / (1j * k[nonzero])
    eps_hat[~nonzero] = 0
    pwr = np.abs(eps_hat)**2
    return k, pwr


# === Lorentzian profile: gam=0.01 um, M=2000 ===
a, gam, nb = 1.0, 0.01, 1.5
M = 2000
dx_l = gam / 100           # 0.0001 um (100 pts per half-width)
xmin_l = -M * gam          # -20 um
xmax_l =  M * gam          # +20 um
xx_l = np.linspace(xmin_l, xmax_l, 1 + int(np.floor((xmax_l - xmin_l) / dx_l)))
ee_l = tmm_h.eps(xx_l, a, gam, nb)   # eps(x) = nb^2 - a*gam/(x + i*gam)
u_l = np.real(ee_l)
v_l = np.imag(ee_l)

print(f"Grid: N = {len(xx_l):,} points, dx = {dx_l:.4f} um, "
      f"dk = {2*np.pi/(xmax_l-xmin_l):.4f} um^-1")

# Numerical spectrum (derivative method removes nb^2 DC spike)
k_num, pwr_num = deriv_ft_integrate(xx_l, u_l, v_l)

# Only k > 0
mask = k_num > 0.5
k_pos = k_num[mask]
pwr_pos = pwr_num[mask]

# Exact analytic: |eps_hat(k)|^2 = C * exp(-2*gam*k) for k > 0
# C = 4*pi^2*a^2*gam^2  from residue at pole x = -i*gam (no fitting)
k_an = np.linspace(0.5, 250, 2000)
C_exact = 4 * np.pi**2 * a**2 * gam**2
pwr_exp_an = C_exact * np.exp(-2 * gam * k_an)

# === Figure: 1×2 (profile left, spectrum right) ===
fig, (ax_prof, ax_spec) = plt.subplots(1, 2, figsize=(15, 7))

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

# --- Right: spectrum ---
ax_spec.semilogy(k_pos, pwr_pos, '-', color='#2ca02c', lw=2.0,
                 label=f'Numerical (M={M})')
ax_spec.semilogy(k_an, pwr_exp_an, '--', color='black', lw=1.8,
                 label=r'$4\pi^2 a^2\gamma^2\,\exp(-2\gamma k)$  [exact]')
ax_spec.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
ax_spec.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$')
ax_spec.set_title(r'Lorentzian spectrum: pure $\exp(-2\gamma k)$, no crossover'
                  '\n' + r'(pole at $x=-i\gamma$ in lower half-plane)')
ax_spec.set_xlim(0.5, 250)
ax_spec.set_ylim(1e-6, 1e-2)
ax_spec.legend(loc='upper right', fontsize=11)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_lorentzian_crossover_M2000_unnorm.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
