"""
fig_logistic_sinh_crossover.py
==============================
Left: logistic dielectric profile ε'(x) and ε''(x), zoomed to ±0.05 μm.
Right: power spectrum |ε̂(k)|² with analytic C/sinh²(πk/k_s) overlay and
       both asymptotes (1/k² for small k, exp(−2πk/k_s) for large k).
       Crossover at k ≈ k_s/π ≈ 32 μm⁻¹.

Uses M=2000 (domain ±20 μm) for fine dk ≈ 0.157 μm⁻¹.
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


# === Logistic profile with large domain (M=2000) for fine dk ===
k_steep = 100
nb_g = 1.7
M = 2000
dx_g = 1 / (100 * k_steep)
xmin_g = -M / k_steep
xmax_g = -xmin_g
xx_g = np.linspace(xmin_g, xmax_g, 1 + int(np.floor((xmax_g - xmin_g) / dx_g)))
u_g = (nb_g**2 - 1) / (1 + np.exp(np.clip(k_steep * xx_g, -500, 500))) + 1

# Compute eps'' via derivative-then-integrate HT
N_g = len(xx_g)
ud_g = np.gradient(u_g, xx_g)
vd_g = np.imag(hilbert(ud_g))
v_g = cumulative_trapezoid(vd_g, xx_g, initial=0)
v_g -= np.linspace(v_g[0], v_g[-1], N_g)

# Numerical spectrum
k_num, pwr_num = deriv_ft_integrate(xx_g, u_g, v_g)

# Only k > 0
mask = k_num > 0.5
k_pos = k_num[mask]
pwr_pos = pwr_num[mask]

# Analytic curves
k_an = np.linspace(0.5, 250, 2000)
k_cross = k_steep / np.pi  # ~32 um^-1

# Exact analytic prefactor from residue theorem (no fitting):
# z = eps' + i*eps'', eps'' = HT[eps'] => z_hat(k) = 2*eps_hat'(k) for k>0
# eps_hat'(k) = pi*i*Delta / (k_s * sinh(pi*k/k_s))  from pole sum
# => |z_hat(k)|^2 = 4*pi^2*Delta^2 / (k_s^2 * sinh^2(pi*k/k_s))
Delta = nb_g**2 - 1
C_exact = 4 * np.pi**2 * Delta**2 / k_steep**2
pwr_sinh = C_exact / np.sinh(np.pi * k_an / k_steep)**2

# Asymptotes derived from C_exact (no fitting):
# small k: sinh(pi*k/ks) ~ pi*k/ks  =>  |z_hat|^2 ~ 4*Delta^2/k^2
# large k: sinh ~ exp(pi*k/ks)/2    =>  |z_hat|^2 ~ 4*C_exact*exp(-2*pi*k/ks)
pwr_1overk2 = 4 * Delta**2 / k_an**2
pwr_exp     = 4 * C_exact * np.exp(-2 * np.pi * k_an / k_steep)

# === Figure: 1×2 (profile left, spectrum right) ===
fig, (ax_prof, ax_spec) = plt.subplots(1, 2, figsize=(15, 7))

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

# --- Right: spectrum ---
ax_spec.semilogy(k_pos, pwr_pos, '-', color='#2ca02c', lw=2.0,
                 label=f'Numerical (M={M})')
ax_spec.semilogy(k_an, pwr_sinh, '--', color='black', lw=1.5,
                 label=r'$C/\sinh^2(\pi k / k_s)$')
ax_spec.semilogy(k_an, pwr_1overk2, ':', color='#d62728', lw=2.0,
                 label=r'$1/k^2$ (small $k$)')
ax_spec.semilogy(k_an, pwr_exp, ':', color='#1f77b4', lw=2.0,
                 label=r'$\exp(-2\pi k/k_s)$ (large $k$)')
ax_spec.axvline(k_cross, color='gray', ls='--', lw=1.0, alpha=0.7)
ax_spec.text(k_cross + 3, 1e-7, f'$k = k_s/\\pi \\approx {k_cross:.0f}$',
             fontsize=11, color='gray')
ax_spec.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
ax_spec.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$')
ax_spec.set_title(r'Logistic spectrum: $1/k^2 \to \exp(-2\pi k/k_s)$ crossover at $k \approx k_s/\pi$')
ax_spec.set_xlim(0.5, 250)
ax_spec.set_ylim(1e-9, 1e3)
ax_spec.legend(loc='upper right', fontsize=11)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_logistic_sinh_crossover_unnorm.png')
plt.savefig(outpath)
plt.close()
print(f"Saved: {outpath}")
