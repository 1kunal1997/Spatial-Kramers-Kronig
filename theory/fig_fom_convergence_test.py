"""
fig_fom_convergence_test.py
===========================
Convergence test: if the 44% FoM (after ÷ik) is a numerical artifact,
both the derivative FoM and integrated FoM should converge toward 100%
with finer grids and more padding.

Varies: grid density (points per transition width) and padding factor.
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})


def build_grin_profile(k_steep, nb, pts_per_width, domain_half_widths):
    """Build GRIN logistic profile with specified resolution."""
    dx = 1.0 / (pts_per_width * k_steep)
    xmin = -domain_half_widths / k_steep
    xmax = -xmin
    N = 1 + int(np.floor((xmax - xmin) / dx))
    xx = np.linspace(xmin, xmax, N)
    u = (nb**2 - 1) / (1 + np.exp(k_steep * xx)) + 1

    # Compute ε'' via derivative-HT-integrate
    ud = np.gradient(u, xx)
    pad_ht = 8 * N
    ud_pad = np.pad(ud, (pad_ht, pad_ht), mode='constant')
    vd_pad = np.imag(hilbert(ud_pad))
    vd = vd_pad[pad_ht:pad_ht + N]
    v = cumulative_trapezoid(vd, xx, initial=0)
    v -= np.linspace(v[0], v[-1], N)

    return xx, u, v


def fom_derivative(x, u, v, pad_factor):
    """FoM from FT of dε/dx (the standard method)."""
    ud = np.gradient(u, x)
    vd = np.gradient(v, x)
    z = ud + 1j * vd
    pad = pad_factor * len(x)
    z = np.pad(z, (pad, pad), mode='constant')
    dx = x[1] - x[0]
    Z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=dx))
    pwr = np.abs(Z)**2
    E_pos = np.sum(pwr[k > 0])
    E_neg = np.sum(pwr[k < 0])
    return 100 * max(0.0, (E_pos - E_neg) / (E_pos + E_neg))


def fom_integrated(x, u, v, pad_factor):
    """FoM from derivative → FT → ÷ ik (the integrated method)."""
    ud = np.gradient(u, x)
    vd = np.gradient(v, x)
    z = ud + 1j * vd
    pad = pad_factor * len(x)
    z = np.pad(z, (pad, pad), mode='constant')
    dx = x[1] - x[0]
    Z_d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=dx))
    # Divide by ik
    eps_hat = np.zeros_like(Z_d)
    nonzero = k != 0
    eps_hat[nonzero] = Z_d[nonzero] / (1j * k[nonzero])
    pwr = np.abs(eps_hat)**2
    E_pos = np.sum(pwr[k > 0])
    E_neg = np.sum(pwr[k < 0])
    return 100 * max(0.0, (E_pos - E_neg) / (E_pos + E_neg))


# === Sweep 1: vary grid density (pts per transition width) ===
k_steep = 100; nb = 1.7
pts_list = [50, 100, 200, 400, 800]
pad_factor_fixed = 8
domain_hw_fixed = 20  # ±20/k_steep

fom_deriv_pts = []
fom_integ_pts = []
for pts in pts_list:
    xx, u, v = build_grin_profile(k_steep, nb, pts, domain_hw_fixed)
    fd = fom_derivative(xx, u, v, pad_factor_fixed)
    fi = fom_integrated(xx, u, v, pad_factor_fixed)
    fom_deriv_pts.append(fd)
    fom_integ_pts.append(fi)
    print(f"pts={pts:4d}, N={len(xx):6d}: deriv FoM={fd:.2f}%, integrated FoM={fi:.2f}%")

# === Sweep 2: vary padding factor ===
pts_fixed = 100
pad_list = [4, 8, 16, 32, 64]

fom_deriv_pad = []
fom_integ_pad = []
for pf in pad_list:
    xx, u, v = build_grin_profile(k_steep, nb, pts_fixed, domain_hw_fixed)
    fd = fom_derivative(xx, u, v, pf)
    fi = fom_integrated(xx, u, v, pf)
    fom_deriv_pad.append(fd)
    fom_integ_pad.append(fi)
    print(f"pad_factor={pf:3d}: deriv FoM={fd:.2f}%, integrated FoM={fi:.2f}%")

# === Sweep 3: vary domain size ===
domain_hw_list = [10, 20, 40, 80, 160]

fom_deriv_dom = []
fom_integ_dom = []
for dhw in domain_hw_list:
    xx, u, v = build_grin_profile(k_steep, nb, pts_fixed, dhw)
    fd = fom_derivative(xx, u, v, pad_factor_fixed)
    fi = fom_integrated(xx, u, v, pad_factor_fixed)
    fom_deriv_dom.append(fd)
    fom_integ_dom.append(fi)
    print(f"domain=±{dhw}/k_steep, N={len(xx):6d}: deriv FoM={fd:.2f}%, integrated FoM={fi:.2f}%")

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: grid density
ax = axes[0]
ax.plot(pts_list, fom_deriv_pts, 'o-', color='#2ca02c', label=r'FT of $d\varepsilon/dx$', markersize=8)
ax.plot(pts_list, fom_integ_pts, 's--', color='#d62728', label=r'FT of $d\varepsilon/dx$ $\div\, ik$', markersize=8)
ax.axhline(100, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Grid points per transition width')
ax.set_ylabel('Spectral FoM (%)')
ax.set_title('Sweep: grid density\n(pad=8, domain=±0.2 μm)')
ax.legend()
ax.set_ylim(0, 105)

# Panel 2: padding factor
ax = axes[1]
ax.plot(pad_list, fom_deriv_pad, 'o-', color='#2ca02c', label=r'FT of $d\varepsilon/dx$', markersize=8)
ax.plot(pad_list, fom_integ_pad, 's--', color='#d62728', label=r'FT of $d\varepsilon/dx$ $\div\, ik$', markersize=8)
ax.axhline(100, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Padding factor')
ax.set_ylabel('Spectral FoM (%)')
ax.set_title('Sweep: padding\n(100 pts/width, domain=±0.2 μm)')
ax.legend()
ax.set_ylim(0, 105)

# Panel 3: domain size
dom_labels = [f'±{dhw}' for dhw in domain_hw_list]
ax = axes[2]
ax.plot(domain_hw_list, fom_deriv_dom, 'o-', color='#2ca02c', label=r'FT of $d\varepsilon/dx$', markersize=8)
ax.plot(domain_hw_list, fom_integ_dom, 's--', color='#d62728', label=r'FT of $d\varepsilon/dx$ $\div\, ik$', markersize=8)
ax.axhline(100, color='gray', ls=':', alpha=0.5)
ax.set_xlabel(r'Domain half-width ($\times 1/k_{\rm steep}$)')
ax.set_ylabel('Spectral FoM (%)')
ax.set_title('Sweep: domain size\n(100 pts/width, pad=8)')
ax.legend()
ax.set_ylim(0, 105)

plt.tight_layout()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
outpath = os.path.join(FIGDIR, 'fig_fom_convergence_test.png')
plt.savefig(outpath)
plt.close()
print(f"\nSaved: {outpath}")
