"""Rigorous scale-invariance test + large-d/lambda diagnostic. Throwaway.

Builds the sKK/GRIN coating at TWO absolute scales through the FULL pipeline
(build -> discretize -> classify -> TMM):
  x1:  T=5 um  (k=8)
  x4:  T=20 um (k=2)   [thickness x4, wavelengths x4, same d/lambda grid]
If scale invariance holds through discretization + coherence classification,
the two ratio colorplots are identical. Then a normal-incidence diagnostic
table reports R, T, A and incoherent-layer counts vs d/lambda.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import tmm_helper as tmm_h
from tmm_helper import _make_c_list
from skk_analysis_consolidated import R_nobulk_2D

nb = 1.7; delta = 0.01


def build(T):
    k = 40.0 / T
    dx = 1 / (100 * k); xmin = -20 / k; xmax = -xmin
    nx = 1 + int(np.floor((xmax - xmin) / dx))
    xx = np.linspace(xmin, xmax, nx)
    e_re = tmm_h.logistic(xx, k, nb)
    e_im = tmm_h.ht_derivative(xx, e_re)
    nc_skk, dc_skk = tmm_h.discretize_profile(xx, e_re + 1j * e_im, delta=delta)
    nc_grin, dc_grin = tmm_h.discretize_profile(xx, e_re + 0j, delta=delta)
    return dict(T=T, nc_skk=nc_skk, dc_skk=dc_skk, nc_grin=nc_grin, dc_grin=dc_grin)


x1 = build(5.0)
x4 = build(20.0)
print(f"layer counts  sKK: x1={len(x1['nc_skk'])}  x4={len(x4['nc_skk'])}   "
      f"GRIN: x1={len(x1['nc_grin'])}  x4={len(x4['nc_grin'])}")
print(f"dc_skk ratio (x4/x1) min/max: "
      f"{np.min(np.array(x4['dc_skk'])/np.array(x1['dc_skk'])):.6f} / "
      f"{np.max(np.array(x4['dc_skk'])/np.array(x1['dc_skk'])):.6f}  (expect 4.0)")

dlam = np.logspace(np.log10(0.1), np.log10(100), 110)
angles = np.arange(0, 90, 2.0)
pols = ['s', 'p']


def ratio_grid(case, pol):
    lam = case['T'] / dlam
    R_s = R_nobulk_2D(case['nc_skk'], case['dc_skk'], nb, lam, angles, pol)
    R_g = R_nobulk_2D(case['nc_grin'], case['dc_grin'], nb, lam, angles, pol)
    return R_g / np.clip(R_s, 1e-15, None)


ratio_x1 = {p: ratio_grid(x1, p) for p in pols}
ratio_x4 = {p: ratio_grid(x4, p) for p in pols}
for p in pols:
    d = np.abs(np.log10(ratio_x1[p]) - np.log10(ratio_x4[p]))
    print(f"{p}-pol  max|log10(ratio_x1) - log10(ratio_x4)| = {np.max(d):.3e}  "
          f"(median {np.median(d):.3e})")

# Save x4 colorplot for visual comparison
norm = matplotlib.colors.LogNorm(vmin=1, vmax=1e5)
cmap = matplotlib.cm.inferno.copy(); cmap.set_over('white')
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, p in zip(axes, pols):
    im = ax.pcolormesh(angles, dlam, ratio_x4[p].T, norm=norm, cmap=cmap, shading='auto')
    ax.set_yscale('log'); ax.set_xlabel('Angle in Air (degrees)')
    ax.set_title(f'$%s$-pol  (T=20 um, x4)' % p, fontsize=13)
axes[0].set_ylabel(r'$d/\lambda$')
fig.colorbar(im, ax=axes, location='right', shrink=0.9, pad=0.02, extend='max').set_label(
    r'$R_{\mathrm{GRIN}} / R_{\mathrm{sKK}}$')
out = os.path.join(_PROJECT_ROOT, 'theory', '_mockup_fig3_collapsed_x4.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)

# ---- Large-d/lambda diagnostic at normal incidence (x1 coating) ----
print("\nNormal incidence (s-pol), x1 coating:")
print(f"{'d/lam':>7} {'R_sKK':>11} {'R_GRIN':>11} {'T_sKK':>10} {'A_sKK':>9} "
      f"{'inc_sKK':>8} {'inc_GRIN':>9} {'Nlay':>6}")
n_t_skk = [nb] + list(x1['nc_skk']) + [1.0]
d_t_skk = [np.inf] + list(x1['dc_skk']) + [np.inf]
n_t_grin = [nb] + list(x1['nc_grin']) + [1.0]
d_t_grin = [np.inf] + list(x1['dc_grin']) + [np.inf]
Nlay = len(x1['nc_skk'])
for r in (0.5, 1, 2, 5, 10, 20, 50, 100):
    lam = 5.0 / r
    T_s, R_s, A_s = tmm_h.TRA(n_t_skk, d_t_skk, lamb=lam, angle=0.0, pol='s')
    T_g, R_g, A_g = tmm_h.TRA(n_t_grin, d_t_grin, lamb=lam, angle=0.0, pol='s')
    c_skk = _make_c_list(n_t_skk, d_t_skk, lam, 0.0)
    c_grin = _make_c_list(n_t_grin, d_t_grin, lam, 0.0)
    inc_skk = c_skk[1:-1].count('i')      # exclude semi-inf ends
    inc_grin = c_grin[1:-1].count('i')
    print(f"{r:>7.1f} {R_s:>11.3e} {R_g:>11.3e} {T_s:>10.3e} {A_s:>9.4f} "
          f"{inc_skk:>8d} {inc_grin:>9d} {Nlay:>6d}")
