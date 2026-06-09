"""Throwaway mockup: 6-curve angle panel for Fig 2 (bare/GRIN/sKK x s/p).
Reproduces the no-bulk angle sweep from skk_analysis_consolidated.setup()
exactly, then plots all 6 curves on one axis to judge legibility.
Not a paper figure — delete after review.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import tmm_helper as tmm_h

nb = 1.7
delta = 0.01
lam = 3.0
angle_list = np.arange(0, 90, 1)

# Thick coating, identical to setup() lines 488-492
xx = np.linspace(-2.5, 2.5, 5001)
e_re = (nb**2 - 1) / (1 + np.exp(4 * xx)) + 1
e_im = tmm_h.ht_derivative(xx, e_re)
nc_full, dc_full = tmm_h.discretize_profile(xx, e_re + 1j * e_im, delta=delta)
nc_grin, dc_grin = tmm_h.discretize_profile(xx, e_re + 0j, delta=delta)


def R_vs_angle(nc, dc, pol, bare=False):
    if bare:
        n_t = [nb, 1.0]; d_t = [np.inf, np.inf]
    else:
        n_t = [nb] + list(nc) + [1.0]
        d_t = [np.inf] + list(dc) + [np.inf]
    R = np.zeros(len(angle_list))
    for i, ang_air in enumerate(angle_list):
        ang_nb = np.arcsin(np.sin(np.radians(ang_air)) / nb)
        _, Ri, _ = tmm_h.TRA(n_t, d_t, lamb=lam, angle=ang_nb, pol=pol)
        R[i] = Ri
    return R

curves = {}
for pol in ('s', 'p'):
    curves[('bare', pol)] = R_vs_angle(None, None, pol, bare=True)
    curves[('grin', pol)] = R_vs_angle(nc_grin, dc_grin, pol)
    curves[('skk',  pol)] = R_vs_angle(nc_full, dc_full, pol)

# ---- Plot: color = config, solid = s, dashed = p ----
col = {'bare': '#7f7f7f', 'grin': '#ff7f0e', 'skk': '#2ca02c'}
lab = {'bare': 'Bare', 'grin': 'GRIN', 'skk': 'sKK'}

plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.2})
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for cfg in ('bare', 'grin', 'skk'):
    ax.semilogy(angle_list, curves[(cfg, 's')], '-',  color=col[cfg], lw=2.2,
                label=f'{lab[cfg]} ($s$)')
    ax.semilogy(angle_list, curves[(cfg, 'p')], '--', color=col[cfg], lw=2.2,
                label=f'{lab[cfg]} ($p$)')
ax.set_xlabel('Angle in Air (°)')
ax.set_ylabel(r'$R$')
ax.set_xlim(0, 89)
ax.legend(fontsize=9, ncol=3, columnspacing=1.0, handlelength=1.8)
fig.tight_layout()
out = os.path.join(_PROJECT_ROOT, 'theory', '_mockup_fig2_6curve.png')
fig.savefig(out, dpi=150)
print('saved', out)

# Print a few numbers for the wavelength<->thickness redundancy discussion
print('\nR_sKK(s) at select angles:', {a: f'{curves[("skk","s")][a]:.2e}' for a in (0, 45, 80)})
