"""Fig 4 — Where does sKK beat GRIN? Collapsed ratio map. FIRST DRAFT.

R_GRIN / R_sKK over (theta_b, d/lambda), 2 panels (s, p). inferno + white over-color for
saturated wins. Message: advantage peaks at d/lambda ~ 2-3; GRIN catches up by ~10.
Replaces the old 8-panel thickness colorplots. Untested-run draft (slow colorplot).
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import tmm_helper as tmm_h
import theory.skk_fig_common as C

C.apply_style()

xx, e_re, e_im = C.logistic_profile()        # canonical k=8, domain ±20/k, d=5 um
n_skk,  d_skk  = C.build_stack(xx, e_re + 1j * e_im, n_out=1.0, delta=C.DELTA_COLORPLOT)
n_grin, d_grin = C.build_stack(xx, e_re + 0j,        n_out=1.0, delta=C.DELTA_COLORPLOT)
d_tot = C.coating_thickness(d_skk)

angles = C.theta_b_deg(step=1.5)
lams = d_tot / np.logspace(np.log10(0.1), np.log10(12), 35)   # d/lambda in [0.1, 12] (plan range)
DL = d_tot / lams

cmap = mpl.colormaps[C.CMAP_RATIO].copy(); cmap.set_over('white')
norm = LogNorm(vmin=1, vmax=1e5)        # match _mockup_fig3_collapsed; white over = sKK wins big

# shared colorbar, log d/lambda y, 2 panels (matches _mockup_fig3_collapsed.png)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
titles = {'s': r'$s$-polarization', 'p': r'$p$-polarization'}
for ax, pol, letter in [(axes[0], 's', 'a'), (axes[1], 'p', 'b')]:
    Rg, _ = C.tra_map(n_grin, d_grin, angles, lams, pol=pol)
    Rs, _ = C.tra_map(n_skk,  d_skk,  angles, lams, pol=pol)
    ratio = Rg / np.clip(Rs, 1e-15, None)
    im = ax.pcolormesh(angles, DL, ratio.T, cmap=cmap, norm=norm, shading='auto')
    ax.set_yscale('log'); ax.set_xlabel(r'$\theta_b$ (deg)')
    ax.set_title(titles[pol], fontsize=13, fontweight='bold')
    C.subplot_letter(ax, letter)
axes[0].set_ylabel(r'$d/\lambda$')
cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.9, pad=0.02, extend='max')
cbar.set_label(r'$R_{\mathrm{GRIN}} / R_{\mathrm{sKK}}$', fontsize=12)
C.save(fig, os.path.join('theory', '_mockup_fig4.png'))
