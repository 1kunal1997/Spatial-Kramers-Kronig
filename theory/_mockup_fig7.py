"""Fig 7 — Discretization / finite-N limit (NEW). FIRST DRAFT.

The advisor's requested discretization figure. Frames the staircase floor as a real
fabrication limit for BOTH sKK and GRIN (ties to the fab team's stratified layers).
  a  staircased eps'(x) at a few layer counts N (overlaid on the continuous profile)
  b  resulting R vs d/lambda for each N, showing the staircase floor flattening at large d/lambda
Keep staircase-floor (finite-N) vs T-underflow (float limit) distinct in the caption.
Reference prototype: theory/_mockup_discretization_test.py.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import tmm_helper as tmm_h
import theory.skk_fig_common as C

C.apply_style()
lam = C.LAM_REF
xx, e_re, e_im = C.logistic_profile()        # canonical k=8, domain ±20/k = ±2.5 = ±lambda
ee = e_re + 1j * e_im

# A few coarse layer counts via different delta values (larger delta -> fewer layers)
delta_vals = [0.04, 0.01, C.DELTA]    # coarse -> fine
colors = [C.C_DERIV, C.C_EPS_RE, C.C_R]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.5))

# a — continuous profile + staircases
axa.plot(xx, e_re, color='k', lw=1.2, alpha=0.4, label='continuous')
for dl, col in zip(delta_vals, colors):
    nc, dc = tmm_h.discretize_profile(xx, ee, delta=dl)
    edges = np.concatenate([[xx[0]], xx[0] + np.cumsum(dc)])
    nlayer = len(dc)
    re_layers = np.real(np.array(nc) ** 2)
    axa.step(edges[:-1], re_layers, where='post', color=col, lw=1.5, label=f'N={nlayer}')
axa.set_xlabel(r'$x$'); axa.set_ylabel(r"$\epsilon'(x)$")
axa.set_xticks([-lam, 0, lam]); axa.set_xticklabels([r'$-\lambda$', r'$0$', r'$\lambda$'])
axa.legend(fontsize=8); C.subplot_letter(axa, 'a')

# b — R vs d/lambda per N (sKK)
d_tot = C.coating_thickness(C.build_stack(xx, ee, n_out=1.0)[1])
dlam = np.logspace(np.log10(0.1), np.log10(20), 45)
lams = d_tot / dlam
for dl, col in zip(delta_vals, colors):
    nc, dc = tmm_h.discretize_profile(xx, ee, delta=dl)
    n_t = [C.NB] + list(nc) + [1.0]; d_t = [np.inf] + list(dc) + [np.inf]
    _, R, _ = C.sweep_tra_vs_wavelength(n_t, d_t, lams, angle_deg=0.0, pol='s')
    axb.plot(dlam, R, color=col, lw=2, label=f'N={len(dc)}')
axb.set_xscale('log'); axb.set_yscale('log')
axb.set_xlabel(r'$d/\lambda$'); axb.set_ylabel('Reflectance')
axb.legend(fontsize=8, title='layer count'); C.subplot_letter(axb, 'b')
C.dlambda_top_axis(axb, d_tot)

fig.tight_layout()
C.save(fig, os.path.join('theory', '_mockup_fig7.png'))
