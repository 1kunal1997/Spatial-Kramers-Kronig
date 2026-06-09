"""Fig 5 — Shape optimality (PUNCHLINE). FIRST DRAFT.

5 loss shapes with MATCHED total loss, R vs d/lambda, NO wavelength averaging, converged delta.
Reframe: the HT-prescribed (sKK) loss shape is uniquely optimal at d/lambda ~ O(1).
Reuses the shape generators from skk_analysis_consolidated (make_*_profile).
Layout [D3 open] -- draft shows one panel, s-pol, normal incidence; add p / second angle later.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import tmm_helper as tmm_h
import theory.skk_fig_common as C
import skk_analysis_consolidated as cons   # reuse make_*_profile (no setup() runs on import)

C.apply_style()

xx, e_re, e_im_skk = C.logistic_profile()      # canonical k=8, domain ±20/k, d=5 um
target = float(np.trapezoid(e_im_skk, xx))     # total loss to match across shapes

shapes = {
    'sKK (HT)':   (e_im_skk,                                   C.C_R),
    'constant':   (cons.make_constant_profile(xx, target),     'gray'),
    'gaussian':   (cons.make_gaussian_profile(xx, target),     C.C_EPS_RE),
    'double':     (cons.make_double_peak_profile(xx, target),  C.C_DERIV),
    'random':     (cons.make_random_profile(xx, target),       '#6e016b'),
}

d_tot = C.coating_thickness(C.build_stack(xx, e_re + 1j * e_im_skk, n_out=1.0)[1])
dlam = np.logspace(np.log10(0.1), np.log10(15), 50)
lams = d_tot / dlam

fig, ax = plt.subplots(figsize=(7, 5))
for name, (eim, col) in shapes.items():
    n_t, d_t = C.build_stack(xx, e_re + 1j * eim, n_out=1.0)
    _, R, _ = C.sweep_tra_vs_wavelength(n_t, d_t, lams, angle_deg=0.0, pol='s')
    ax.plot(dlam, R, color=col, lw=2, label=name)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$d/\lambda$'); ax.set_ylabel('Reflectance')
ax.legend(fontsize=9, title='loss shape (matched total)')
C.dlambda_top_axis(ax, d_tot)
C.subplot_letter(ax, 'a')

fig.tight_layout()
C.save(fig, os.path.join('theory', '_mockup_fig5.png'))
