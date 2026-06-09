"""Fig 8 — Lossy air -> gating (sigma_gating). FIRST DRAFT.

Core realizability fix: the HT-prescribed eps'' extends into the eps' -> 1 (air-like) region,
which is unphysical ("lossy air"). A smooth tanh gate on eps'(x) suppresses loss there.
  a  eps'' full vs sigma-gated (vs x), plus the gate g(x)
  b  R vs d/lambda: full sKK vs gated sKK (gating costs a little R but is realizable)
Uses tmm_h.smooth_gate(eps_re, eps0, sigma). VERIFY gate API/params when editing.
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

eps0, sigma = 1.1, 0.05                       # gate threshold near air; draft params
gate = tmm_h.smooth_gate(e_re, eps0, sigma)  # ~ [0,1], suppresses loss where eps' -> air
e_im_gated = e_im * gate

n_full,  d_full  = C.build_stack(xx, e_re + 1j * e_im,       n_out=1.0)
n_gate,  d_gate  = C.build_stack(xx, e_re + 1j * e_im_gated, n_out=1.0)
d_tot = C.coating_thickness(d_full)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.5))

# a — eps'' full vs gated + gate
axa.plot(xx, e_im, color=C.C_EPS_IM, lw=2, label=r"$\epsilon''$ full")
axa.plot(xx, e_im_gated, color=C.C_EPS_IM, lw=2, ls='--', label=r"$\epsilon''$ gated")
axar = axa.twinx()
axar.plot(xx, gate, color='gray', lw=1.2, alpha=0.6)
axar.set_ylabel(r'gate $g(x)$', color='gray'); axar.set_ylim(-0.05, 1.05)
axa.set_xlabel(r'$x$'); axa.set_ylabel(r"$\epsilon''(x)$", color=C.C_EPS_IM)
axa.set_xticks([-lam, 0, lam]); axa.set_xticklabels([r'$-\lambda$', r'$0$', r'$\lambda$'])
axa.legend(fontsize=8); C.subplot_letter(axa, 'a')

# b — R vs d/lambda, full vs gated
dlam = np.logspace(np.log10(0.1), np.log10(15), 45)
lams = d_tot / dlam
_, Rf, _ = C.sweep_tra_vs_wavelength(n_full, d_full, lams, angle_deg=0.0, pol='s')
_, Rg, _ = C.sweep_tra_vs_wavelength(n_gate, d_gate, lams, angle_deg=0.0, pol='s')
axb.plot(dlam, Rf, color=C.C_R, lw=2, label='full sKK')
axb.plot(dlam, Rg, color=C.C_DERIV, lw=2, label='gated sKK')
axb.set_xscale('log'); axb.set_yscale('log')
axb.set_xlabel(r'$d/\lambda$'); axb.set_ylabel('Reflectance')
axb.legend(fontsize=9); C.subplot_letter(axb, 'b')
C.dlambda_top_axis(axb, d_tot)

fig.tight_layout()
C.save(fig, os.path.join('theory', '_mockup_fig8.png'))
