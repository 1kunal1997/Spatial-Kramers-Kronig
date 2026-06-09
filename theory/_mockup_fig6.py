"""Fig 6 — Spectral FoM method (geometry-agnostic). FIRST DRAFT.

Introduces the spectral one-sidedness FoM as the diagnostic tool used to quantify Figs 7 & 8.
Method: derivative -> FFT -> divide by ik (tmm_h.skk_spectral_fom). The FoM = fraction of
spectral power in k>0 (one-sided => reflectionless). Geometry-agnostic; no TMM here.
  a  eps' and eps'' of the logistic profile (vs x)
  b  spectral power |eps_hat(k)|^2 with green/red one-sidedness shading + FoM annotation
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

fom, k, pwr = tmm_h.skk_spectral_fom(xx, e_re, e_im)
print(f"spectral FoM = {fom:.3f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4))

# a — profile
C.profile_panel(axa, xx, e_re, e_im, 'a', xlabel=r'$x$',
                xlim=(-lam, lam), xticks=[-lam, 0, lam],
                xticklabels=[r'$-\lambda$', r'$0$', r'$\lambda$'])

# b — spectral FoM (helper draws green/red shading on log scale)
tmm_h.plot_spectral_fom(axb, k, pwr, fom)
C.subplot_letter(axb, 'b')

fig.tight_layout()
C.save(fig, os.path.join('theory', '_mockup_fig6.png'))
