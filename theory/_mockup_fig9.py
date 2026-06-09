"""Fig 9 — Application: sKK on a real sapphire substrate, backside reflectance. FIRST DRAFT.

The d/lambda scale-invariance DELIBERATELY breaks here (sapphire is dispersive) -> real
wavelength axis returns. Closes the loop: ideal ceiling vs achievable vs baselines.
  One axis, four R_back vs wavelength curves:
    bare sapphire | lossless GRIN | idealized continuous sKK (ceiling) | realistic sKK (gated)
Reuses the sapphire machinery in skk_analysis_consolidated (load_sapphire_data, Rback_*).
NOTE: an application schematic PNG does NOT exist yet (PPT export pending) -> top row omitted
      in this draft; add it (C.APPLICATION_SCHEMATIC) once available.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import tmm_helper as tmm_h
import theory.skk_fig_common as C
import skk_analysis_consolidated as cons   # sapphire data + Rback helpers

C.apply_style()
ANGLE_DEG, POL = 80, 's'          # backside ellipsometry-like incidence (draft defaults)

lamdata, ndata, kdata = cons.load_sapphire_data()

# coating profiles — canonical k=8 logistic (same sKK coating as the theory figs)
xx, e_re, e_im = C.logistic_profile()
gate = tmm_h.smooth_gate(e_re, 1.1, 0.05)

nc_skk,  dc_skk  = tmm_h.discretize_profile(xx, e_re + 1j * e_im, delta=C.DELTA)
nc_grin, dc_grin = tmm_h.discretize_profile(xx, e_re + 0j,        delta=C.DELTA)
nc_real, dc_real = tmm_h.discretize_profile(xx, e_re + 1j * e_im * gate, delta=C.DELTA)

R_bare, _ = cons.Rback_bulk_wl(ndata, kdata, lamdata, ANGLE_DEG, POL)
R_grin, _ = cons.Rback_vs_wavelength(nc_grin, dc_grin, ndata, kdata, lamdata, ANGLE_DEG, POL)
R_skk,  _ = cons.Rback_vs_wavelength(nc_skk,  dc_skk,  ndata, kdata, lamdata, ANGLE_DEG, POL)
R_real, _ = cons.Rback_vs_wavelength(nc_real, dc_real, ndata, kdata, lamdata, ANGLE_DEG, POL)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(lamdata, R_bare, color='gray',     lw=2, label='bare sapphire')
ax.plot(lamdata, R_grin, color=C.C_EPS_RE, lw=2, label='lossless GRIN')
ax.plot(lamdata, R_skk,  color=C.C_R,      lw=2, label='idealized sKK (ceiling)')
ax.plot(lamdata, R_real, color=C.C_DERIV,  lw=2, ls='--', label='realistic sKK (gated)')
ax.set_yscale('log')
ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel('Backside reflectance')
ax.legend(fontsize=9, title=f'{POL}-pol, {ANGLE_DEG}$^\\circ$')
C.subplot_letter(ax, 'a')

fig.tight_layout()
C.save(fig, os.path.join('theory', '_mockup_fig9.png'))
