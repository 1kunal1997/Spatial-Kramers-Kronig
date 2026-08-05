"""SI Fig 4 — measured sapphire optical constants used in the application section.

Single panel, dual y-axis: ordinary-ray refractive index n (left, blue) and extinction
coefficient kappa (right, red) over the 2-5 um band, exactly the arrays the main-text
application figure feeds to the TMM. Dual-axis styling follows the profile panels of the
main text (colour-matched axis labels and tick labels).

Data are loaded through skk_analysis_consolidated.load_sapphire_data() rather than re-read
here, so this figure cannot drift from the numbers the application section actually used.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import theory.skk_fig_common as C
import skk_analysis_consolidated as cons     # same loader the application figure uses

C.apply_style()

lamdata, ndata, kdata = cons.load_sapphire_data()
print('lambda %.3f-%.3f um, %d points' % (lamdata[0], lamdata[-1], len(lamdata)))
print('n     %.4f-%.4f' % (ndata.min(), ndata.max()))
print('kappa %.3e-%.3e' % (kdata.min(), kdata.max()))

fig, axs, _ = C.panel_grid(nrows=1, ncols=1, has_twin=True)
ax = axs[0, 0]
axr = ax.twinx()
ax._skk_twin = [axr]                          # so relayout measures + moves the twin

ax.plot(lamdata, ndata, color=C.C_EPS_RE, lw=C.LW_MAIN)
axr.plot(lamdata, kdata, color=C.C_EPS_IM, lw=C.LW_MAIN)

ax.set_xlabel(r'$\lambda$ ($\mu$m)')
ax.set_ylabel(r'$n$', color=C.C_EPS_RE)
axr.set_ylabel(r'$\kappa$', color=C.C_EPS_IM)
ax.tick_params(axis='y', labelcolor=C.C_EPS_RE)
axr.tick_params(axis='y', labelcolor=C.C_EPS_IM)
ax.set_xlim(lamdata.min(), lamdata.max())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# kappa spans several decades across the band -> log right axis, linear left.
axr.set_yscale('log')

C.relayout_grid(fig, axs, buffer_in=0.05)

C.save(fig, os.path.join('theory', '_mockup_SI_4.png'))
