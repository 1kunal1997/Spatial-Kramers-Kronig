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

# Panel width = ONE main-text 2-column panel, centred in the column — NOT stretched to the full
# column width. Same treatment the main text gives its lone loss-shape panel
# (_mockup_fig5_6_combined.py: panel a is re-placed at one panel width, centred). A full-width
# single panel is ~2x the area of every other panel in the paper, so the shared 7.5/6.5 pt type
# reads far too small against it.
TARGET_PANEL_W_IN = 2.2      # realised data-box width of a 2-column panel (fig2/fig3, SI1)
_W_PROV = 3.0                # provisional narrow canvas; corrected exactly below

fig, axs, _ = C.panel_grid(nrows=1, ncols=1, has_twin=True, width_in=_W_PROV)
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

# Lay out on the provisional canvas, measure the realised data box, then correct: every
# reservation (y-label + ticks left, twin-axis block right, buffers) is a fixed inch amount
# INDEPENDENT of width_in, so panel_w = width_in - const and one step lands exactly on target.
C.relayout_grid(fig, axs, buffer_in=0.05, width_in=_W_PROV)
W_SMALL = _W_PROV + (TARGET_PANEL_W_IN - ax.get_position().width * _W_PROV)
C.relayout_grid(fig, axs, buffer_in=0.05, width_in=W_SMALL)

# Widen the canvas to the full column and re-centre the (unchanged) panel in it. Capture the
# inch geometry FIRST: set_size_inches leaves axes positions as fractions, which would stretch
# the panel with the canvas. Height is already correct — it was derived from the small panel.
H = fig.get_size_inches()[1]
_boxes = [(a, a.get_position()) for a in [ax] + C._skk_twins(ax)]
_dx = (C.COL_WIDTH_IN - W_SMALL) / 2.0
fig.set_size_inches(C.COL_WIDTH_IN, H)
for a, p in _boxes:
    a.set_position([(p.x0 * W_SMALL + _dx) / C.COL_WIDTH_IN, p.y0,
                    p.width * W_SMALL / C.COL_WIDTH_IN, p.height])
print('panel data box %.3f x %.3f in on a %.2f x %.2f in canvas'
      % (ax.get_position().width * C.COL_WIDTH_IN, ax.get_position().height * H,
         C.COL_WIDTH_IN, H))

C.save(fig, os.path.join('theory', '_mockup_SI_4.png'))
