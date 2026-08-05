"""SI Fig 2 — loss shapes — reflectance vs lambda at two ADDITIONAL incidence angles (theta_b = 10, 35 deg).

Companion to the main combined figure (_mockup_fig5_6_combined), which shows theta_b = 20 deg. This
is the EXACT fig5 b-e template — shared "Reflectance" y-label, shared lambda x-label, identical
column/row spacing and tick-anchored letter placement — with the schematic / loss-shape panel a
REMOVED (the shapes live in the main figure). The shapes are identified by ONE centred 2x3 legend
between the panel rows instead of being re-plotted.

theta_b = 10 deg (not 0 deg) for the top row: at normal incidence s- and p-pol are identical, which
would make the two top panels redundant. 35 deg (bottom) matches the main figure's high-angle case.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter
import tmm_helper as tmm_h
import theory.skk_fig_common as C
import skk_analysis_consolidated as cons   # reuse make_*_profile (no setup() runs on import)

C.apply_style()

# --- profiles (identical to fig5 / the combined figure) ---
xx, e_re, e_im_skk = C.logistic_profile()
target = float(np.trapezoid(e_im_skk, xx))
_sig_g = (xx[-1] - xx[0]) / 8 * 0.7
_g = np.exp(-xx**2 / (2 * _sig_g**2)); _g *= target / np.trapezoid(_g, xx)
e_im_gauss = np.maximum(_g, 0.0)

SHAPES = [
    ('sKK',          e_im_skk,                                  '#8B0000', 's'),
    ('Constant',     cons.make_constant_profile(xx, target),    '#1f77b4', 'o'),
    ('Gaussian',     e_im_gauss,                                '#ff7f0e', '^'),
    ('Double peaks', cons.make_double_peak_profile(xx, target), '#9467bd', 'D'),
    ('Random',       cons.make_random_profile(xx, target),      '#2ca02c', 'v'),
]
ZORDER = {'sKK': 6, 'Gaussian': 5, 'Double peaks': 4, 'Random': 3, 'Constant': 2}
stacks = [(name, C.build_stack(xx, e_re + 1j * eim, n_out=1.0), col, mk)
          for name, eim, col, mk in SHAPES]
L_tot = C.coating_thickness(stacks[0][1][1])

# --- sweep axis = lambda/L (fig5) ---
PTS_PER_DECADE = 10
LOL_MIN, LOL_MAX = 8 / 100.0, 10.0
n_lol = int(round(np.log10(LOL_MAX / LOL_MIN) * PTS_PER_DECADE)) + 1
lol = np.logspace(np.log10(LOL_MIN), np.log10(LOL_MAX), n_lol)
lams = lol * L_tot
LOL_TICKS = [0.1, 0.2, 1, 2, 10]
LOL_LABELS = [r'$L/10$', r'$L/5$', r'$L$', r'$2L$', r'$10L$']
LOL_BAND = (1 / 5.0, 2.0)

# rows = angle (10 deg top, 35 deg bottom); cols = polarization (s, p)
PANELS = [(0, 0, 's', 10), (0, 1, 'p', 10),
          (1, 0, 's', 35), (1, 1, 'p', 35)]


def _title(pol, th):
    return pol + r'-polarization, $\theta_b = %d^\circ$' % th


fig, axs, _ = C.panel_grid(nrows=2, ncols=2)         # no schematic band
(axa, axb), (axc, axd) = axs
grid = {(0, 0): axa, (0, 1): axb, (1, 0): axc, (1, 1): axd}

for r, c, pol, th in PANELS:
    ax = grid[(r, c)]
    for _xb in LOL_BAND:
        ax.axvline(_xb, color='black', lw=C.LW_GUIDE, ls='--', zorder=1)
    for name, (nl, dl), col, mk in stacks:
        _, R, _ = C.sweep_tra_vs_wavelength(nl, dl, lams, angle_deg=th, pol=pol)
        ax.plot(lol, R, mk + '-', color=col, lw=C.LW_MAIN, ms=3.2, mew=0, zorder=ZORDER[name])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(LOL_MIN, LOL_MAX)
    ax.xaxis.set_major_locator(FixedLocator(LOL_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(LOL_LABELS))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(_title(pol, th))
    if r == 0:
        ax.tick_params(labelbottom=False)            # shared x: only bottom row shows tick labels

# shared single x/y labels (both rows are reflectance); wide first (only) inter-row gap holds the legend.
C.relayout_grid(fig, axs, col_ws_in=0.2, row_ws_in=0.2, row1_gap=0.55,
                shared_xlabel=r'$\lambda$', shared_ylabel='Reflectance')

# panel letters: fig5-exact (tick-anchored; no individual y-labels here so tightbbox == ticks).
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_Wpx = fig.bbox.width
for ax, lab in [(axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')]:
    p = ax.get_position()
    x_let = ax.yaxis.get_tightbbox(_rend).x0 / _Wpx
    fig.text(x_let - 0.004, p.y1, r'$\mathbf{%s}$' % lab, fontsize=10, va='top', ha='right')

# --- one shared legend (2 rows x 3 cols), centred in the gap between the panel rows ---
handles = [plt.Line2D([], [], color=col, marker=mk, lw=C.LW_MAIN, ms=3.2, mew=0, label=name)
           for name, _stk, col, mk in stacks]
ptop, pright, pbot = axa.get_position(), axb.get_position(), axc.get_position()
xc = (ptop.x0 + pright.x1) / 2
yc = (pbot.y1 + ptop.y0) / 2
fig.legend(handles=handles, ncol=3, loc='center', bbox_to_anchor=(xc, yc), fontsize=7)

C.save(fig, os.path.join('theory', '_mockup_SI_2.png'))
