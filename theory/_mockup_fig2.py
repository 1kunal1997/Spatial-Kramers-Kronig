"""Fig 2 — The DiHiTI method (logistic). 5-panel: schematic on top + 2x2 below.

The core contribution. Geometry: nb | eps(x) | air.
  top   schematic (theory_schematic.png; eps_b | eps(x) | air, theta_b, R, T)
  a     naive HT: eps' (blue) + naive eps'' = Im H[eps'] (red) -> pathology (eps'' < 0)
  b     naive TRA vs theta_b -> T > 1, A < 0  (GAIN -- FAILS)
  c     DiHiTI: eps' (blue, left) ; eps'' (red) + H[deps'/dx] pre-integration curve (orange) (right)
  d     corrected TRA vs theta_b -> R ~ 0, T+A = 1  (WORKS)

Reading: left col = construction, right col = reflectance; top row naive/fails,
bottom corrected/works; b<->d = before/after. Conventions live in skk_fig_common.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from scipy.signal import hilbert            # naive HT -- no helper by design
import tmm_helper as tmm_h
import theory.skk_fig_common as C

C.apply_style()

# --- profile + DiHiTI intermediates ---
lam = C.LAM_REF                              # 2.5 um snapshot -> d/lambda = 2 (d=5 um)
xx, e_re, e_im = C.logistic_profile()        # canonical k=8, domain ±20/k = ±2.5 (±20 widths)
# Horsley x-axis: physical x with ticks at multiples of lambda (domain ±2.5 = ±lambda here)
XTICKS = [-lam, 0, lam]
XTICKLABELS = [r'$-\lambda$', r'$0$', r'$\lambda$']
e_im_naive = np.imag(hilbert(e_re))          # naive HT of eps' directly (pathological)
dedx = np.gradient(e_re, xx)                 # DiHiTI intermediates (ht_derivative internals)
h_dedx = np.imag(hilbert(dedx))              # e_im (corrected) returned by logistic_profile

# --- TMM stacks: nb | coating | air, theta_b axis ---
angles = C.theta_b_deg()
n_naive, d_naive = C.build_stack(xx, e_re + 1j * e_im_naive, n_out=1.0)
n_corr,  d_corr  = C.build_stack(xx, e_re + 1j * e_im,       n_out=1.0)
Tn, Rn, An = C.sweep_tra_vs_theta_b(n_naive, d_naive, angles)
Tc, Rc, Ac = C.sweep_tra_vs_theta_b(n_corr,  d_corr,  angles)
print(f"naive    : T max {Tn.max():.2f}, A min {An.min():.3f}  (gain -> fails)")
print(f"corrected: R range [{Rc.min():.2e}, {Rc.max():.2e}], |T+A-1| max {np.max(np.abs(Tc+Ac-1)):.1e}")

# --- layout: schematic band on top spanning both cols, 2x2 results grid below ---
# panel_grid places every axes at an explicit inch rectangle, so each data panel is EXACTLY
# 4:3 (width fixed to the column, height derived; papers scroll down). has_twin=True widens
# the column gap to clear the right-side eps'' axis on the left-column panels (b, d).
# schem_panels=1.0 = schematic band height in units of one data-panel height (kept modest
# so the data panels, not the schematic, are the visual focus).
fig, axs, ax_schem = C.panel_grid(nrows=2, ncols=2, schem_panels=1.0, has_twin=True)
(axa, axb), (axc, axd) = axs

# a — schematic image (spans both columns; imshow's equal aspect centers it in the band)
ax_schem.imshow(mpimg.imread(C.THEORY_SCHEMATIC))
ax_schem.axis('off')

# b — naive HT profile (pathology)
C.profile_panel(axa, xx, e_re, e_im_naive, ylab_im=r"$\mathcal{H}[\epsilon'(x)] = \epsilon''(x)$",
                xlabel=r'$x$', xlim=(-lam, lam), xticks=XTICKS, xticklabels=XTICKLABELS)

# c — naive TRA (gain, fails)
C.tra_panel(axb, angles, Tn, Rn, An, mark_gain=True)

# d — DiHiTI: eps' on LEFT axis (blue), eps'' on RIGHT axis (red).
#     The intermediates deps'/dx and H[deps'/dx] are overlaid in orange on a HIDDEN axis
#     (no scale -- shape cues only), each labelled inline. eps'' is their integral (Ti step).
axcr = axc.twinx()
axc.plot(xx, e_re, color=C.C_EPS_RE, lw=C.LW_MAIN)
axcr.plot(xx, e_im, color=C.C_EPS_IM, lw=C.LW_MAIN)
axc.set_xlabel(r'$x$'); axc.set_ylabel(r"$\epsilon'(x)$", color=C.C_EPS_RE)
axcr.set_ylabel(r"$\int \mathcal{H}[d\epsilon'/dx] = \epsilon''(x)$", color=C.C_EPS_IM)
axc.tick_params(axis='y', labelcolor=C.C_EPS_RE)
axcr.tick_params(axis='y', labelcolor=C.C_EPS_IM)
axcr.set_yticks([0.0, 0.5, 1.0, 1.5])                      # one decimal, not 0.25 clutter
axcr.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axcr.yaxis.set_minor_locator(AutoMinorLocator(2))         # one minor between majors (2nd axis)
axc.set_xlim(-lam, lam); axc.set_xticks(XTICKS); axc.set_xticklabels(XTICKLABELS)

# Intermediates on a hidden axis (no scale -- shape cues only). Each dashed curve is the
# DERIVATIVE of the solid curve of the SAME color: deps'/dx (light blue) = d/dx of eps';
# H[deps'/dx] = deps''/dx (light red) = d/dx of eps''. The HT is the bridge between them.
C_DERIV_RE = '#7fb8da'   # light blue  -- kin to eps' (its derivative)
C_DERIV_IM = '#d98c8c'   # light red   -- kin to eps'' (its derivative, = H[deps'/dx])
axo = axc.twinx()
axo.plot(xx, dedx, color=C_DERIV_RE, lw=C.LW_THIN, ls='--')
axo.plot(xx, h_dedx, color=C_DERIV_IM, lw=C.LW_THIN, ls='--')
axo.set_yticks([])
for sp in axo.spines.values():
    sp.set_visible(False)
# inline labels for the dashed curves. Blended transform: x in axes fraction, y in eps'
# (left/blue-axis) DATA units, so each label pins to a specific blue-axis tick.
tb = axc.get_yaxis_transform()
axc.text(0.1, 2.0, r"$d\epsilon'/dx$", transform=tb, color=C_DERIV_RE,
         fontsize=7, ha='left', va='center',                    # right next to the 2.0 blue tick
         bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
axc.text(0.07, 2.5, r"$\mathcal{H}[d\epsilon'/dx]$", transform=tb, color=C_DERIV_IM,
         fontsize=7, ha='left', va='center',                    # just above the 2.5 blue tick
         bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

# e — corrected TRA (works)
C.tra_panel(axd, angles, Tc, Rc, Ac, ylim=(-0.05, 1.05))

# Panel letters: ONE call, last. Each sits at its panel's top-left, in line with that
# panel's y-axis label (same x as the label) -> letters auto-align down the figure.
C.label_panels(fig, [(ax_schem, 'a'), (axa, 'b'), (axb, 'c'), (axc, 'd'), (axd, 'e')])

C.save(fig, os.path.join('theory', '_mockup_fig2.png'))
