"""Fig 1 — Intro: canonical sKK (Lorentzian), STANDALONE (no schematic).

Geometry: eps_b | Lorentzian | eps_b (background nb both sides -> NO air interface, NO TIR).
Canonical conventions (skk_fig_common, LOCKED 2026-06-07):
  xi = gamma = 0.1*lambda (Horsley);  domain = ±100*gamma (1/x tail; 100x keeps it visible).
  1a  eps'/eps'' vs x (units of lambda); MAIN zoomed to ±10 lambda, INSET shows the feature (±lambda).
  1b  TRA vs angle of incidence, FULL 0-90 deg -> R ~ 0 at all propagating angles
      (mild grazing kick-up = finite-truncation residual at 100 gamma, accepted; not physics).
All lengths in units of lambda; non-dispersive => wavelength-independent.

Layout/format mirrors fig2: deterministic 4:3 panels via panel_grid, dual-axis profile_panel,
tra_panel, and a single trailing label_panels call. Conventions live in skk_fig_common.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import theory.skk_fig_common as C

C.apply_style()
lam = C.LAM_REF
xx, ee, gam = C.lorentzian_profile(lam)            # xi=0.1*lam, domain ±100*gam
xmax = C.M_LORENTZ * gam                            # = 10*lam
print(f"Lorentzian: gam={gam:.3f} (=0.1 lam), domain ±{xmax:.1f} um (±{xmax/lam:.0f} lam = ±{C.M_LORENTZ} gam)")

# --- TMM: nb | Lorentzian | nb, full angle range (no TIR cutoff) ---
n_list, d_list = C.build_stack(xx, ee, n_out=C.NB)
angles = np.arange(0, 90, 0.5)
T, R, A = C.sweep_tra_vs_theta_b(n_list, d_list, angles, lam=lam)
print(f"R range [{R.min():.2e}, {R.max():.2e}], R@80deg={R[np.argmin(abs(angles-80))]:.2e}")

# --- layout: single row of two EXACTLY-4:3 panels (no schematic). has_twin=True widens the
#     column gap to clear panel a's right-side eps'' axis. ---
fig, axs, _ = C.panel_grid(nrows=1, ncols=2, has_twin=True)
(axa, axb), = axs

# a — MAIN = full profile (±10 lambda), dual-axis (eps'' dashed to avoid overlap);
#     INSET = zoomed feature (±lambda); shape only -> x-ticks kept, no title, no y-ticks/labels.
axr = C.profile_panel(axa, xx, ee.real, ee.imag, xlabel=r'$x$',
                xlim=(-xmax, xmax), xticks=[-xmax, 0, xmax],
                xticklabels=[r'$-10\lambda$', r'$0$', r'$10\lambda$'], im_ls='--')
axin = axa.inset_axes([0.59, 0.575, 0.37, 0.40])
axin.plot(xx, ee.real, color=C.C_EPS_RE, lw=C.LW_THIN)
axin.set_xlim(-lam, lam)
axin.set_xticks([-lam, 0, lam]); axin.set_xticklabels([r'$-\lambda$', '', r'$\lambda$'], fontsize=6.5)
axin.set_yticks([])
axin2 = axin.twinx()
axin2.plot(xx, ee.imag, color=C.C_EPS_IM, lw=C.LW_THIN, ls='--')
axin2.set_yticks([])

'''
uncomment this if you want to control individual minor ticks
axa.xaxis.set_minor_locator(AutoMinorLocator(2))   # 1 minor between x majors
axa.yaxis.set_minor_locator(AutoMinorLocator(4))   # 3 minors on left (blue) y
axr.yaxis.set_minor_locator(AutoMinorLocator(2))   # overrides the module default

# panel b
axb.xaxis.set_minor_locator(AutoMinorLocator(5))   # 4 minors between x majors
axb.yaxis.set_minor_locator(AutoMinorLocator(2))
'''

# b — TRA vs angle, full 0-90 deg (no TIR cutoff for the eps_b|...|eps_b Lorentzian)
C.tra_panel(axb, angles, T, R, A, xlabel='Angle of Incidence (deg)', xmax=90)

# Panel letters: ONE call, last. Each sits at its panel's top-left, in line with that
# panel's y-axis label -> letters auto-align across the figure.
C.label_panels(fig, [(axa, 'a'), (axb, 'b')])

C.save(fig, os.path.join('theory', '_mockup_fig1.png'))
