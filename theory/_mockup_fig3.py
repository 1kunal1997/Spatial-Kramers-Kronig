"""Fig 3 — Reflectance performance (no-bulk), 2x2. Ported to skk_fig_common (fig2 format).

Geometry: nb | logistic coating | air. The coating thickness L is held CONSTANT (capital L ->
fixed parameter); the swept variable is the wavelength lambda. Because the non-dispersive system
is scale-invariant, sweeping lambda at fixed L traverses the same landscape as sweeping L/lambda
-- so we plot the SWEPT variable directly as lambda in units of L (lambda/L), ticked as fractions
of L (L/10 ... 10L), parallel to the profile x-axis which is in units of lambda. Plotting the
swept variable means NO secondary wavelength axis is needed. The physics still depends only on
the ratio L/lambda; prose may refer to L/lambda freely (see paper_status / figure_plan).
  3a  R_sKK colorplot, theta_b x lambda/L (viridis, log R)
  3b  A_sKK colorplot, theta_b x lambda/L (cividis)
  3c  6-curve R vs theta_b at lambda/L = 0.5 (== L/lambda = 2): bare/GRIN/sKK x s/p
  3d  6-curve R vs lambda/L at theta_b = 30 deg: bare/GRIN/sKK x s/p (reflectance only)

lambda INCREASES left->right (and bottom->top on the colorplots) for now; see paper_status TODO
on whether to flip it so the coating's relative thickness grows along the sweep.
Conventions (axis labeling, colors, layout) live in skk_fig_common.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullLocator
import theory.skk_fig_common as C

C.apply_style()

# --- canonical logistic coating (k=8, domain ±20/k, L=5 um) ---
xx, e_re, e_im = C.logistic_profile()
n_skk,  d_skk  = C.build_stack(xx, e_re + 1j * e_im, n_out=1.0)              # primary delta
n_grin, d_grin = C.build_stack(xx, e_re + 0j,        n_out=1.0)
n_skk_c, d_skk_c = C.build_stack(xx, e_re + 1j * e_im, n_out=1.0, delta=C.DELTA_COLORPLOT)
L_tot = C.coating_thickness(d_skk)                                          # FIXED coating thickness
bare_n, bare_d = [C.NB, 1.0], [np.inf, np.inf]
print(f"coating thickness L = {L_tot:.3f} um")

# --- sweep axis = lambda/L. Tick anchors are fractions of the fixed L (parallel to the
#     profile ticks, which are multiples of the fixed lambda). Range lambda/L in [L/15, 10L]
#     == L/lambda in [0.1, 15] (the figure_plan trim). ---
LOL_MIN, LOL_MAX = 1.0 / 15.0, 10.0
LOL_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10]
LOL_LABELS = [r'$L/10$', r'$L/5$', r'$L/2$', r'$L$', r'$2L$', r'$5L$', r'$10L$']

angles = C.theta_b_deg()                                  # 0..theta_c (logistic exits to air)

# --- layout: 2x2 of EXACTLY-4:3 panels. Colorplots (a,b) carve their colorbar from inside
#     their own 4:3 box via colorbar_in_panel, so no twin/extra column room is needed. ---
fig, axs, _ = C.panel_grid(nrows=2, ncols=2)
(axa, axb), (axc, axd) = axs

# --- 3a / 3b — colorplots (s-pol), lambda/L on the y-axis (log) ---
ang_c = C.theta_b_deg(step=1.5)
lol_c = np.logspace(np.log10(LOL_MIN), np.log10(LOL_MAX), 35)   # lambda/L grid
lam_c = lol_c * L_tot                                           # physical lambda per column
Rmap, Amap = C.tra_map(n_skk_c, d_skk_c, ang_c, lam_c, pol='s')

m1 = axa.pcolormesh(ang_c, lol_c, Rmap.T, cmap=C.CMAP_RSKK,
                    norm=LogNorm(vmin=max(Rmap.min(), 1e-8), vmax=Rmap.max()), shading='auto')
m2 = axb.pcolormesh(ang_c, lol_c, Amap.T, cmap=C.CMAP_ASKK, shading='auto')
for ax, m, lab in [(axa, m1, r'$R_{\rm sKK}$'), (axb, m2, r'$A_{\rm sKK}$')]:
    ax.set_yscale('log'); ax.set_ylim(LOL_MIN, LOL_MAX)
    ax.set_xlabel(r'$\theta_b$ (deg)'); ax.set_ylabel(r'$\lambda$')
    ax.yaxis.set_major_locator(FixedLocator(LOL_TICKS)); ax.set_yticklabels(LOL_LABELS)
    ax.yaxis.set_minor_locator(NullLocator())              # clean fraction-of-L ticks only
    C.colorbar_in_panel(fig, ax, m, lab)

# config table shared by panels c & d: color = config (bare gray, GRIN orange, sKK green);
# solid = s-pol, dashed = p-pol.
cfg = {'Bare': ('gray', bare_n, bare_d),
       'GRIN': (C.C_DERIV, n_grin, d_grin),
       'sKK':  (C.C_R, n_skk, d_skk)}

# --- 3c — 6-curve R vs theta_b at lambda/L = 0.5 (== L/lambda = 2) ---
lam_3c = L_tot / 2.0                                        # lambda/L = 0.5
for pol, ls in [('s', '-'), ('p', '--')]:
    for name, (col, nn, dd) in cfg.items():
        _, Rc, _ = C.sweep_tra_vs_theta_b(nn, dd, angles, lam=lam_3c, pol=pol)
        axc.plot(angles, Rc, color=col, ls=ls, lw=C.LW_MAIN, label=f'{name} (${pol}$)')
axc.set_yscale('log'); axc.set_xlim(0, C.THETA_C_DEG)
axc.set_xlabel(r'$\theta_b$ (deg)'); axc.set_ylabel('Reflectance')
axc.legend(ncol=2, columnspacing=1.0, handlelength=1.6)    # one legend for the figure (c & d share it)

# --- 3d — 6-curve R vs lambda/L at theta_b = 30 deg (reflectance only, no absorption) ---
lol_d = np.logspace(np.log10(LOL_MIN), np.log10(LOL_MAX), 60)
lam_d = lol_d * L_tot
for pol, ls in [('s', '-'), ('p', '--')]:
    for name, (col, nn, dd) in cfg.items():
        _, Rd, _ = C.sweep_tra_vs_wavelength(nn, dd, lam_d, angle_deg=20.0, pol=pol)
        axd.plot(lol_d, Rd, color=col, ls=ls, lw=C.LW_MAIN)
axd.set_xscale('log'); axd.set_yscale('log'); axd.set_xlim(LOL_MIN, LOL_MAX)
axd.set_xlabel(r'$\lambda$'); axd.set_ylabel('Reflectance')
axd.xaxis.set_major_locator(FixedLocator(LOL_TICKS)); axd.set_xticklabels(LOL_LABELS)
axd.xaxis.set_minor_locator(NullLocator())

# Panel letters: ONE call, last (heatmap axes already repositioned by colorbar_in_panel; their
# left edges are unchanged, so the letters still line up with the y-labels).
C.label_panels(fig, [(axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')])

C.save(fig, os.path.join('theory', '_mockup_fig3.png'))
