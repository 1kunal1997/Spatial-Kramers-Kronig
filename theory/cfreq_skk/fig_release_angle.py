"""Figures for item 5 (release after a growing drive; angle dependence). Reads
data/release_angle.npz written by exp_release_angle.py; no physics computed here.
  python theory/cfreq_skk/fig_release_angle.py            # both
  python theory/cfreq_skk/fig_release_angle.py release    # B5
  python theory/cfreq_skk/fig_release_angle.py angle      # B6
"""
import sys, os, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
import numpy as np
import matplotlib.pyplot as plt
import theory.skk_fig_common as C

C.apply_style()
OUT = os.path.join(_HERE, 'figures')
D = np.load(os.path.join(_HERE, 'data', 'release_angle.npz'))
META = json.loads(str(D['meta']))
B = META['budgets']


def _smooth(x, n):
    """Running mean over n samples (one drive period) to show instantaneous power."""
    return np.convolve(x, np.ones(n) / n, mode='same')


def fig_release():
    fig, axs, _ = C.panel_grid(2, 2, margin_r=0.42, col_gap=0.6)
    (ax_a, ax_b), (ax_c, ax_d) = axs
    dt, t0, tstop = META['dt'], META['t0'], META['t_stop']
    t = D['trace_t'] - t0; per = int(round(1 / dt))
    norm = np.max(D['trace_drive'] ** 2)
    m = t > 0.5
    ax_a.loglog(t[m], _smooth(D['trace_drive'] ** 2, per)[m] / norm, color='gray', lw=C.LW_THIN, label='incident')
    ax_a.loglog(t[m], _smooth(D['trace_r'] ** 2, per)[m] / norm, color=C.C_R, lw=C.LW_THIN, label='reflected (left inc.)')
    ax_a.loglog(t[m], _smooth(D['trace_tr'] ** 2, per)[m] / norm, color=C.C_T, lw=C.LW_THIN, label='transmitted')
    ax_a.axvline(tstop - t0, color='k', ls=':', lw=C.LW_GUIDE)
    ax_a.text(tstop - t0, 3e-1, ' drive off', fontsize=5.5, va='top')
    ax_a.set_xlim(1, 1e4); ax_a.set_ylim(1e-12, 3)
    ax_a.set_xlabel(r'time ($T_r$)'); ax_a.set_ylabel('power (norm.)')
    ax_a.legend(loc='lower left', fontsize=5)

    keys = [k for k in B if ',th=' not in k]
    labels = [k.replace('Q=', 'Q=').replace(',N=', '\nN=') for k in keys]
    x = np.arange(len(keys)); bottom = np.zeros(len(keys))
    for comp, col, lab, hatch in [('L_R_prompt', C.C_R, 'R during drive', None),
                                  ('L_R_delayed', C.C_R, 'R after drive', '////'),
                                  ('L_T', C.C_T, 'T (total)', None), ('L_A', C.C_A, 'A (total)', None)]:
        v = np.array([B[k][comp] for k in keys])
        ax_b.bar(x, v, bottom=bottom, color=col, alpha=0.35 if hatch else 0.8, hatch=hatch,
                 edgecolor=col, lw=0.4, label=lab)
        bottom += v
    ax_b.set_xticks(x); ax_b.set_xticklabels(labels, fontsize=5.5)
    ax_b.set_ylim(0, 1.22); ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax_b.set_ylabel('energy fraction (left inc.)')
    ax_b.legend(loc='upper center', fontsize=5, ncol=4, handlelength=1.2, columnspacing=0.8)
    ax_b.minorticks_off()

    w = D['spec_w']
    for N, ls in [(400, ':'), (4000, '--'), (40000, '-')]:
        ax_c.semilogy(w, D[f'spec_RL_N{N}'], color=C.C_R, ls=ls, lw=C.LW_THIN, label=f'N = {N}')
    ax_c.set_xlabel(r'real frequency $\omega/\omega_r$'); ax_c.set_ylabel(r'$|r_L(\omega)|^2$, $Q=10^4$')
    ax_c.set_xlim(w[0], w[-1]); ax_c.set_ylim(1e-8, 1.5); ax_c.legend(loc='lower right', fontsize=5)

    y = D['design_y']
    a2 = (2 * np.pi) ** 2                                  # omega_r^2 in these units
    ax_d.plot(y, D['design_w02'] / a2, color=C.C_EPS_RE, label=r'$\omega_0^2(y)/\omega_r^2$')
    axr = ax_d.twinx()
    axr.plot(y, D['design_wp2'] / D['design_wp2'].max(), color='#ff7f0e', label=r'$\omega_p^2/\max$')
    ax_d.set_xlabel(r'$y/\lambda_r$'); ax_d.set_ylabel(r'$\omega_0^2(y)/\omega_r^2$', color=C.C_EPS_RE)
    axr.set_ylabel(r'$\omega_p^2(y)$ (norm.)', color='#ff7f0e')
    ax_d.tick_params(axis='y', labelcolor=C.C_EPS_RE); axr.tick_params(axis='y', labelcolor='#ff7f0e')
    ax_d.set_xlim(y[0], y[-1])
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')])
    C.save(fig, os.path.join(OUT, 'B5_release.png'))


def fig_angle():
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.42, col_gap=0.55)
    ax_a, ax_b = axs[0]
    th = D['ang_theta']
    for pol, ls in [('s', '-'), ('p', '--')]:
        a = D[f'ang_{pol}']
        ax_a.semilogy(th, a[:, 0], color=C.C_R, ls=ls, label=rf'$|r_L|^2$ ({pol})')
        ax_a.semilogy(th, a[:, 1], color='#7f7f7f', ls=ls, label=rf'$|r_R|^2$ ({pol})')
        ax_b.semilogy(th, a[:, 3], color=C.C_R, ls=ls, label=rf'$R_L$ ({pol})')
        ax_b.semilogy(th, a[:, 5], color=C.C_A, ls=ls, label=rf'$A_L$ ({pol})')
    for (thd, pol) in META['td_angles']:
        b = B[f'Q=1e4,N=400,th={thd:g},{pol}']
        ax_a.plot([thd], [b['L_ratio_drive_max'] ** 2], marker='o' if pol == 's' else 's',
                  mfc='none', mec='k', ms=4, ls='none')
    ax_a.plot([], [], marker='o', mfc='none', mec='k', ls='none', ms=4, label='time domain')
    ax_a.set_xlabel(r'angle of incidence (deg)'); ax_a.set_ylabel(r'at $\omega_c$ (growing drive)')
    ax_a.set_xlim(0, 90); ax_a.set_ylim(1e-8, 2); ax_a.legend(loc='upper left', fontsize=5, ncol=2)
    ax_b.set_xlabel(r'angle of incidence (deg)'); ax_b.set_ylabel(r'real $\omega$, band mean, $Q=100$')
    ax_b.set_xlim(0, 90); ax_b.set_ylim(1e-8, 2); ax_b.legend(loc='lower right', fontsize=5, ncol=2)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B6_angles.png'))


FIGS = {'release': fig_release, 'angle': fig_angle}
if __name__ == '__main__':
    for nm in (sys.argv[1:] or list(FIGS)):
        FIGS[nm]()
