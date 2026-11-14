"""Figures B7-B9 (stable virtual gain from passive two-pole media). Reads data/stable_gain.npz
(exp_stable_gain.py) and data/fdtd_twopole.npz (fdtd_twopole.py); no physics computed here.
  python theory/cfreq_skk/fig_stable_gain.py              # all
  python theory/cfreq_skk/fig_stable_gain.py slab         # B7: before/after, homogeneous slab (B7_eit, B7_fano2)
  python theory/cfreq_skk/fig_stable_gain.py skk          # B8: layered one-way-invisible sKK
  python theory/cfreq_skk/fig_stable_gain.py angle        # B9: angles + EIT layer parameters
  python theory/cfreq_skk/fig_stable_gain.py slab_thick   # B7 1 lam vs 5 lam (B7_{eit,fano2}_thick)
  python theory/cfreq_skk/fig_stable_gain.py eps_map      # |eps(w)| vs |t(w)| maps (B7_{eit,fano2}_eps_map)
slab_thick / eps_map and the B7 m = 1 label read data/slab_poles.npz (exp_slab_poles.py).
"""
import sys, os, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
import numpy as np
import matplotlib.pyplot as plt
import theory.skk_fig_common as C

C.apply_style()
OUT = os.path.join(_HERE, 'figures')
D = np.load(os.path.join(_HERE, 'data', 'stable_gain.npz'))
F = np.load(os.path.join(_HERE, 'data', 'fdtd_twopole.npz'))
META = json.loads(str(D['meta']))
P = np.load(os.path.join(_HERE, 'data', 'slab_poles.npz'))
PMETA = json.loads(str(P['meta']))
LABEL_FP_M = 1        # TEMPORARY (2026-11-14, Kunal): label this Fabry-Perot order in B7 panel d;
                      # set to None to remove
C_ORANGE = '#ff7f0e'
C_GRAY = '#7f7f7f'
ETA_SLAB = 0.05
MAP_LIM = (-1.0, 3.0)
FD_EVERY = 100                       # FDTD samples are 0.05 T_r apart -> a marker every 5 T_r


def _cbar_right(fig, ax, im, label, gap_in=0.08, w_in=0.08):
    """Colorbar in the reserved right margin, beside `ax`, without touching its 4:3 box."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + gap_in / W, pos.y0, w_in / W, pos.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label(label); cb.outline.set_linewidth(0.6)
    return cb


def _map(ax, key, poles_list, eta):
    im = ax.pcolormesh(D['map_x'] if 'slab' in key else D['skk_map_x'],
                       D['map_y'] if 'slab' in key else D['skk_map_y'],
                       np.clip(D[key], *MAP_LIM), cmap='magma', vmin=MAP_LIM[0], vmax=MAP_LIM[1],
                       shading='auto', rasterized=True)
    ax.axhline(-eta, color='w', lw=C.LW_GUIDE, ls='--')
    ax.axhline(0, color='w', lw=C.LW_GUIDE, ls=':')
    ax.plot([1], [-eta], marker='*', color='c', ms=7, ls='none')
    ax.text(1.008, -eta + 0.004, r'$\omega_c$', color='c', fontsize=6)
    for poles, mk in poles_list:
        pp = poles[poles.real > 0]
        ax.plot(pp.real, pp.imag, marker=mk, color='w', ms=4, mew=0.9, ls='none')
    ax.set_xlabel(r"Re $\omega/\omega_r$"); ax.set_ylabel(r"Im $\omega/\omega_r$")
    return im


def fig_slab(stable='eit'):
    """B7: same eps(w_c) = 1 - 0.3i in a homogeneous 1-lambda slab. Left column: single Lorentz
    (Long et al.); right column: one STABLE two-pole medium (`stable` = 'eit' or 'fano2').
    One figure per stable medium (split 2026-11-13: the shared map mixed two different slabs).
    c, d: log10|t(w)| of THAT slab over complex w (bright = slab poles); white markers = poles of
    the material eps(w) itself (the slab poles pile up around them, where |eps| -> inf, t -> 0).
    e, f: |t(t)/t_free(t)| = |tau(t)|, dotted = frequency-domain |tau(w_c)| (same for all media)."""
    title = {'eit': 'EIT', 'fano2': 'two-line'}[stable]
    fig, axs, _ = C.panel_grid(3, 2, has_twin=True, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12)
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axs
    w = D['slab_w']
    for ax, nm, txt in ((ax_a, 'lorentz', 'single Lorentz'), (ax_b, stable, title)):
        e = D[f'slab_eps_{nm}']
        axr = C.profile_panel(ax, w, e.real, e.imag, xlabel=r'real frequency $\omega/\omega_r$',
                              ylab_im=r"$\epsilon''(\omega)$", xlim=(w[0], w[-1]))
        ax.set_ylabel(r"$\epsilon'(\omega)$", color=C.C_EPS_RE)
        axr.set_ylim(0, 0.42); ax.set_ylim(0.8, 1.25)
        axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
        ax.axvline(1, color='gray', lw=C.LW_GUIDE, ls=':')
        ax.text(0.06, 0.92, txt, transform=ax.transAxes, fontsize=6, va='top')

    im = _map(ax_c, 'slab_map_lorentz', [(D['slab_poles_lorentz'], 'x')], ETA_SLAB)
    _map(ax_d, f'slab_map_{stable}', [(D[f'slab_poles_{stable}'], 'x')], ETA_SLAB)
    for ax, txt in ((ax_c, 'single Lorentz'), (ax_d, title)):
        ax.set_ylim(D['map_y'][0], D['map_y'][-1])
        ax.text(0.03, 0.06, txt + r' slab, $1\lambda_r$', transform=ax.transAxes, color='w', fontsize=6)
    if LABEL_FP_M is not None:
        _label_fp(ax_d, stable, 1, LABEL_FP_M)
    _cbar_right(fig, ax_d, im, r'$\log_{10}|t(\omega)|$')

    t = D['slab_t']; m = t > 1
    for ax, nm, txt in ((ax_e, 'lorentz', 'single Lorentz'), (ax_f, stable, title)):
        tc = META['slab'][nm]['tau_c']
        ax.semilogy(t[m], D[f'slab_tau_{nm}'][m] * tc, color=C.C_T, lw=C.LW_THIN, label=txt)
        ft, fr = F[f'slab_{nm}_t'], F[f'slab_{nm}_tau']
        k = (ft > 2) & (ft < 70)
        ax.plot(ft[k][::FD_EVERY], fr[k][::FD_EVERY] * tc, marker='o', mfc='none', mec='k', ms=3,
                mew=0.6, ls='none', label='FDTD')
        ax.axhline(tc, color=C.C_T, lw=C.LW_GUIDE, ls=':')
        ax.set_xlim(0, 100); ax.set_ylim(1e-2, 1e3)
        ax.set_xlabel(r'time after turn-on ($T_r$)')
        ax.set_ylabel(r'$|t(t)/t_{\rm free}(t)|$')
        ax.legend(loc='upper left', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')])
    C.save(fig, os.path.join(OUT, f'B7_{stable}_slab.png'))


def _label_fp(ax, nm, L, m):
    """Circle + label the slab pole of Fabry-Perot order m (exp_slab_poles.fp_order)."""
    pl, ms = P[f'poles_{nm}_L{L:g}'], P[f'm_{nm}_L{L:g}']
    k = np.nonzero(np.abs(ms - m) < 1e-3)[0]
    if not k.size:
        return
    z = pl[k[0]]
    ax.plot(z.real, z.imag, marker='o', ms=7, mfc='none', mec='c', mew=0.7, ls='none')
    ax.annotate(rf'$m={m}$', (z.real, z.imag), xytext=(z.real - 0.045, z.imag - 0.03),
                color='c', fontsize=6, arrowprops=dict(arrowstyle='-', color='c', lw=0.6))


def _pmap(ax, key, xkey, ykey, nm, lim=MAP_LIM):
    """Like _map, for data/slab_poles.npz maps (material-pole x markers of medium nm)."""
    im = ax.pcolormesh(P[xkey], P[ykey], np.clip(P[key], *lim), cmap='magma', vmin=lim[0],
                       vmax=lim[1], shading='auto', rasterized=True)
    ax.axhline(-ETA_SLAB, color='w', lw=C.LW_GUIDE, ls='--')
    ax.axhline(0, color='w', lw=C.LW_GUIDE, ls=':')
    ax.plot([1], [-ETA_SLAB], marker='*', color='c', ms=7, ls='none')
    ax.text(1.008, -ETA_SLAB + 0.004, r'$\omega_c$', color='c', fontsize=6)
    mp = P[f'mat_poles_{nm}']; mp = mp[mp.real > 0]
    ax.plot(mp.real, mp.imag, marker='x', color='w', ms=4, mew=0.9, ls='none')
    ax.set_xlabel(r"Re $\omega/\omega_r$"); ax.set_ylabel(r"Im $\omega/\omega_r$")
    ax.set_ylim(P[ykey][0], P[ykey][-1])
    return im


def fig_slab_thick(stable='eit'):
    """B7 follow-up: the SAME homogeneous slab (eps(w_c) = 1 - 0.3i, `stable` medium) at 1 lam_r
    (left, = right column of B7) and 5 lam_r (right). a, b: eps(w) (identical). c, d: log10|t(w)|;
    at 5 lam Fabry-Perot poles rise above the drive line (the slab 'lases' in the co-decaying
    frame). e, f: |t(t)/t_free(t)|; dotted = frequency-domain |tau(w_c)|, which the 5 lam slab
    never reaches. Data: exp_slab_poles.py; FDTD circles (1 lam only): fdtd_twopole.py."""
    title = {'eit': 'EIT', 'fano2': 'two-line'}[stable]
    fig, axs, _ = C.panel_grid(3, 2, has_twin=True, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12)
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axs
    w, e = D['slab_w'], D[f'slab_eps_{stable}']
    for ax in (ax_a, ax_b):
        axr = C.profile_panel(ax, w, e.real, e.imag, xlabel=r'real frequency $\omega/\omega_r$',
                              ylab_im=r"$\epsilon''(\omega)$", xlim=(w[0], w[-1]))
        ax.set_ylabel(r"$\epsilon'(\omega)$", color=C.C_EPS_RE)
        axr.set_ylim(0, 0.42); ax.set_ylim(0.8, 1.25)
        axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
        ax.axvline(1, color='gray', lw=C.LW_GUIDE, ls=':')
        ax.text(0.06, 0.92, title, transform=ax.transAxes, fontsize=6, va='top')
    t = P['td_t']; m = t > 1
    for axm, axt, L in ((ax_c, ax_e, 1.0), (ax_d, ax_f, 5.0)):
        key = f'{stable}_L{L:g}'
        im = _pmap(axm, f'map_{key}', 'map_x', 'map_y', stable)
        axm.text(0.03, 0.06, rf'{title} slab, ${L:g}\lambda_r$', transform=axm.transAxes,
                 color='w', fontsize=6)
        tc = PMETA[stable][f'L{L:g}']['tau_c']
        axt.semilogy(t[m], P[f'td_{key}'][m], color=C.C_T, lw=C.LW_THIN, label=rf'{title}, ${L:g}\lambda_r$')
        if L == 1.0:
            ft, fr = F[f'slab_{stable}_t'], F[f'slab_{stable}_tau']
            k = (ft > 2) & (ft < 70)
            axt.plot(ft[k][::FD_EVERY], fr[k][::FD_EVERY] * tc, marker='o', mfc='none', mec='k',
                     ms=3, mew=0.6, ls='none', label='FDTD')
        axt.axhline(tc, color=C.C_T, lw=C.LW_GUIDE, ls=':')
        axt.set_xlim(0, 100); axt.set_ylim(1e-2, 1e6)
        axt.set_xlabel(r'time after turn-on ($T_r$)')
        axt.set_ylabel(r'$|t(t)/t_{\rm free}(t)|$')
        axt.legend(loc='upper left', fontsize=5)
    _cbar_right(fig, ax_d, im, r'$\log_{10}|t(\omega)|$')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')])
    C.save(fig, os.path.join(OUT, f'B7_{stable}_thick.png'))


def fig_eps_map(stable='eit'):
    """a: log10|eps(w)| of the `stable` medium over complex w (bright = material poles, dark =
    zeros of eps). b: log10|t(w)| of the 1 lam_r slab of that medium on the same window. Same
    x markers (material poles) in both. Window taller than B7's so every material pole is in it.
    Data: exp_slab_poles.py."""
    title = {'eit': 'EIT', 'fano2': 'two-line'}[stable]
    gap = C.YLABEL_W_IN + 0.08 + 0.08 + C.YLABEL_W_IN + 0.12 + C.CENTER_GAP_IN
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12, col_gap=gap)
    ax_a, ax_b = axs[0]
    im = _pmap(ax_a, f'tallmap_eps_{stable}', 'tall_x', 'tall_y', stable)
    ax_a.text(0.03, 0.04, rf'{title} medium, $\epsilon(\omega)$', transform=ax_a.transAxes,
              color='w', fontsize=6)
    _cbar_right(fig, ax_a, im, r'$\log_{10}|\epsilon(\omega)|$')
    im = _pmap(ax_b, f'tallmap_t_{stable}', 'tall_x', 'tall_y', stable)
    ax_b.text(0.03, 0.04, rf'{title} slab, $1\lambda_r$, $t(\omega)$', transform=ax_b.transAxes,
              color='w', fontsize=6)
    _cbar_right(fig, ax_b, im, r'$\log_{10}|t(\omega)|$')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, f'B7_{stable}_eps_map.png'))


def fig_skk():
    """B8: layered one-way-invisible sKK built from passive EIT layers."""
    fig, axs, _ = C.panel_grid(3, 2, has_twin=True, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12)
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axs
    case = 'simple_eta0.05'
    y, chi = D[f'{case}_y'], D[f'{case}_chi']
    Y = META['simple']['Y']
    axr = C.profile_panel(ax_a, y, 1 + chi.real, chi.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_c)$", xlim=(-Y, Y))
    ax_a.set_ylabel(r"$\epsilon'(y,\omega_c)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.fill_between(y, chi.imag, 0, where=chi.imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)
    axr.text(0.62, 0.08, 'gain', transform=axr.transAxes, color=C.C_EPS_IM, fontsize=6)
    er = D[f'{case}_eit_eps_real']
    axr = C.profile_panel(ax_b, y, er.real, er.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_r)$", xlim=(-Y, Y))
    ax_b.set_ylabel(r"$\epsilon'(y,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.set_ylim(bottom=-0.02)

    im = _map(ax_c, 'skk_map_lorentz', [], 0.05)
    _map(ax_d, 'skk_map_eit', [], 0.05)
    for ax in (ax_c, ax_d):
        ax.set_ylim(D['skk_map_y'][0], D['skk_map_y'][-1])
    ax_c.text(0.03, 0.95, 'single Lorentz', transform=ax_c.transAxes, color='w', fontsize=6, va='top')
    ax_d.text(0.03, 0.95, 'EIT', transform=ax_d.transAxes, color='w', fontsize=6, va='top')
    _cbar_right(fig, ax_d, im, r'$\log_{10}|r_R(\omega)|$')

    _td_panel(ax_e, case, 150, fdtd=True)
    ax_e.set_ylabel(r'$|\mathrm{out}(t)|\,/\,|E_{\rm inc}(t)|$')
    _td_panel(ax_f, 'b1_eta0.01', 800)
    ax_f.set_ylabel(r'$|\mathrm{out}(t)|\,/\,|E_{\rm inc}(t)|$')
    ax_e.text(0.03, 0.04, r'4th-order pole, $\eta=0.05$', transform=ax_e.transAxes, fontsize=6)
    ax_f.text(0.03, 0.04, r'random 6-pole target, $\eta=0.01$', transform=ax_f.transAxes, fontsize=6)
    for ax in (ax_e, ax_f):
        ax.text(0.97, 0.73, r'$|r_R|$', transform=ax.transAxes, color=C.C_R, fontsize=6, ha='right')
        ax.text(0.97, 0.58, r'$|\tau|$', transform=ax.transAxes, color=C.C_T, fontsize=6, ha='right')
        ax.text(0.97, 0.12, r'$|r_L|$', transform=ax.transAxes, color=C.C_R, fontsize=6, ha='right')
    ax_e.plot([], [], color='k', lw=C.LW_MAIN, label='EIT')
    ax_e.plot([], [], color=C.C_R, ls=':', lw=C.LW_THIN, label=r'two-line ($r_L$)')
    ax_e.plot([], [], color=C_GRAY, lw=C.LW_THIN, label=r'single Lorentz ($r_L$)')
    ax_e.plot([], [], marker='o', mfc='none', mec='k', ms=3, ls='none', label='FDTD')
    ax_e.legend(loc='upper center', fontsize=5, ncol=2)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')])
    C.save(fig, os.path.join(OUT, 'B8_one_way_invisible_skk.png'))


def fig_angle():
    """B9: angle dependence at w_c (EIT realisation, s and p) + time-domain checks; EIT layers."""
    fig, axs, _ = C.panel_grid(1, 2, has_twin=True, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12)
    ax_a, ax_b = axs[0]
    th = np.array(META['thetas'])
    for pol, ls in (('s', '-'), ('p', '--')):
        a = D[f'ang_{pol}']
        ax_a.semilogy(th, a[:, 3], color=C.C_R, ls=ls, label=rf'$|r_L|^2$ ({pol})')
        ax_a.semilogy(th, a[:, 4], color=C_GRAY, ls=ls, label=rf'$|r_R|^2$ ({pol})')
    for key, b in META['oblique_td'].items():
        thd, pol = float(key[:-1]), key[-1]
        mk = 'o' if pol == 's' else 's'
        ax_a.plot([thd, thd], [b['td_rL'] ** 2, b['td_rR'] ** 2], marker=mk, mfc='none', mec='k', ms=4, ls='none')
    ax_a.plot([], [], marker='o', mfc='none', mec='k', ms=4, ls='none', label='time domain')
    ax_a.set_xlim(0, 90); ax_a.set_ylim(1e-14, 1e6)
    ax_a.set_xlabel('angle of incidence (deg)'); ax_a.set_ylabel(r'at $\omega_c$, EIT layers')
    ax_a.legend(loc='upper center', fontsize=5, ncol=3, columnspacing=0.8, handlelength=1.5)

    for case, ls in (('simple_eta0.05', '-'), ('simple_eta0.01', '--')):
        y = D[f'{case}_y']
        ax_b.plot(y, D[f'{case}_eit_w0'], color=C.C_EPS_RE, ls=ls)
    axr = ax_b.twinx()
    for case, ls in (('simple_eta0.05', '-'), ('simple_eta0.01', '--')):
        axr.plot(D[f'{case}_y'], D[f'{case}_eit_wp2'] / (2 * np.pi) ** 2, color=C_ORANGE, ls=ls)
    ax_b.set_xlabel(r'$y/\lambda_r$'); ax_b.set_ylabel(r'bright resonance $\omega_0/\omega_r$', color=C.C_EPS_RE)
    axr.set_ylabel(r'$\omega_p^2/\omega_r^2$', color=C_ORANGE)
    ax_b.tick_params(axis='y', labelcolor=C.C_EPS_RE); axr.tick_params(axis='y', labelcolor=C_ORANGE)
    ax_b.set_xlim(-META['simple']['Y'], META['simple']['Y'])
    ax_b.plot([], [], color='k', ls='-', label=r'$\eta=0.05$'); ax_b.plot([], [], color='k', ls='--', label=r'$\eta=0.01$')
    ax_b.legend(loc='upper left', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B9_angles_eit_layers.png'))


FIGS = {'slab': lambda: (fig_slab('eit'), fig_slab('fano2')), 'skk': fig_skk, 'angle': fig_angle,
        'slab_thick': lambda: (fig_slab_thick('eit'), fig_slab_thick('fano2')),
        'eps_map': lambda: (fig_eps_map('eit'), fig_eps_map('fano2'))}
if __name__ == '__main__':
    for nm in (sys.argv[1:] or list(FIGS)):
        FIGS[nm]()
