"""Figures B7-B9 (stable virtual gain from passive two-pole media). Reads data/stable_gain.npz
(exp_stable_gain.py) and data/fdtd_twopole.npz (fdtd_twopole.py); no physics computed here.
  python theory/cfreq_skk/fig_stable_gain.py              # all
  python theory/cfreq_skk/fig_stable_gain.py slab         # B7: before/after, homogeneous slab (B7_eit, B7_fano2)
  python theory/cfreq_skk/fig_stable_gain.py skk          # B8: layered one-way-invisible sKK
  python theory/cfreq_skk/fig_stable_gain.py skk_fano2    # B8 built from two-line layers (B8_fano2_skk)
  python theory/cfreq_skk/fig_stable_gain.py slab_noFDTD  # B7_fano2_slab without the FDTD circles
  python theory/cfreq_skk/fig_stable_gain.py angle        # B9: angles + EIT layer parameters
  python theory/cfreq_skk/fig_stable_gain.py slab_thick   # B7 1 lam vs 5 lam (B7_{eit,fano2}_thick)
  python theory/cfreq_skk/fig_stable_gain.py eps_map      # |eps(w)| vs |t(w)| maps (B7_{eit,fano2}_eps_map)
  python theory/cfreq_skk/fig_stable_gain.py pole_vs_L    # highest slab pole vs thickness (B7_pole_vs_thickness)
  python theory/cfreq_skk/fig_stable_gain.py longhi       # B10: Longhi 2016 bidirectional target (data/longhi_bidir.npz)
slab_thick / eps_map / pole_vs_L and the two-line B7 maps read data/slab_poles.npz (exp_slab_poles.py).
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
C_ORANGE = '#ff7f0e'
C_GRAY = '#7f7f7f'
ETA_SLAB = 0.05
MAP_LIM = (-1.0, 3.0)
FD_EVERY = 100                       # FDTD samples are 0.05 T_r apart -> a marker every 5 T_r


def _cbar_right(fig, ax, im, label, gap_in=0.08, w_in=0.08):
    """Colorbar in the reserved right margin, beside `ax`, without touching its 4:3 box. The
    triangle on top = values above the range (drawn white on the maps, 2026-11-16)."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + gap_in / W, pos.y0, w_in / W, pos.height])
    cb = fig.colorbar(im, cax=cax, extend='max'); cb.set_label(label); cb.outline.set_linewidth(0.6)
    return cb


MAGMA_OVER = plt.get_cmap('magma').copy(); MAGMA_OVER.set_over('w')


T_LIM = (-2.0, 6.0)


def _to_T(Z, y, L):
    """log10|t| map -> log10 T = 2 log10|t/t_free|, |t_free(w)| = e^{-Im(w) L} (c = 1, w = 2 pi y).
    Same poles as |t|; removes the vacuum propagation factor that otherwise dominates thick stacks
    far below the real axis (2026-11-16, presentation maps)."""
    return 2 * (Z + (2 * np.pi * np.asarray(y))[:, None] * L / np.log(10))


def _map(ax, key, poles_list, eta, ykey=None, L=None):
    ykey = ykey or ('map_y' if 'slab' in key else 'skk_map_y')
    Z, lim = (D[key], MAP_LIM) if L is None else (_to_T(D[key], D[ykey], L), T_LIM)
    im = ax.pcolormesh(D['map_x'] if 'slab' in key else D['skk_map_x'], D[ykey],
                       np.maximum(Z, lim[0]), cmap=MAGMA_OVER, vmin=lim[0],
                       vmax=lim[1], shading='auto', rasterized=True)
    ax.axhline(-eta, color='w', lw=C.LW_GUIDE, ls='--')
    ax.axhline(0, color='w', lw=C.LW_GUIDE, ls=':')
    ax.plot([1], [-eta], marker='*', color='c', ms=7, ls='none')
    ax.text(1.008, -eta + 0.004, r'$\omega_c$', color='c', fontsize=6)
    for poles, mk in poles_list:
        pp = poles[poles.real > 0]
        ax.plot(pp.real, pp.imag, marker=mk, color='w', ms=4, mew=0.9, ls='none')
    ax.set_xlabel(r"Re $\omega/\omega_r$"); ax.set_ylabel(r"Im $\omega/\omega_r$")
    return im


def fig_slab(stable='eit', fdtd=True, wide=False):
    """B7: same eps(w_c) = 1 - 0.3i in a homogeneous 1-lambda slab. Left column: single Lorentz
    (Long et al.); right column: one STABLE two-pole medium (`stable` = 'eit' or 'fano2').
    One figure per stable medium (split 2026-11-13: the shared map mixed two different slabs).
    c, d: log10|t(w)| of THAT slab over complex w (bright = slab poles); white markers = poles of
    the material eps(w) itself (the slab poles pile up around them, where |eps| -> inf, t -> 0).
    e, f: T(t) = |t(t)/t_free(t)|^2 = |tau(t)|^2, dotted = frequency-domain T(w_c) (same for all
    media). fdtd=False is the presentation version (2026-11-16): titles on a, b; no text boxes or
    legends. wide=True (2026-11-17, slide layout): the same six panels as 2 rows x 3 columns, one
    row per medium (eps, map, T(t)), lettered a-c / d-f across the rows."""
    pres = not fdtd
    title = {'eit': 'EIT', 'fano2': 'two-line'}[stable]
    if wide:   # rows = media; column gaps hold a twin axis / colourbar plus the next y-label
        fig, axs, _ = C.panel_grid(2, 3, width_in=7.5, margin_l=0.47,
                                   col_gap=0.16 + 2 * C.YLABEL_W_IN + C.CENTER_GAP_IN,
                                   margin_t=C.LAYOUT_PAD_IN + (0.16 if pres else 0))
        (ax_a, ax_c, ax_e), (ax_b, ax_d, ax_f) = axs
    else:
        fig, axs, _ = C.panel_grid(3, 2, has_twin=True, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12,
                                   margin_t=C.LAYOUT_PAD_IN + (0.16 if pres else 0))
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
        if pres:
            ax.set_title(txt[0].upper() + txt[1:] + r' oscillator, $\epsilon(\omega_c)=1-0.3i$', fontsize=7)
        else:
            ax.text(0.06, 0.92, txt, transform=ax.transAxes, fontsize=6, va='top')

    if stable == 'fano2':        # tall window: shows the broad pole (-0.25) and its FP chain too
        Lm = 1.0 if pres else None
        im = _pmap(ax_c, 'tallmap_t_lorentz', 'tall_x', 'tall_y', 'lorentz', L=Lm)
        _pmap(ax_d, 'tallmap_t_fano2', 'tall_x', 'tall_y', 'fano2', L=Lm)
    else:
        im = _map(ax_c, 'slab_map_lorentz', [(D['slab_poles_lorentz'], 'x')], ETA_SLAB)
        _map(ax_d, f'slab_map_{stable}', [(D[f'slab_poles_{stable}'], 'x')], ETA_SLAB)
        for ax in (ax_c, ax_d):
            ax.set_ylim(D['map_y'][0], D['map_y'][-1])
    for ax, txt in ((ax_c, 'single Lorentz'), (ax_d, title)):
        if not pres:
            ax.text(0.03, 0.06, txt + r' slab, $1\lambda_r$', transform=ax.transAxes, color='w', fontsize=6)
    for ax in ((ax_c, ax_d) if wide else (ax_d,)):
        _cbar_right(fig, ax, im, r'$\log_{10}T(\omega)$' if pres else r'$\log_{10}|t(\omega)|$')

    t = D['slab_t']; m = t > 1
    for ax, nm, txt in ((ax_e, 'lorentz', 'single Lorentz'), (ax_f, stable, title)):
        tc = META['slab'][nm]['tau_c']
        ax.semilogy(t[m], (D[f'slab_tau_{nm}'][m] * tc) ** 2, color=C.C_T, lw=C.LW_THIN, label=txt)
        if fdtd:
            ft, fr = F[f'slab_{nm}_t'], F[f'slab_{nm}_tau']
            k = (ft > 2) & (ft < 70)
            ax.plot(ft[k][::FD_EVERY], (fr[k][::FD_EVERY] * tc) ** 2, marker='o', mfc='none', mec='k', ms=3,
                    mew=0.6, ls='none', label='FDTD')
        ax.axhline(tc ** 2, color=C.C_T, lw=C.LW_GUIDE, ls=':')
        ax.set_xlim(0, 100); ax.set_ylim(1e-3, 1e6)
        ax.set_xlabel(r'time after turn-on ($T_r$)')
        ax.set_ylabel(r'$T(t)$')
        if not pres:
            ax.legend(loc='upper left', fontsize=5)
    order = (ax_a, ax_c, ax_e, ax_b, ax_d, ax_f) if wide else (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)
    C.label_panels(fig, list(zip(order, 'abcdef')))
    name = f'B7_{stable}_slab' + ('' if fdtd else '_noFDTD') + ('_wide' if wide else '')
    C.save(fig, os.path.join(OUT, name + '.png'))


def _pmap(ax, key, xkey, ykey, nm, lim=MAP_LIM, L=None):
    """Like _map, for data/slab_poles.npz maps (material-pole x markers of medium nm)."""
    Z = P[key]
    if L is not None:
        Z, lim = _to_T(Z, P[ykey], L), T_LIM
    im = ax.pcolormesh(P[xkey], P[ykey], np.maximum(Z, lim[0]), cmap=MAGMA_OVER, vmin=lim[0],
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
        im = _pmap(axm, f'map_{key}', 'map_x', PMETA[stable]['map_y'], stable)
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


def fig_pole_vs_L():
    """Highest slab pole (slowest-decaying Fabry-Perot mode) vs thickness of the homogeneous
    eps(w_c) = 1 - 0.3i slab; a EIT, b two-line. Exact poles (line + dots); envelope of the exact
    modulus condition L Im k(w) = ln|r(w)| (every pole lies on that curve); first-order closed
    form Im w_c + v_g [g - ln(1/|r|)/L]; drive line (dashed); highest material pole (dash-dot)."""
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.47)
    L = P['sweep_L']
    for ax, nm, title in ((axs[0][0], 'eit', 'EIT'), (axs[0][1], 'fano2', 'two-line')):
        sw = PMETA[nm]['sweep']
        ax.plot(L, P[f'sweep_top_{nm}'].imag, color=C.C_T, lw=C.LW_MAIN, marker='o', ms=1.6,
                label='highest slab pole (exact)')
        ax.plot(L, P[f'sweep_env_{nm}'], color=C_ORANGE, lw=C.LW_THIN,
                label=r'envelope: $L\,{\rm Im}\,k(\omega)=\ln|r(\omega)|$')
        ax.plot(L, P[f'sweep_lin_{nm}'], color=C_GRAY, lw=C.LW_THIN, ls=':',
                label=r'$-|\omega^{\prime\prime}|+v_g\,[g-\ln(1/|r|)/L]$')
        ax.axhline(-ETA_SLAB, color='k', lw=C.LW_GUIDE, ls='--')
        ax.text(9.8, -ETA_SLAB + 0.002, 'drive', fontsize=6, ha='right')
        ax.axhline(sw['top_material_pole'], color=C.C_EPS_IM, lw=C.LW_GUIDE, ls='-.',
                   label='highest material pole')
        ax.axvline(sw['L_th_first_order'], color=C_GRAY, lw=C.LW_GUIDE, ls=':')
        ax.text(sw['L_th_first_order'] + 0.15, -0.115, rf"$L_{{\rm th}}={sw['L_th_first_order']:.2f}\lambda_r$",
                fontsize=6, color=C_GRAY)
        ax.set_xlim(0, L[-1]); ax.set_ylim(-0.12, 0.005)
        ax.set_xlabel(r'slab thickness $L/\lambda_r$')
        ax.set_ylabel(r'Im $\omega/\omega_r$ of slowest mode')
        ax.text(0.04, 0.92, title, transform=ax.transAxes, fontsize=6, va='top')
    axs[0][0].legend(loc='lower right', bbox_to_anchor=(1.0, 0.11), fontsize=4.5)
    C.label_panels(fig, [(axs[0][0], 'a'), (axs[0][1], 'b')])
    C.save(fig, os.path.join(OUT, 'B7_pole_vs_thickness.png'))


def _td_panel(ax, case, xmax, fdtd=False, stable='eit'):
    """|out(t)|/|E_inc(t)| of the layered sKK: the `stable` medium's r_L, r_R, tau (thick), the
    single-Lorentz r_L (gray, unshifted synthesis, cut where the drive drops below 1e-12) and,
    on the EIT version only, the two-line r_L dotted. Restored from 6e430af (lost in 7953ee4)."""
    t = D[f'{case}_t']; m = (t > 1) & (t <= xmax)
    M = META['skk'][case]
    for key, col, ls in (('rL', C.C_R, '-'), ('rR', C.C_R, '--'), ('tauL', C.C_T, '-')):
        ax.semilogy(t[m], D[f'{case}_{stable}_{key}'][m], color=col, ls=ls, lw=C.LW_MAIN)
        ref = M[stable]['rL' if key == 'rL' else 'rR' if key == 'rR' else 'tau']
        ax.axhline(ref, color=col, ls=':', lw=C.LW_GUIDE)
    if stable == 'eit':
        ax.semilogy(t[m], D[f'{case}_fano2_rL'][m], color=C.C_R, ls=':', lw=C.LW_THIN)
    eta = float(case.split('eta')[1])
    ml = m & (t < np.log(1e12) / (2 * np.pi * eta))   # unshifted synthesis: drive > 1e-12 of its peak
    ax.semilogy(t[ml], D[f'{case}_lorentz_rL'][ml], color=C_GRAY, ls='-', lw=C.LW_THIN)
    if fdtd:
        for nm in ('eit', 'lorentz'):
            ft, fr = F[f'{nm}_t_rL'], np.abs(F[f'{nm}_rL'])
            k = (ft > 2) & (ft < 66)
            ax.plot(ft[k][::FD_EVERY], fr[k][::FD_EVERY], marker='o', mfc='none', mec='k', ms=3, mew=0.6, ls='none')
        ft, fr = F['eit_t_tau'], np.abs(F['eit_tau'])
        k = (ft > 2) & (ft < 62)
        ax.plot(ft[k][::FD_EVERY], fr[k][::FD_EVERY], marker='o', mfc='none', mec='k', ms=3, mew=0.6, ls='none')
    ax.set_xlim(0, xmax); ax.set_ylim(1e-6, 1e3)
    ax.set_xlabel(r'time after turn-on ($T_r$)')


def fig_skk(stable='eit'):
    """B8: layered one-way-invisible sKK built from passive EIT layers (`stable`='eit', with FDTD
    circles) or two-line layers ('fano2' -> B8_fano2_skk.png: no FDTD, maps down to -0.3i so the
    broad two-line poles at -0.25 are in view)."""
    title = {'eit': 'EIT', 'fano2': 'two-line'}[stable]
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
    er = D[f'{case}_{stable}_eps_real']
    axr = C.profile_panel(ax_b, y, er.real, er.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_r)$", xlim=(-Y, Y))
    ax_b.set_ylabel(r"$\epsilon'(y,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.set_ylim(bottom=-0.02)

    pre, yk = ('skk_map', 'skk_map_y') if stable == 'eit' else ('skk_tallmap', 'skk_tall_y')
    im = _map(ax_c, f'{pre}_lorentz', [], 0.05, ykey=yk)
    _map(ax_d, f'{pre}_{stable}', [], 0.05, ykey=yk)
    for ax in (ax_c, ax_d):
        ax.set_ylim(D[yk][0], D[yk][-1])
    ax_c.text(0.03, 0.95, 'single Lorentz', transform=ax_c.transAxes, color='w', fontsize=6, va='top')
    ax_d.text(0.03, 0.95, title, transform=ax_d.transAxes, color='w', fontsize=6, va='top')
    _cbar_right(fig, ax_d, im, r'$\log_{10}|r_R(\omega)|$')

    _td_panel(ax_e, case, 150, fdtd=(stable == 'eit'), stable=stable)
    ax_e.set_ylabel(r'$|\mathrm{out}(t)|\,/\,|E_{\rm inc}(t)|$')
    _td_panel(ax_f, 'b1_eta0.01', 800, stable=stable)
    ax_f.set_ylabel(r'$|\mathrm{out}(t)|\,/\,|E_{\rm inc}(t)|$')
    ax_e.text(0.03, 0.04, r'4th-order pole, $\eta=0.05$', transform=ax_e.transAxes, fontsize=6)
    ax_f.text(0.03, 0.04, r'random 6-pole target, $\eta=0.01$', transform=ax_f.transAxes, fontsize=6)
    for ax in (ax_e, ax_f):
        ax.text(0.97, 0.73, r'$|r_R|$', transform=ax.transAxes, color=C.C_R, fontsize=6, ha='right')
        ax.text(0.97, 0.58, r'$|\tau|$', transform=ax.transAxes, color=C.C_T, fontsize=6, ha='right')
        ax.text(0.97, 0.12, r'$|r_L|$', transform=ax.transAxes, color=C.C_R, fontsize=6, ha='right')
    ax_e.plot([], [], color='k', lw=C.LW_MAIN, label=title)
    if stable == 'eit':
        ax_e.plot([], [], color=C.C_R, ls=':', lw=C.LW_THIN, label=r'two-line ($r_L$)')
    ax_e.plot([], [], color=C_GRAY, lw=C.LW_THIN, label=r'single Lorentz ($r_L$)')
    if stable == 'eit':
        ax_e.plot([], [], marker='o', mfc='none', mec='k', ms=3, ls='none', label='FDTD')
    ax_e.legend(loc='upper center', fontsize=5, ncol=2)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')])
    C.save(fig, os.path.join(OUT, 'B8_one_way_invisible_skk.png' if stable == 'eit' else 'B8_fano2_skk.png'))


B8_MAP_LIM = (-1.0, 2.0)            # 2026-11-16: narrower than T_LIM so the stack's features show


def _b8_profiles(ax_a, ax_d):
    """a: target eps(y, w_c) (every realisation hits it); d: the two-line layers at real w_r."""
    case = 'simple_eta0.05'
    y = D[f'{case}_y']
    yf = np.linspace(-1, 1, 2001); s = META['simple']
    tf = s['chimax'] * s['delta'] ** 4 / (yf + 1j * s['delta']) ** 4
    axr = C.profile_panel(ax_a, yf, 1 + tf.real, tf.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_c)$", xlim=(-1, 1))
    ax_a.set_ylabel(r"$\epsilon'(y,\omega_c)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.fill_between(yf, tf.imag, 0, where=tf.imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)
    er = D[f'{case}_fano2_eps_real']
    axr = C.profile_panel(ax_d, y, er.real, er.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_r)$", xlim=(-1, 1))
    ax_d.set_ylabel(r"$\epsilon'(y,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.set_ylim(bottom=-0.02)
    ax_a.set_title(r'target at $\omega_c$', fontsize=7)
    ax_d.set_title(r'two-line layers at real $\omega_r$', fontsize=7)


def _b8_lorentz_zoom():
    """Fine log10|t| map of the single-Lorentz stack around w_c (Re 0.95-1.05, Im -0.09-0.005), for
    B8 b; the stored tall map is too coarse there. Same design call as exp_stable_gain.py; cached."""
    fn = os.path.join(_HERE, 'data', 'b8_lorentz_zoom.npz')
    if os.path.exists(fn):
        Q = np.load(fn); return Q['x'], Q['y'], Q['Z']
    sys.path.insert(0, _HERE)
    import cfreq_lib as cf
    case = 'simple_eta0.05'
    y, chi = D[f'{case}_y'], D[f'{case}_chi']
    d = np.full(y.size, 2 * META['simple']['Y'] / y.size)
    wc = cf.omega_c(0.05); a = wc.real
    p = cf.lorentz_design(chi, wc, 0.5 * cf.kappa_max(chi, wc)); assert p['ok']
    x = np.linspace(0.95, 1.05, 301); yy = np.linspace(-0.09, 0.005, 286)
    W = (a * (x[None, :] + 1j * yy[:, None])).ravel()
    Z = np.log10(np.abs(cf.rt_model(cf.eps_lorentz, p, d, W)['tL'])).reshape(yy.size, x.size)
    np.savez(fn, x=x, y=yy, Z=Z)
    return x, yy, Z


def fig_skk_twoline():
    """B8 (two-line, slide layout 2x3, 2026-11-16): the B1s 4th-order-pole target at eta = 0.05.
    rows: single-Lorentz layers (top), two-line layers (bottom).
    a: target eps(y, w_c); d: the two-line layers at real w_r (passive).
    b, e: log10 T(w) = log10|t/t_free|^2 over complex w of each stack (colour range -1..2).
    c, f: R_L, R_R, T(t) = |t/t_free|^2 in the time domain, dotted = frequency-domain values.
    Time axes stop at t = 88, where the single-Lorentz drive has fallen to 1e-12 of its peak
    (beyond that its unshifted synthesis is noise)."""
    gap = 0.08 + 0.08 + 2 * C.YLABEL_W_IN + C.CENTER_GAP_IN
    fig, axs, _ = C.panel_grid(2, 3, has_twin=True, width_in=7.5, margin_l=0.56, col_gap=gap,
                               margin_t=C.LAYOUT_PAD_IN + 0.16,
                               row_gap=C.XLABEL_H_IN + C.GAP_WS_IN + 0.16)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axs
    case = 'simple_eta0.05'
    _b8_profiles(ax_a, ax_d)

    Lst = 2 * META['simple']['Y']
    for ax in (ax_b, ax_e):
        nm = 'lorentz' if ax is ax_b else 'fano2'
        if nm == 'lorentz':
            xm, ym, Zl = _b8_lorentz_zoom()
        else:
            xm, ym, Zl = D['skk_map_x'], D['skk_tall_y'], D[f'skk_tallmap_t_{nm}']
        Z = _to_T(Zl, ym, Lst)
        im = ax.pcolormesh(xm, ym, np.maximum(Z, B8_MAP_LIM[0]),
                           cmap=MAGMA_OVER, vmin=B8_MAP_LIM[0], vmax=B8_MAP_LIM[1],
                           shading='auto', rasterized=True)
        ax.axhline(-0.05, color='w', lw=C.LW_GUIDE, ls='--')
        ax.axhline(0, color='w', lw=C.LW_GUIDE, ls=':')
        ax.plot([1], [-0.05], marker='*', color='c', ms=7, ls='none', zorder=5)
        ax.text(1.008, -0.05 + 0.004, r'$\omega_c$', color='c', fontsize=6)
        ax.set_xlabel(r"Re $\omega/\omega_r$"); ax.set_ylabel(r"Im $\omega/\omega_r$")
        ax.set_ylim(ym[0], ym[-1])
        _cbar_right(fig, ax, im, r'$\log_{10}T(\omega)$')
    # b zoomed onto the single-Lorentz stack's ring of features around w_c (its layer poles; half
    # lie above the drive line); at e's window that ring is a dot
    ax_b.set_title('single Lorentz layers', fontsize=7); ax_e.set_title('two-line layers', fontsize=7)

    t = D[f'{case}_t']; M = META['skk'][case]
    for ax, nm in ((ax_c, 'lorentz'), (ax_f, 'fano2')):
        m = (t > 1) & (t < np.log(1e12) / (2 * np.pi * 0.05))
        for key, ref, col, ls, lab in (('rL', 'rL', C.C_R, '-', r'$R_L$'), ('rR', 'rR', C.C_R, '--', r'$R_R$'),
                                       ('tauL', 'tau', C.C_T, '-', r'$T$')):
            ax.semilogy(t[m], D[f'{case}_{nm}_{key}'][m] ** 2, color=col, ls=ls, lw=C.LW_MAIN, label=lab)
            ax.axhline(M[nm][ref] ** 2, color=col, ls=':', lw=C.LW_GUIDE)
        ax.set_xlim(0, 88); ax.set_ylim(1e-9, 1e3)
        ax.set_ylabel(r'$R_L,\ R_R,\ T$')
        ax.legend(loc='center right', fontsize=5)
        i = np.searchsorted(t, 85.0)
        print(f'B8 two-line fig, {nm}: at t=85 R_L={D[f"{case}_{nm}_rL"][i]**2:.3g} '
              f'R_R={D[f"{case}_{nm}_rR"][i]**2:.3g} T={D[f"{case}_{nm}_tauL"][i]**2:.4g}; predicted '
              f'R_L={M[nm]["rL"]**2:.3g} R_R={M[nm]["rR"]**2:.3g} T={M[nm]["tau"]**2:.4f}')
    ax_c.set_title('single Lorentz layers', fontsize=7); ax_f.set_title('two-line layers', fontsize=7)
    ax_f.set_xlabel(r'time after turn-on ($T_r$)'); ax_c.set_xlabel(r'time after turn-on ($T_r$)')
    C.label_panels(fig, [(ax, l) for ax, l in zip(axs.ravel(), 'abcdef')], dx_in=0.54)
    C.save(fig, os.path.join(OUT, 'B8_fano2_skk.png'))


def fig_skk_twoline_params():
    """B8 backup (2x2): a, b = B8 a, d; c, d = per-layer parameters of the two lines,
    eps = 1 + wp1^2/(w0^2 - w^2 - i g1 w) - wp2^2/(w0^2 - w^2 - i g2 w), common w0,
    g1/2|w''| = 5 (broad), g2/2|w''| = 2 (narrow), wp2^2 = 0.9 (g2/g1) wp1^2 (cfreq_lib.fano2_design;
    recomputed here from the stored chi(y), same call as exp_stable_gain.py)."""
    sys.path.insert(0, _HERE)
    import cfreq_lib as cf
    case = 'simple_eta0.05'
    y, chi = D[f'{case}_y'], D[f'{case}_chi']
    wc = cf.omega_c(0.05); a, b = wc.real, abs(wc.imag)
    p = cf.fano2_design(chi, wc)
    fig, axs, _ = C.panel_grid(2, 2, has_twin=True, margin_l=0.47, margin_r=C.YLABEL_W_IN + C.LAYOUT_PAD_IN,
                               margin_t=C.LAYOUT_PAD_IN + 0.16,
                               row_gap=C.XLABEL_H_IN + C.GAP_WS_IN + 0.16)
    (ax_a, ax_b), (ax_c, ax_d) = axs
    _b8_profiles(ax_a, ax_b)
    act = np.abs(chi) > 1e-3                    # vacuum-like tail layers carry no oscillator
    for ax, g, f, ttl in ((ax_c, p['g1'], p['f1'], r'broad line ($+$): $\gamma_1/2|\omega^{\prime\prime}|=%g$'),
                          (ax_d, p['g2'], p['f2'], r'narrow line ($-$): $\gamma_2/2|\omega^{\prime\prime}|=%g$')):
        ax.plot(y[act], p['w0'][act] / a, color=C.C_EPS_RE, label=r'$\omega_0/\omega_r$')
        ax.set_ylim(0.8, 1.2); ax.set_xlim(-1, 1)
        ax.set_ylabel(r'$\omega_0/\omega_r$', color=C.C_EPS_RE)
        ax.tick_params(axis='y', labelcolor=C.C_EPS_RE)
        ax2 = ax.twinx(); ax._skk_twin = [ax2]
        ax2.plot(y[act], np.sqrt(f[act]) / a, color=C_ORANGE)
        ax2.set_ylabel(r'$\omega_p/\omega_r$', color=C_ORANGE)
        ax2.tick_params(axis='y', labelcolor=C_ORANGE); ax2.set_ylim(0, 0.7)
        ax.set_xlabel(r'$y/\lambda_r$')
        ax.set_title(ttl % round(g / (2 * b), 6), fontsize=7)
    print(f'B8 params: w0/w_r in [{(p["w0"][act]/a).min():.3f}, {(p["w0"][act]/a).max():.3f}], '
          f'wp1/w_r max {np.sqrt(p["f1"]).max()/a:.3f}, wp2/w_r max {np.sqrt(p["f2"]).max()/a:.3f}')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')])
    C.save(fig, os.path.join(OUT, 'B8b_fano2_params.png'))


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


def fig_longhi():
    """B10 (2026-11-19, same 2x3 layout as B8): Longhi's BIDIRECTIONAL-invisibility target (Opt. Lett.
    2016, Eq. 13, Theta = 3 k0) at eta = 0.05, from data/longhi_bidir.npz (exp_longhi_bidir.py).
    rows: single-Lorentz layers (top), two-line layers (bottom).
    a: target eps(x, w_c) (central part; the stack runs over |x| <= 4.77 lam_r);
    d: the two-line layers at real w_r (passive).
    b, e: log10 T(w) over complex w of each stack (b zoomed on w_c, as B8 b).
    c, f: R_L, R_R, T(t) in the time domain; dotted = frequency-domain values at w_c."""
    Q = np.load(os.path.join(_HERE, 'data', 'longhi_bidir.npz')); QM = json.loads(str(Q['meta']))
    gap = 0.08 + 0.08 + 2 * C.YLABEL_W_IN + C.CENTER_GAP_IN
    fig, axs, _ = C.panel_grid(2, 3, has_twin=True, width_in=7.5, margin_l=0.56, col_gap=gap,
                               margin_t=C.LAYOUT_PAD_IN + 0.16,
                               row_gap=C.XLABEL_H_IN + C.GAP_WS_IN + 0.16)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axs
    XL = 1.5
    xf, tf = Q['xf'], Q['chif']
    axr = C.profile_panel(ax_a, xf, 1 + tf.real, tf.imag, xlabel=r'$x/\lambda_r$',
                          ylab_im=r"$\epsilon''(x,\omega_c)$", xlim=(-XL, XL))
    ax_a.set_ylabel(r"$\epsilon'(x,\omega_c)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.fill_between(xf, tf.imag, 0, where=tf.imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)
    er = Q['fano2_eps_real']
    axr = C.profile_panel(ax_d, Q['x'], er.real, er.imag, xlabel=r'$x/\lambda_r$',
                          ylab_im=r"$\epsilon''(x,\omega_r)$", xlim=(-XL, XL))
    ax_d.set_ylabel(r"$\epsilon'(x,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.set_ylim(bottom=-0.02)
    ax_a.set_title(r'target at $\omega_c$ (Longhi 2016)', fontsize=7)
    ax_d.set_title(r'two-line layers at real $\omega_r$', fontsize=7)

    L = float(Q['L'])
    for ax, (xk, yk, zk) in ((ax_b, ('zoom_x', 'zoom_y', 'zoom_t_lorentz')), (ax_e, ('map_x', 'map_y', 'map_t_fano2'))):
        xm, ym = Q[xk], Q[yk]
        Z = _to_T(Q[zk], ym, L)
        im = ax.pcolormesh(xm, ym, np.maximum(Z, B8_MAP_LIM[0]), cmap=MAGMA_OVER, vmin=B8_MAP_LIM[0],
                           vmax=B8_MAP_LIM[1], shading='auto', rasterized=True)
        ax.axhline(-0.05, color='w', lw=C.LW_GUIDE, ls='--')
        ax.axhline(0, color='w', lw=C.LW_GUIDE, ls=':')
        ax.plot([1], [-0.05], marker='*', color='c', ms=7, ls='none', zorder=5)
        ax.text(1.008 if ax is ax_e else 1.002, -0.05 + 0.004, r'$\omega_c$', color='c', fontsize=6)
        ax.set_xlabel(r"Re $\omega/\omega_r$"); ax.set_ylabel(r"Im $\omega/\omega_r$")
        ax.set_ylim(ym[0], ym[-1])
        _cbar_right(fig, ax, im, r'$\log_{10}T(\omega)$')
    ax_b.set_title('single Lorentz layers', fontsize=7); ax_e.set_title('two-line layers', fontsize=7)

    t = Q['t']
    m = (t > 1) & (t < np.log(1e12) / (2 * np.pi * 0.05))
    for ax, nm in ((ax_c, 'lorentz'), (ax_f, 'fano2')):
        M = QM['media'][nm]
        for key, ref, col, ls, lab in (('rL', 'RL', C.C_R, '-', r'$R_L$'), ('rR', 'RR', C.C_R, '--', r'$R_R$'),
                                       ('tauL', 'T', C.C_T, '-', r'$T$')):
            ax.semilogy(t[m], Q[f'{nm}_{key}'][m] ** 2, color=col, ls=ls, lw=C.LW_MAIN, label=lab)
            ax.axhline(M[ref], color=col, ls=':', lw=C.LW_GUIDE)
        ax.set_xlim(0, 88); ax.set_ylim(1e-10, 1e4)
        ax.set_ylabel(r'$R_L,\ R_R,\ T$')
        ax.set_xlabel(r'time after turn-on ($T_r$)')
        ax.legend(loc='center right', fontsize=5)
        print(f'B10 {nm}: at t=85 ' + ' '.join(f'{k}={v:.3g}' for k, v in M['td_t85'].items())
              + f'; predicted R_L={M["RL"]:.3g} R_R={M["RR"]:.3g} T={M["T"]:.6f}; highest stack pole '
              f'{M["highest_stack_pole"]}')
    ax_c.set_title('single Lorentz layers', fontsize=7); ax_f.set_title('two-line layers', fontsize=7)
    C.label_panels(fig, [(ax, l) for ax, l in zip(axs.ravel(), 'abcdef')], dx_in=0.54)
    C.save(fig, os.path.join(OUT, 'B10_fano2_longhi_bidir.png'))


FIGS = {'slab': lambda: (fig_slab('eit'), fig_slab('fano2')), 'skk': fig_skk, 'angle': fig_angle,
        'skk_fano2': fig_skk_twoline, 'skk_fano2_params': fig_skk_twoline_params, 'slab_noFDTD': lambda: fig_slab('fano2', fdtd=False),
        'slab_wide': lambda: fig_slab('fano2', fdtd=False, wide=True),
        'slab_thick': lambda: (fig_slab_thick('eit'), fig_slab_thick('fano2')),
        'eps_map': lambda: (fig_eps_map('eit'), fig_eps_map('fano2')), 'pole_vs_L': fig_pole_vs_L,
        'longhi': fig_longhi}
if __name__ == '__main__':
    for nm in (sys.argv[1:] or list(FIGS)):
        FIGS[nm]()
