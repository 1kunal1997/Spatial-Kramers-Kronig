"""Idea B feasibility (2026-10-12): gain-requiring sKK profiles built from PASSIVE Lorentz
media under complex-frequency (decaying) excitation, after Long et al., LSA 15:48 (2026).

Three figures, each a standalone function (run all, or name them):
  python theory/cfreq_skk/fig_cfreq_feasibility.py                 # all
  python theory/cfreq_skk/fig_cfreq_feasibility.py freq time       # some
    freq   B1: at w_c the realised passive structure is one-way reflectionless with t = 1
    trunc  B2: truncation errors are amplified ~ e^{4|k''|L}; steep tails + smooth window fix it
    time   B3: time domain: virtual gain never settles (lossy-at-w_c designs do)
    grow   B4: growing drive + LOSSLESS media: stable one-way 'virtual absorption' 
Units: c = 1, lam_r = 1 (period T_r = 1), w_c = 2 pi (1 - i eta). Library: cfreq_lib.py.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PROJECT_ROOT); sys.path.insert(0, _HERE)
import numpy as np
import matplotlib.pyplot as plt
import theory.skk_fig_common as C
import cfreq_lib as cf

C.apply_style()
OUT = os.path.join(_HERE, 'figures')
C_ORANGE = '#ff7f0e'

# --- the gain-requiring target: disordered sum of 4th-order poles (KHP 2017 family) ---
ORDER, Y = 4, 2.5
POLES, AMPS = cf.random_khp(6, 1.0, (0.12, 0.3), 0.8, seed=1, order=ORDER)
ETA = 0.05                                     # decay ratio used for the realisation
N_LAYERS = 500                                 # realised structure: lam/100 layers


def target(y):
    return cf.pole_profile(y, POLES, AMPS, ORDER) * cf.cos2_window(y, Y)


def realised(eta=ETA, kfrac=0.5):
    y, chi, d = cf.discretize(target, -Y, Y, N_LAYERS)
    wc = cf.omega_c(eta)
    p = cf.lorentz_design(chi, wc, kfrac * cf.kappa_max(chi, wc))
    assert p['ok']
    return y, chi, d, wc, p


def passive_skk_control(eta=ETA):
    """Horsley Lorentzian absorber (Im chi >= 0): needs no virtual gain; gam = 1.5 * 2|b|."""
    f = lambda y: cf.lorentzian_profile(y, -0.1, 0.15) * cf.cos2_window(y, 2.0)
    y, chi, d = cf.discretize(f, -2.0, 2.0, 400)
    wc = cf.omega_c(eta)
    p = cf.lorentz_design_gamma(chi, wc, 1.5 * 2 * abs(wc.imag))
    assert p['ok']
    return y, chi, d, wc, p


def fig_freq():
    """B1: frequency-domain picture at w_c — it works."""
    y, chi, d, wc, p = realised()
    a, b = wc.real, abs(wc.imag)
    fig, axs, _ = C.panel_grid(2, 2, has_twin=True, margin_r=C.YLABEL_W_IN + C.LAYOUT_PAD_IN)
    (ax_a, ax_b), (ax_c, ax_d) = axs

    yf = np.linspace(-Y, Y, 4001)
    axr = C.profile_panel(ax_a, yf, 1 + target(yf).real, target(yf).imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_c)$", xlim=(-Y, Y))
    ax_a.set_ylabel(r"$\epsilon'(y,\omega_c)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.fill_between(yf, target(yf).imag, 0, where=target(yf).imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)
    axr.text(0.03, 0.06, 'gain', transform=axr.transAxes, color=C.C_EPS_IM, fontsize=6)

    er = cf.eps_lorentz([a], p)[0]
    axr = C.profile_panel(ax_b, y, er.real, er.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_r)$", xlim=(-Y, Y))
    ax_b.set_ylabel(r"$\epsilon'(y,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    print(f'B1 realised: min Im eps over real w in [0.05,5]a = '
          f'{cf.eps_lorentz(a*np.linspace(0.05,5,4000), p).imag.min():.2e}  (passive if > 0)')

    etas = np.linspace(0, 0.12, 49)
    yi, chii, di = cf.discretize(target, -Y, Y, 5000)
    o = cf.rt(1 + chii, di, cf.omega_c(etas))
    ax_c.semilogy(etas, np.abs(o['rL']) ** 2, color=C.C_R, label=r'$|r_L|^2$ (left)')
    ax_c.semilogy(etas, np.abs(o['rR']) ** 2, color=C.C_R, ls='--', label=r'$|r_R|^2$ (right)')
    ax_c.semilogy(etas, np.abs(o['tauL'] - 1), color=C.C_T, label=r'$|\tau-1|$')
    ax_c.axvline(ETA, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_c.set_xlabel(r'decay ratio $\eta = |\omega^{\prime\prime}|/\omega^{\prime}$')
    ax_c.set_ylabel(r'at $\omega_c$')
    ax_c.set_xlim(0, etas[-1]); ax_c.set_ylim(1e-10, 1e7); ax_c.legend(loc='upper left', ncol=3, fontsize=5)
    oR = cf.rt(cf.eps_lorentz([wc], p), d, [wc])
    print(f'B1 realised at eta={ETA}: |rL|^2={abs(oR["rL"][0])**2:.2e} |rR|^2={abs(oR["rR"][0])**2:.2e} '
          f'|tau-1|={abs(oR["tauL"][0]-1):.2e}')

    ax_d.plot(y, p['gam'] / (2 * b), color=C.C_EPS_IM, label=r'$\gamma/2|\omega^{\prime\prime}|$')
    ax_d.plot(y, np.sqrt(p['w02']) / a, color=C.C_EPS_RE, label=r'$\omega_0/\omega_r$')
    ax_d.plot(y, 10 * np.sqrt(p['wp2']) / a, color=C_ORANGE, label=r'$10\,\omega_p/\omega_r$')
    ax_d.axhline(1, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_d.set_xlabel(r'$y/\lambda_r$'); ax_d.set_ylabel('Lorentz parameters')
    ax_d.set_xlim(-Y, Y); ax_d.legend(loc='upper right')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')])
    C.save(fig, os.path.join(OUT, 'B1_frequency_domain.png'))


# Opening/group-update figure (2026-10-13): B1's layout on the SIMPLE target (one 4th-order KHP
# pole, as in B7/B8) built from Long-type single-Lorentz layers. Separate function because the
# target, the eta sweep (realised structure re-designed per eta) and the panel-d shading differ.
S_DL, S_CM, S_Y, S_N = 0.2, 0.5, 2.0, 400


def simple_target(y):
    return S_CM * S_DL ** 4 / (y + 1j * S_DL) ** 4 * cf.cos2_window(y, S_Y)


def fig_freq_simple():
    """B1s: single-Lorentz realisation of the simple unit-transmission sKK, frequency domain."""
    y, chi, d = cf.discretize(simple_target, -S_Y, S_Y, S_N)
    wc = cf.omega_c(ETA); a, b = wc.real, abs(wc.imag)
    p = cf.lorentz_design(chi, wc, 0.5 * cf.kappa_max(chi, wc)); assert p['ok']
    fig, axs, _ = C.panel_grid(2, 2, has_twin=True, margin_l=0.47,
                               margin_r=C.YLABEL_W_IN + C.LAYOUT_PAD_IN)
    (ax_a, ax_b), (ax_c, ax_d) = axs

    yf = np.linspace(-S_Y, S_Y, 4001); tf = simple_target(yf)
    axr = C.profile_panel(ax_a, yf, 1 + tf.real, tf.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_c)$", xlim=(-S_Y, S_Y))
    ax_a.set_ylabel(r"$\epsilon'(y,\omega_c)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.fill_between(yf, tf.imag, 0, where=tf.imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)
    axr.text(0.62, 0.08, 'gain', transform=axr.transAxes, color=C.C_EPS_IM, fontsize=6)

    er = cf.eps_lorentz([a], p)[0]
    axr = C.profile_panel(ax_b, y, er.real, er.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_r)$", xlim=(-S_Y, S_Y))
    ax_b.set_ylabel(r"$\epsilon'(y,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    wr = a * np.linspace(0.05, 5, 4000)
    print(f'B1s: min Im eps over real w in [0.05,5]w_r = {cf.eps_lorentz(wr, p).imag.min():.2e}')
    orr = cf.rt(er[None, :], d, [a])
    print(f'B1s at real w_r: |rL|^2={abs(orr["rL"][0])**2:.3g} |rR|^2={abs(orr["rR"][0])**2:.3g} '
          f'|tau|^2={abs(orr["tauL"][0])**2:.3g}')

    # realised single-Lorentz structure, re-designed at every eta, evaluated at its own w_c
    etas = np.linspace(0.005, 0.12, 47); res = []
    for e in etas:
        w = cf.omega_c(e); q = cf.lorentz_design(chi, w, 0.5 * cf.kappa_max(chi, w)); assert q['ok']
        o = cf.rt(cf.eps_lorentz([w], q), d, [w]); res.append([o['rL'][0], o['rR'][0], o['tauL'][0]])
    res = np.array(res)
    ax_c.semilogy(etas, np.abs(res[:, 0]) ** 2, color=C.C_R, label=r'$|r_L|^2$ (from left)')
    ax_c.semilogy(etas, np.abs(res[:, 1]) ** 2, color=C.C_R, ls='--', label=r'$|r_R|^2$ (from right)')
    ax_c.semilogy(etas, np.abs(res[:, 2] - 1), color=C.C_T, label=r'$|\tau-1|$')
    ax_c.axvline(ETA, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_c.set_xlabel(r'decay ratio $\eta = |\omega^{\prime\prime}|/\omega^{\prime}$')
    ax_c.set_ylabel(r'response at $\omega_c$')
    ax_c.set_xlim(0, etas[-1]); ax_c.set_ylim(1e-12, 1e4); ax_c.legend(loc='upper left', ncol=2, fontsize=5)
    i = np.argmin(abs(etas - ETA))
    print(f'B1s at eta={etas[i]:.3f}: |rL|^2={abs(res[i,0])**2:.2e} |rR|^2={abs(res[i,1])**2:.3g} '
          f'|tau-1|={abs(res[i,2]-1):.2e}')

    g = p['gam'] / (2 * b)
    ax_d.plot(y, g, color=C.C_EPS_IM, label=r'damping $\gamma/2|\omega^{\prime\prime}|$')
    ax_d.plot(y, np.sqrt(p['w02']) / a, color=C.C_EPS_RE, label=r'resonance $\omega_0/\omega_r$')
    ax_d.plot(y, 10 * np.sqrt(p['wp2']) / a, color=C_ORANGE, label=r'strength $10\,\omega_p/\omega_r$')
    ax_d.fill_between(y, g, 1, where=g < 1, color=C.C_EPS_IM, alpha=0.15, lw=0)
    ax_d.axhline(1, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_d.set_xlabel(r'$y/\lambda_r$'); ax_d.set_ylabel('Lorentz parameters')
    ax_d.set_xlim(-S_Y, S_Y); ax_d.legend(loc='upper left', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')])
    C.save(fig, os.path.join(OUT, 'B1s_single_lorentz_simple.png'))


def fig_eta_zero():
    """B1z: the eta -> 0 limit of B1s. Each eta gets its own (re-designed) passive Lorentz
    structure; at real w_r its profile converges to a fixed PASSIVE profile, the line narrows
    as ~|w''|, and only the shrinking band around w_r ever sees the target (gain-like) eps."""
    y, chi, d = cf.discretize(simple_target, -S_Y, S_Y, S_N)
    etas = [0.05, 0.01, 0.002]
    cols = plt.cm.viridis(np.linspace(0, 0.8, len(etas)))
    fig, axs, _ = C.panel_grid(3, 2, margin_l=0.47)
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axs
    j = np.argmin(chi.imag)                       # the most gain-like layer
    ax_a.plot(y, chi.imag, color='gray', ls='--', lw=C.LW_THIN, label=r'target, $\omega_c$')
    ax_b.plot(y, 1 + chi.real, color='gray', ls='--', lw=C.LW_THIN, label=r'target, $\omega_c$')
    for e, col, lw in zip(etas, cols, [2.6, 1.6, 0.8]):
        w = cf.omega_c(e); a, b = w.real, abs(w.imag)
        p = cf.lorentz_design(chi, w, 0.5 * cf.kappa_max(chi, w)); assert p['ok']
        er = cf.eps_lorentz([a], p)[0]
        ax_a.plot(y, er.imag, color=col, lw=lw, label=rf'$\omega_r$, $\eta={e}$')
        ax_b.plot(y, er.real, color=col, lw=lw, label=rf'$\omega_r$, $\eta={e}$')
        wr = a * np.linspace(0.8, 1.2, 8001)
        ax_c.plot(wr / a, cf.eps_lorentz(wr, p)[:, j].imag, color=col, lw=C.LW_MAIN, label=rf'$\eta={e}$')
        x = np.linspace(-12, 12, 4001)
        ax_d.plot(x, cf.eps_lorentz(a + b * x, p)[:, j].imag, color=col, lw=lw, label=rf'$\eta={e}$')
        o = cf.rt_lorentz(p, d, wr)
        ax_e.plot(wr / a, np.abs(o['tauL']) ** 2, color=col, lw=C.LW_MAIN, label=rf'$\eta={e}$')
        print(f'B1z eta={e}: min Im eps(y, w_r)={er.imag.min():.2e}, min over real w={cf.eps_lorentz(wr, p).imag.min():.1e}, '
              f'|tau(w_r)|^2={abs(cf.rt(er[None, :], d, [a])["tauL"][0])**2:.3f}')
    for ax in (ax_a, ax_c, ax_d):
        ax.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_a.fill_between(y, chi.imag, 0, where=chi.imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)
    ax_a.set_ylabel(r"$\epsilon''(y)$"); ax_b.set_ylabel(r"$\epsilon'(y)$")
    for ax in (ax_a, ax_b):
        ax.set_xlabel(r'$y/\lambda_r$'); ax.set_xlim(-S_Y, S_Y)
    ax_a.legend(loc='upper left', fontsize=5); ax_b.legend(loc='upper left', fontsize=5)
    ax_c.set_xlabel(r'real frequency $\omega/\omega_r$'); ax_c.set_ylabel(r"$\epsilon''(\omega)$, layer $y_j$")
    ax_c.set_xlim(0.8, 1.2); ax_c.legend(loc='upper left', fontsize=5)
    ax_d.set_xlabel(r"$(\omega-\omega_r)/|\omega''|$, real $\omega$"); ax_d.set_ylabel(r"$\epsilon''(\omega)$, layer $y_j$")
    ax_d.set_xlim(-12, 12); ax_d.legend(loc='upper left', fontsize=5)
    ax_e.axhline(1, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_e.set_xlabel(r'real frequency $\omega/\omega_r$'); ax_e.set_ylabel(r'$|\tau(\omega)|^2$')
    ax_e.set_xlim(0.8, 1.2); ax_e.legend(loc='lower left', fontsize=5)

    es = np.geomspace(1e-4, 0.1, 31); res = []
    for e in es:
        w = cf.omega_c(e); a = w.real
        p = cf.lorentz_design(chi, w, 0.5 * cf.kappa_max(chi, w)); assert p['ok']
        oc = cf.rt(cf.eps_lorentz([w], p), d, [w]); orr = cf.rt(cf.eps_lorentz([a], p), d, [a])
        res.append([abs(oc['tauL'][0] - 1), 1 - abs(orr['tauL'][0]) ** 2,
                    p['gam'].max() / a, np.sqrt(p['wp2'].max()) / a])
    res = np.array(res)
    ax_f.loglog(es, res[:, 0], color=C.C_T, label=r'$|\tau(\omega_c)-1|$')
    ax_f.loglog(es, res[:, 1], color=C.C_T, ls='--', label=r'$1-|\tau(\omega_r)|^2$')
    ax_f.loglog(es, res[:, 2], color=C.C_EPS_IM, label=r'linewidth $\gamma_{max}/\omega_r$')
    ax_f.loglog(es, res[:, 3], color=C_ORANGE, label=r'strength $\omega_{p,max}/\omega_r$')
    ax_f.set_xlabel(r'decay ratio $\eta$'); ax_f.set_ylabel('value'); ax_f.set_ylim(1e-8, 3)
    ax_f.legend(loc='lower right', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')])
    C.save(fig, os.path.join(OUT, 'B1z_eta_to_zero.png'))


def fig_trunc():
    """B2: complex frequency amplifies far-side truncation as ~e^{4|k''|L}."""
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.45, margin_b=0.33, col_gap=0.55)
    ax_a, ax_b = axs[0]
    dl = 0.15
    etas = [0, 0.01, 0.02, 0.05, 0.1]
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(etas)))
    Ys = np.geomspace(0.6, 12, 18)
    f2 = lambda yy: 0.8 * dl ** 2 / (yy + 1j * dl) ** 2
    RL = []
    for Yh in Ys:
        yy, chi, d = cf.discretize(f2, -Yh, Yh, int(2 * Yh / 0.004))
        RL.append(np.abs(cf.rt(1 + chi, d, cf.omega_c(np.array(etas)))['rL']) ** 2)
    RL = np.array(RL)
    for k, e in enumerate(etas):
        ax_a.loglog(2 * Ys, RL[:, k], color=cols[k], label=rf'$\eta={e}$')
    ax_a.set_xticks([1, 2, 5, 10, 20]); ax_a.set_xticklabels(['1', '2', '5', '10', '20'])
    ax_a.set_xlabel(r'slab length $L/\lambda_r$'); ax_a.set_ylabel(r'$|r_L(\omega_c)|^2$')
    ax_a.legend(loc='upper left', ncol=2)

    etas2 = np.linspace(0, 0.12, 25)
    for order, ls in [(2, ':'), (3, '--'), (4, '-')]:
        fo = lambda yy: 0.8 * dl ** order / (yy + 1j * dl) ** order * cf.cos2_window(yy, 2.0)
        yy, chi, d = cf.discretize(fo, -2.0, 2.0, 2000)
        o = cf.rt(1 + chi, d, cf.omega_c(etas2))
        ax_b.semilogy(etas2, np.abs(o['rL']) ** 2, color=C.C_R, ls=ls, label=f'pole order {order}')
    ax_b.set_xlabel(r'decay ratio $\eta$'); ax_b.set_ylabel(r'$|r_L(\omega_c)|^2$, $L=4\lambda_r$')
    ax_b.legend(loc='upper left')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B2_truncation_amplification.png'))


# --- time domain helpers ---
DT, NT = 1 / 32, 2 ** 16
T_ON, T0 = 10.0, 2.0


def _td(resp, eta, scale=1.0):
    """Complex output/drive ratio vs time (in drive periods) with noise-floor masking."""
    t = np.arange(NT) * DT
    dcx, out = cf.complex_response(resp, t, eta, T_ON, T0)
    ok = np.abs(dcx) > 1e-10 * np.abs(dcx).max()
    ratio = np.where(ok, out / np.where(ok, dcx, 1), np.nan)
    return (t - T0) * scale, ratio


def fig_time():
    """B3: the decisive time-domain test."""
    fig, axs, _ = C.panel_grid(2, 2, margin_r=0.5, col_gap=0.75)
    (ax_a, ax_b), (ax_c, ax_d) = axs
    eta = ETA; wc = cf.omega_c(eta); a, b = wc.real, abs(wc.imag)
    wr = cf.rfft_omega(NT, DT)

    # a: homogeneous 1-lambda slab, Long's one-parameter family
    Ls, nl = 1.0, 40; d = np.full(nl, Ls / nl)
    cases = [(1 - 0.3j, 0.5, C.C_EPS_IM, '-'), (1 - 0.3j, 0.95, C.C_EPS_IM, '--'),
             (1 + 0.3j, 1.5, C.C_EPS_RE, '-'), (1 + 0.3j, 3.0, C.C_EPS_RE, '--')]
    for tgt, g, col, ls in cases:
        p = cf.lorentz_design_gamma(np.array([tgt - 1]), wc, g * 2 * b)
        e = lambda w: np.repeat(cf.eps_lorentz(w, p), nl, axis=1)
        tau_c = cf.rt(e([wc]), d, [wc])['tauL'][0]
        o = cf.rt(e(wr), d, wr)
        t, rt_ = _td(o['tL'], eta); _, rf = _td(np.exp(1j * wr * Ls), eta)
        # transmitted / free-propagated, both at the exit face (both ratios share the drive
        # denominator, so it cancels); the exit face sees the drive Ls later
        sh = int(round(Ls / DT))
        tt = t[sh:] - Ls
        r = np.abs(rt_[sh:]) / np.abs(rf[sh:])           # |t/t_free| -> |tau(w_c)| if it settles
        lab = (rf'$\epsilon(\omega_c)={tgt.real:.0f}{tgt.imag:+.1f}i$, $\gamma/2|\omega^{{\prime\prime}}|={g}$')
        ax_a.semilogy(tt, r, color=col, ls=ls, lw=C.LW_THIN, label=lab)
        ax_a.axhline(abs(tau_c), color=col, lw=C.LW_GUIDE, ls=':')   # freq-domain |tau(w_c)|
    ax_a.set_xlim(0, 60); ax_a.set_ylim(1e-2, 1e3)
    ax_a.set_xlabel(r'time after turn-on ($T_r$)')
    ax_a.set_ylabel(r'$|t(t)/t_{\rm free}(t)|$')
    ax_a.legend(loc='upper left', fontsize=5)

    # b: complex-w map of |t| for the virtual-gain slab (gam/2|b| = 0.5): poles above w_c
    p = cf.lorentz_design_gamma(np.array([-0.3j]), wc, 0.5 * 2 * b)
    xr = np.linspace(0.9, 1.1, 241) * a; yi = np.linspace(-0.08, 0.004, 169) * a
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    o = cf.rt(np.repeat(cf.eps_lorentz(W, p), nl, axis=1), d, W)
    Z = np.log10(np.abs(o['tL'])).reshape(yi.size, xr.size)
    im = ax_b.pcolormesh(xr / a, yi / a, Z, cmap='magma', vmin=-1, vmax=3, shading='auto')
    ax_b.axhline(-eta, color='w', lw=C.LW_GUIDE, ls='--')
    ax_b.axhline(0, color='w', lw=C.LW_GUIDE, ls=':')
    ax_b.plot([1], [-eta], marker='*', color='c', ms=7)
    ax_b.text(1.005, -eta + 0.004, r'$\omega_c$', color='c', fontsize=6)
    ax_b.set_xlabel(r"Re $\omega/\omega_r$"); ax_b.set_ylabel(r"Im $\omega/\omega_r$")
    cb = fig.colorbar(im, ax=ax_b, pad=0.02, fraction=0.06); cb.set_label(r'$\log_{10}|t(\omega)|$')

    # c: sKK structures — virtual-gain KHP realisation vs passive-at-w_c control
    for (y, chi, dd, wc_, pp), col, name in [(realised(), C.C_EPS_IM, 'gain target'),
                                              (passive_skk_control(), C.C_EPS_RE, 'lossy target')]:
        o = cf.rt(cf.eps_lorentz(wr, pp), dd, wr)
        oc = cf.rt(cf.eps_lorentz([wc_], pp), dd, [wc_])
        for key, ls in [('rL', '-'), ('rR', '--')]:
            t, r = _td(o[key], eta)
            ax_c.semilogy(t, np.abs(r), color=col, ls=ls, lw=C.LW_THIN)
            ax_c.axhline(abs(oc[key][0]), color=col, ls=ls, lw=C.LW_GUIDE, alpha=0.6)
        print(f'B3c {name}: predicted |rL|={abs(oc["rL"][0]):.2e} |rR|={abs(oc["rR"][0]):.2e}')
    ax_c.plot([], [], color='k', ls='-', label='left incidence')
    ax_c.plot([], [], color='k', ls='--', label='right incidence')
    ax_c.plot([], [], color=C.C_EPS_IM, label='KHP (virtual gain)')
    ax_c.plot([], [], color=C.C_EPS_RE, label='Lorentzian (lossy)')
    ax_c.set_xlim(0, 60); ax_c.set_ylim(1e-6, 1e3)
    ax_c.set_xlabel(r'time after turn-on ($T_r$)'); ax_c.set_ylabel(r'$|r(t)|/|E_{\rm inc}(t)|$')
    ax_c.legend(loc='lower right', fontsize=5)

    # d: Long et al. 2026 perfect lens (Table 1), 1-D normal incidence
    s = cf.TWO_PI / 1.256; eta_L = 0.4 / 1.256; wcL = cf.omega_c(eta_L)
    M = {'e1': (3.873, 2.559, 2.233), 'm1': (1.483, 1.472, 0.917),
         'e2': (1.225, 1.197, 0.743), 'm2': (1.245, 1.230, 0.755)}
    P = {k: dict(wp2=np.array([(v[0] * s) ** 2]), w02=np.array([(v[1] * s) ** 2]),
                 gam=np.array([v[2] * s])) for k, v in M.items()}
    lam0, ns = 277.98 / 1500.0, 20
    dL = np.concatenate([np.full(ns, 0.5 * lam0 / ns), np.full(2 * ns, lam0 / (2 * ns)),
                         np.full(ns, 0.5 * lam0 / ns)])

    def em(w):
        e1, m1, e2, m2 = [cf.eps_lorentz(w, P[k]) for k in ['e1', 'm1', 'e2', 'm2']]
        cat = lambda u, v: np.concatenate([np.repeat(u, ns, 1), np.repeat(v, 2 * ns, 1),
                                           np.repeat(u, ns, 1)], 1)
        return cat(e1, e2), cat(m1, m2)
    E, U = em([wcL]); oc = cf.rt(E, dL, [wcL], mu=U)
    # eta_L = 0.32: the drive falls e^{-2} per period, so use the contour-shifted synthesis
    # (c = 0.28 w_r, just above the highest strip pole at Im w = -0.289 w_r) to follow the
    # ratio for 40 periods; it matches the c = 0 synthesis wherever the latter is resolved.
    t = np.arange(NT) * DT / 2
    c = 0.28 * cf.TWO_PI
    resp = lambda key: (lambda w: cf.rt(em(w)[0], dL, w, mu=em(w)[1])[key])
    dcx, tl = cf.complex_response_shifted(resp('tL'), t, eta_L, 5.0, T0, c)
    _, rl = cf.complex_response_shifted(resp('rL'), t, eta_L, 5.0, T0, c)
    tl, rl, t = tl / dcx, rl / dcx, t - T0
    ax_d.semilogy(t, np.abs(tl), color=C.C_T, label=r'$|t(t)|/|E_{\rm inc}(t)|$')
    ax_d.semilogy(t, np.abs(rl), color=C.C_R, label=r'$|r(t)|/|E_{\rm inc}(t)|$')
    ax_d.axhline(abs(oc['tL'][0]), color=C.C_T, ls='--', lw=C.LW_GUIDE)
    ax_d.axhline(abs(oc['rL'][0]), color=C.C_R, ls='--', lw=C.LW_GUIDE)
    print(f'B3d Long lens at w_c: |t|={abs(oc["tL"][0]):.4f} |r|={abs(oc["rL"][0]):.2e}')
    ax_d.set_xlim(0, 40); ax_d.set_ylim(1e-3, 1e3)
    ax_d.set_xlabel(r'time after turn-on ($T_r$)'); ax_d.set_ylabel('Long et al. lens')
    ax_d.legend(loc='upper left', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')])
    C.save(fig, os.path.join(OUT, 'B3_time_domain.png'))


def fig_grow():
    """B4: the stable flip side. A GROWING drive (eta < 0) puts every pole below w_c, so the
    steady state is always reached; LOSSLESS Lorentz media (gam -> 0) then realise a lossy
    (passive-type) sKK target: one-way reflectionless 'virtual absorption' with no loss at all."""
    eta = -0.02; wc = cf.omega_c(eta); a = wc.real
    f = lambda y: cf.lorentzian_profile(y, -0.1, 0.15) * cf.cos2_window(y, 2.0)
    y, chi, d = cf.discretize(f, -2.0, 2.0, 400)
    p = cf.lorentz_design_gamma(chi, wc, 1e-4 * a)       # ~lossless (gam = 1e-4 w_r)
    assert p['ok']
    fig, axs, _ = C.panel_grid(1, 2)
    ax_a, ax_b = axs[0]
    ww = a * np.linspace(0.6, 1.4, 1601)
    o = cf.rt(cf.eps_lorentz(ww, p), d, ww)
    ax_a.plot(ww / a, np.abs(o['rL']) ** 2, color=C.C_R, label=r'$|r_L|^2$')
    ax_a.plot(ww / a, np.abs(o['rR']) ** 2, color=C.C_R, ls='--', label=r'$|r_R|^2$')
    ax_a.plot(ww / a, np.abs(o['tL']) ** 2, color=C.C_T, label=r'$|t|^2$')
    ax_a.set_xlabel(r'real frequency $\omega/\omega_r$'); ax_a.set_ylabel('Power Fraction')
    ax_a.set_xlim(0.6, 1.4); ax_a.legend(loc='center right')
    oc = cf.rt(cf.eps_lorentz([wc], p), d, [wc])
    t = np.arange(2 ** 15) * DT
    fn = lambda key: (lambda w: cf.rt(cf.eps_lorentz(w, p), d, w)[key])
    c = cf.TWO_PI * (eta - 0.01)                          # contour moved UP: always allowed
    dcx, rl = cf.complex_response_shifted(fn('rL'), t, eta, T_ON, T0, c)
    _, rr = cf.complex_response_shifted(fn('rR'), t, eta, T_ON, T0, c)
    for r, key, ls in [(rl, 'rL', '-'), (rr, 'rR', '--')]:
        ax_b.semilogy(t - T0, np.abs(r / dcx), color=C.C_R, ls=ls, lw=C.LW_THIN,
                      label=('left' if key == 'rL' else 'right') + ' incidence')
        ax_b.axhline(abs(oc[key][0]), color='gray', ls=ls, lw=C.LW_GUIDE)
    print(f'B4 growing eta={eta}: predicted |rL|={abs(oc["rL"][0]):.2e} |rR|={abs(oc["rR"][0]):.2e} '
          f'|tau|={abs(oc["tauL"][0]):.3f}; real-w min/max |rL|^2 = {np.abs(o["rL"]).min()**2:.2f}/'
          f'{np.abs(o["rL"]).max()**2:.2f}, max |1-R-T| = {np.abs(1-np.abs(o["rL"])**2-np.abs(o["tL"])**2).max():.1e}')
    ax_b.set_xlim(0, 60); ax_b.set_ylim(1e-4, 1)
    ax_b.set_xlabel(r'time after turn-on ($T_r$)'); ax_b.set_ylabel(r'$|r(t)|/|E_{\rm inc}(t)|$')
    ax_b.legend(loc='lower right')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B4_growing_drive_lossless.png'))


FIGS = {'freq': fig_freq, 'freq_simple': fig_freq_simple, 'eta_zero': fig_eta_zero, 'trunc': fig_trunc, 'time': fig_time, 'grow': fig_grow}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for nm in names:
        FIGS[nm]()
