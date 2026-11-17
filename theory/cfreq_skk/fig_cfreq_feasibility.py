"""Idea B feasibility (2026-10-12): gain-requiring sKK profiles built from PASSIVE Lorentz
media under complex-frequency (decaying) excitation, after Long et al., LSA 15:48 (2026).

Three figures, each a standalone function (run all, or name them):
  python theory/cfreq_skk/fig_cfreq_feasibility.py                 # all
  python theory/cfreq_skk/fig_cfreq_feasibility.py freq time       # some
    freq   B1: at w_c the realised passive structure is one-way reflectionless with t = 1
    trunc  B2: truncation errors are amplified ~ e^{4|k''|L}; steep tails + smooth window fix it
    time   B3: time domain: virtual gain never settles (lossy-at-w_c designs do)
    single_layer  B0: one gain Lorentz layer, Im eps over complex w (sign flips at Im w = -gam/2)
    slab_td       B3s: B3a's loss / gain slabs as |t(w)| maps + time domain with COMSOL circles
    lens          B3L: Long et al.'s lens, |t(w)| map with its material poles + time domain
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
                          ylab_im=r"$\epsilon''(y,\omega_c)$", xlim=(-1, 1))
    ax_a.set_ylabel(r"$\epsilon'(y,\omega_c)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    axr.fill_between(yf, tf.imag, 0, where=tf.imag < 0, color=C.C_EPS_IM, alpha=0.15, lw=0)

    er = cf.eps_lorentz([a], p)[0]
    axr = C.profile_panel(ax_b, y, er.real, er.imag, xlabel=r'$y/\lambda_r$',
                          ylab_im=r"$\epsilon''(y,\omega_r)$", xlim=(-1, 1))
    ax_b.set_ylabel(r"$\epsilon'(y,\omega_r)$", color=C.C_EPS_RE)
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')
    wr = a * np.linspace(0.05, 5, 4000)
    print(f'B1s: min Im eps over real w in [0.05,5]w_r = {cf.eps_lorentz(wr, p).imag.min():.2e}')
    orr = cf.rt(er[None, :], d, [a])
    print(f'B1s at real w_r: |rL|^2={abs(orr["rL"][0])**2:.3g} |rR|^2={abs(orr["rR"][0])**2:.3g} '
          f'|tau|^2={abs(orr["tauL"][0])**2:.3g}')

    # realised single-Lorentz structure, re-designed at every eta, evaluated at its own w_c
    # 2026-11-16: R = |r|^2, T = |tau|^2 = |t/t_free|^2 (T = 1 <=> same as vacuum); eta from 0.001
    etas = np.linspace(0.001, 0.12, 120); res = []
    for e in etas:
        w = cf.omega_c(e); q = cf.lorentz_design(chi, w, 0.5 * cf.kappa_max(chi, w)); assert q['ok']
        o = cf.rt(cf.eps_lorentz([w], q), d, [w]); res.append([o['rL'][0], o['rR'][0], o['tauL'][0]])
    res = np.abs(np.array(res)) ** 2
    ax_c.semilogy(etas, res[:, 0], color=C.C_R, label=r'$R_L$')
    ax_c.semilogy(etas, res[:, 1], color=C.C_R, ls='--', label=r'$R_R$')
    ax_c.semilogy(etas, res[:, 2], color=C.C_T, label=r'$T$')
    ax_c.axvline(ETA, color='gray', lw=C.LW_GUIDE, ls=':')
    ax_c.set_xlabel(r'decay ratio $\eta = |\omega^{\prime\prime}|/\omega^{\prime}$')
    ax_c.set_ylabel(r'response at $\omega_c$')
    ax_c.set_xlim(0, 0.12); ax_c.set_ylim(1e-12, 1e4); ax_c.legend(loc='center right', fontsize=5)
    i = np.argmin(abs(etas - ETA))
    print(f'B1s at eta={etas[i]:.3f}: R_L={res[i,0]:.2e} R_R={res[i,1]:.3g} T={res[i,2]:.6f}')

    # d: the pole of layer j sits at ~ w0 - i gam/2 and the drive at w_r - i|w''|, so gam is read
    # against 2|w''| (< 1: pole above the drive line = gain at w_c) and w0 against w_r; w_p is the
    # strength (no partner on the drive), in units of w_r on the right axis
    g = p['gam'] / (2 * b)
    ax_d.plot(y, g, color=C.C_EPS_IM, label=r'$\gamma/2|\omega^{\prime\prime}|$')
    ax_d.plot(y, np.sqrt(p['w02']) / a, color=C.C_EPS_RE, label=r'$\omega_0/\omega_r$')
    ax_d.fill_between(y, g, 1, where=g < 1, color=C.C_EPS_IM, alpha=0.15, lw=0)
    axd2 = ax_d.twinx(); ax_d._skk_twin = [axd2]
    axd2.plot(y, np.sqrt(p['wp2']) / a, color=C_ORANGE)
    axd2.set_ylabel(r'$\omega_p/\omega_r$', color=C_ORANGE)
    axd2.tick_params(axis='y', labelcolor=C_ORANGE); axd2.set_ylim(0, 0.25)
    ax_d.plot([], [], color=C_ORANGE, label=r'$\omega_p/\omega_r$')
    print(f'B1s panel d: gamma/2|w\'\'| in [{g.min():.2f}, {g.max():.2f}], w0/w_r in '
          f'[{(np.sqrt(p["w02"]) / a).min():.4f}, {(np.sqrt(p["w02"]) / a).max():.4f}], '
          f'wp/w_r max {(np.sqrt(p["wp2"]) / a).max():.3f}')
    ax_d.set_xlabel(r'$y/\lambda_r$'); ax_d.set_ylabel(r'$\gamma/2|\omega^{\prime\prime}|$,  $\omega_0/\omega_r$')
    ax_d.set_xlim(-1, 1); ax_d.set_ylim(0.4, 1.6); ax_d.legend(loc='upper left', fontsize=5)
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


# --- Long et al. 2026 perfect lens (their Table 1), 1-D normal incidence -------------------
# Their units: w_c = 1.256 - 0.4i; rescaled by s so that w_r = 2 pi (lam_r = 1, c = 1).
LENS_S = cf.TWO_PI / 1.256
LENS_ETA = 0.4 / 1.256
LENS_M = {'e1': (3.873, 2.559, 2.233), 'm1': (1.483, 1.472, 0.917),      # (wp, w0, gamma)
          'e2': (1.225, 1.197, 0.743), 'm2': (1.245, 1.230, 0.755)}
LENS_P = {k: dict(wp2=np.array([(v[0] * LENS_S) ** 2]), w02=np.array([(v[1] * LENS_S) ** 2]),
                  gam=np.array([v[2] * LENS_S])) for k, v in LENS_M.items()}
LENS_LAM0, LENS_NS = 277.98 / 1500.0, 20
LENS_D = np.concatenate([np.full(LENS_NS, 0.5 * LENS_LAM0 / LENS_NS),
                         np.full(2 * LENS_NS, LENS_LAM0 / (2 * LENS_NS)),
                         np.full(LENS_NS, 0.5 * LENS_LAM0 / LENS_NS)])


def _lens_em(w):
    """(eps, mu) of medium 1 | medium 2 | medium 1, per layer, at the frequencies w."""
    e1, m1, e2, m2 = [cf.eps_lorentz(w, LENS_P[k]) for k in ['e1', 'm1', 'e2', 'm2']]
    ns = LENS_NS
    cat = lambda u, v: np.concatenate([np.repeat(u, ns, 1), np.repeat(v, 2 * ns, 1),
                                       np.repeat(u, ns, 1)], 1)
    return cat(e1, e2), cat(m1, m2)


def _lens_rt(w, clad='vac'):
    """r, t of the lens at normal incidence. Default clad='vac' (back to vacuum 2026-11-17: no
    branch cuts in the map; the time domain is the same, 8.0e4 vs 8.3e4 at t = 40). clad='m1': semi-infinite
    Medium 1 on both sides, as in Long et al. Fig. 2 (their source sits in Medium 1 and their
    propagating/evanescent boundaries are Medium 1's light lines); 'vac': vacuum outside.
    t = exit-face field / entry-face incident field. With Medium 1 outside only its impedance
    enters, kappa = w sqrt(eps1)/sqrt(mu1) (principal roots: the passive branch on the real axis;
    branch points at Medium 1's own poles/zeros, Im w <= -0.365 w_r, below the drive line)."""
    E, U = _lens_em(w)
    if clad == 'vac':
        return cf.rt(E, LENS_D, w, mu=U)
    w = np.atleast_1d(np.asarray(w, complex))
    e1, m1 = E[:, 0], U[:, 0]
    kap = w * np.sqrt(e1) / np.sqrt(m1)
    rL, tL = cf._rt_from_M(cf._layer_product(E, LENS_D, w, 0.0, U, 's'), kap)
    return dict(rL=rL, tL=tL)


def _lens_map(xr, yi):
    """log10 |t(w)|^2 of the vacuum-clad lens on the grid xr + i yi (no branch cuts: vacuum's
    k = w is single-valued). The Medium-1-clad version (2026-11-16) is in git history (ed85492)."""
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    return 2 * np.log10(np.abs(_lens_rt(W)['tL'])).reshape(yi.size, xr.size)


def _lens_td():
    """Time-domain t(t)/E_inc(t), r(t)/E_inc(t) of the lens, and its w_c response.
    eta = 0.32: the drive falls e^{-2} per period, so use the contour-shifted synthesis
    (c = 0.28 w_r, just above the highest strip pole at Im w = -0.289 w_r) to follow the
    ratio for 40 periods; it matches the c = 0 synthesis wherever the latter is resolved."""
    wcL = cf.omega_c(LENS_ETA)
    oc = _lens_rt([wcL])
    t = np.arange(NT) * DT / 2
    c = 0.28 * cf.TWO_PI
    dcx, tl = cf.complex_response_shifted(lambda w: _lens_rt(w)['tL'], t, LENS_ETA, 5.0, T0, c)
    _, rl = cf.complex_response_shifted(lambda w: _lens_rt(w)['rL'], t, LENS_ETA, 5.0, T0, c)
    return t - T0, tl / dcx, rl / dcx, oc


def lorentz_poles(p, j=0):
    """The two poles (complex w) of Lorentz layer j: -i gam/2 +- sqrt(w0^2 - gam^2/4)."""
    return np.roots([-1, -1j * p['gam'][j], p['w02'][j]])


def _cbar_right(fig, ax, im, label, gap_in=0.08, w_in=0.08, extend='neither'):
    """Colorbar beside `ax` in the space reserved to its right, without touching its 4:3 box."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + gap_in / W, pos.y0, w_in / W, pos.height])
    cb = fig.colorbar(im, cax=cax, extend=extend); cb.set_label(label); cb.outline.set_linewidth(0.6)
    return cb


# |t| maps: magma, values above the top of the colour range (near slab poles) drawn white and
# flagged by the triangle on the colourbar (2026-11-16)
MAGMA_OVER = plt.get_cmap('magma').copy(); MAGMA_OVER.set_over('w')


def _tmap(ax, x, y, Z, lim=(-2, 6)):
    return ax.pcolormesh(x, y, np.maximum(Z, lim[0]), cmap=MAGMA_OVER, vmin=lim[0], vmax=lim[1],
                         shading='auto', rasterized=True)


def _drive_marks(ax, eta, line='k', star='k'):
    """Real axis (dotted), drive line Im w = -|w''| (dashed), w_c (star)."""
    ax.axhline(-eta, color=line, lw=C.LW_GUIDE, ls='--')
    ax.axhline(0, color=line, lw=C.LW_GUIDE, ls=':')
    ax.plot([1], [-eta], marker='*', color=star, ms=7, ls='none')


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
    t, tl, rl, oc = _lens_td()
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


# ---------------------------------------------------------------------------------------------
# Presentation figures (2026-11-16). Separate functions because each shows a different object
# (one layer's eps over complex w; the B3a slabs as maps + COMSOL; the Long lens with its map).
# ---------------------------------------------------------------------------------------------
SLAB_CASES = [(1 + 0.3j, 1.5, 'loss at $\\omega_c$'), (1 - 0.3j, 0.5, 'gain at $\\omega_c$')]


def fig_single_layer():
    """B0: ONE passive Lorentz medium with eps(w_c) = 1 - 0.3i (B3a's gain slab, gam/2|w''| = 0.5).
    a: Im eps over complex w. For w = x - iy (x > 0), Im eps = wp^2 x (gam - 2y)/|D|^2, so the sign
       flips exactly on the horizontal line through the pole, Im w = -gam/2: loss above, gain below.
       The drive line sits below the pole -> it sees gain.
    b: eps along the real axis (solid, passive) and along the drive line (dashed, gain at w_r)."""
    wc = cf.omega_c(ETA); a, b = wc.real, abs(wc.imag)
    p = cf.lorentz_design_gamma(np.array([-0.3j]), wc, 0.5 * 2 * b); assert p['ok']
    gap = C.YLABEL_W_IN + 0.08 + 0.08 + C.YLABEL_W_IN + C.CENTER_GAP_IN
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.47, margin_r=C.YLABEL_W_IN + C.LAYOUT_PAD_IN,
                               col_gap=gap)
    ax_a, ax_b = axs[0]
    xr = a * np.linspace(0.85, 1.15, 301); yi = a * np.linspace(-0.10, 0.01, 221)
    W = xr[None, :] + 1j * yi[:, None]
    Z = cf.eps_lorentz(W.ravel(), p)[:, 0].imag.reshape(W.shape)
    # 2026-11-16: viridis, not red/blue (those are eps'/eps'' in panel b)
    im = ax_a.pcolormesh(xr / a, yi / a, np.clip(Z, -0.6, 0.6), cmap='viridis', vmin=-0.6, vmax=0.6,
                         shading='auto', rasterized=True)
    _drive_marks(ax_a, ETA, line='w', star='w')
    pole = lorentz_poles(p); pole = pole[pole.real > 0][0] / a
    ax_a.plot(pole.real, pole.imag, marker='x', color='w', ms=5, mew=1.0, ls='none')
    ax_a.text(0.865, -ETA + 0.003, 'drive', fontsize=6, color='w')
    ax_a.text(0.865, 0.0025, 'real axis', fontsize=6, color='w')
    ax_a.text(pole.real + 0.008, pole.imag + 0.002, r'pole $-i\gamma/2$', fontsize=6, color='w')
    ax_a.text(1.008, -ETA - 0.008, r'$\omega_c$', fontsize=6, color='w')
    ax_a.set_xlabel(r"Re $\omega/\omega_r$"); ax_a.set_ylabel(r"Im $\omega/\omega_r$")
    _cbar_right(fig, ax_a, im, r"$\epsilon''(\omega)$")

    x = np.linspace(0.8, 1.2, 2001)
    er, ec = cf.eps_lorentz(a * x, p)[:, 0], cf.eps_lorentz(a * x - 1j * b, p)[:, 0]
    axr = C.profile_panel(ax_b, x, er.real, er.imag, xlabel=r'Re $\omega/\omega_r$',
                          ylab_im=r"$\epsilon''(\omega)$", xlim=(0.8, 1.2))
    ax_b.set_ylabel(r"$\epsilon'(\omega)$", color=C.C_EPS_RE)
    # gam = |w''| here, so eps on the drive line is exactly conj(eps on the real axis): eps' is the
    # same on both lines (one solid curve), eps'' flips sign
    axr.plot(x, ec.imag, color=C.C_EPS_IM, lw=C.LW_MAIN, ls='--')
    axr.axhline(0, color='gray', lw=C.LW_GUIDE, ls=':')     # eps'' = 0: the mirror line of the conjugate symmetry
    ax_b.set_ylim(0.6, 1.4); axr.set_ylim(-0.6, 0.6)
    ax_b.plot([], [], color='k', lw=C.LW_MAIN, label='real axis')
    ax_b.plot([], [], color='k', lw=C.LW_MAIN, ls='--', label='drive line')
    ax_b.legend(loc='upper left', fontsize=5)
    e_wc = cf.eps_lorentz([wc], p)[0, 0]
    print(f'B0: wp/w_r={np.sqrt(p["wp2"][0])/a:.4f} w0/w_r={np.sqrt(p["w02"][0])/a:.4f} '
          f'gam/w_r={p["gam"][0]/a:.4f} (gam/2|w"|={p["gam"][0]/(2*b):.2f}); pole={pole:.4f}; '
          f'eps(w_c)={e_wc:.4f}; min Im eps on real axis={er.imag.min():.2e}')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B0_single_lorentz_layer.png'))


def _comsol_ratio(chi_im, Ls=1.0):
    """COMSOL (finest available step) |t(t)/t_free(t)| for the B3a slab with Im chi = chi_im."""
    import glob
    from comsol.fig_b3a_comsol import drive_c
    files = sorted(glob.glob(os.path.join(_HERE, 'data', 'b3a_comsol_dt*_rho0.75.npz')),
                   key=lambda f: int(os.path.basename(f).split('_dt')[1].split('_')[0]))
    R = np.load(files[-1])
    k = [i for i in range(0, len(R['combos']), 2) if np.isclose(R['combos'][i, 0], chi_im)][0]
    ec = R['e_exit'][k] - 1j * R['e_exit'][k + 1]
    fc = drive_c(R['t'] - Ls, ETA)
    ok = np.abs(fc) > 1e-10
    return R['t'][ok] - T0 - Ls, np.abs(ec[ok]) / np.abs(fc[ok]), os.path.basename(files[-1])


def fig_slab_td(mode='T'):
    """B3s: B3a's two Long-type single-Lorentz 1 lam_r slabs, side by side.
    left: eps(w_c) = 1 + 0.3i (gam/2|w''| = 1.5, loss at w_c); right: 1 - 0.3i (0.5, gain).
    a, b: log10|t(w)| over complex w (bright = slab poles; x = material pole; dashed = drive).
    c, d, by `mode`:
      'T'   (B3s)  T(t) = |E_out(t)|^2 / |E_inc(t - L/c)|^2 = |t(t)/t_free(t)|^2, the output
                   compared with the drive that entered one vacuum transit time earlier (vacuum:
                   T = 1). TMM x FFT (line), COMSOL (circles, one per 5 T_r); dotted = T(w_c).
      't'   (B3s2) |t(t)|^2 = |E_out(t)|^2 / |E_inc(t)|^2, same instant: only the constant
                   |t_free|^2 = e^{2|w''|L/c} = 1.87 differs (B3s x 1.87).
      'raw' (B3s3) no division at all: |E_out(t)|^2 (blue) and the drive |E_inc(t)|^2 (gray)."""
    sys.path.insert(0, os.path.join(_HERE, 'comsol'))
    wc = cf.omega_c(ETA); a, b = wc.real, abs(wc.imag)
    fig, axs, _ = C.panel_grid(2, 2, margin_l=0.47, margin_r=C.YLABEL_W_IN + 0.12)
    (ax_a, ax_b), (ax_c, ax_d) = axs
    wr = cf.rfft_omega(NT, DT)
    xr = a * np.linspace(0.9, 1.1, 241); yi = a * np.linspace(-0.12, 0.006, 190)
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    Ls, nl = 1.0, 40; d = np.full(nl, Ls / nl)
    sh = int(round(Ls / DT))
    for (tgt, g, lab), axm, axt in zip(SLAB_CASES, (ax_a, ax_b), (ax_c, ax_d)):
        p = cf.lorentz_design_gamma(np.array([tgt - 1]), wc, g * 2 * b); assert p['ok']
        e = lambda w: np.repeat(cf.eps_lorentz(w, p), nl, axis=1)
        # T(w) = |t/t_free|^2 over complex w (same poles as t; same quantity as panels c, d)
        Z = 2 * np.log10(np.abs(cf.rt(e(W), d, W)['tauL'])).reshape(yi.size, xr.size)
        im = _tmap(axm, xr / a, yi / a, Z)
        _drive_marks(axm, ETA, line='w', star='c')
        axm.text(1.005, -ETA + 0.004, r'$\omega_c$', color='c', fontsize=6)
        pole = lorentz_poles(p); pole = pole[pole.real > 0][0] / a
        axm.plot(pole.real, pole.imag, marker='x', color='w', ms=4, mew=0.9, ls='none')
        axm.text(0.03, 0.06, rf'$\epsilon(\omega_c)={tgt.real:.0f}{tgt.imag:+.1f}i$, '
                 rf'$\gamma/2|\omega^{{\prime\prime}}|={g}$', transform=axm.transAxes, color='w', fontsize=6)
        axm.set_xlabel(r"Re $\omega/\omega_r$")

        oc = cf.rt(e([wc]), d, [wc])
        o = cf.rt(e(wr), d, wr)
        t = np.arange(NT) * DT
        if mode == 'raw':
            dcx, out = cf.complex_response(o['tL'], t, ETA, T_ON, T0)
            ok = np.abs(dcx) > 1e-14 * np.abs(dcx).max()
            axt.semilogy(t[ok] - T0, np.abs(dcx[ok]) ** 2, color='gray', lw=C.LW_THIN, ls='--',
                         label=r'drive $|E_{\rm inc}(t)|^2$')
            axt.semilogy(t[ok] - T0, np.abs(out[ok]) ** 2, color=C.C_T, lw=C.LW_THIN,
                         label=r'output $|E_{\rm out}(t)|^2$')
            axt.set_ylim(1e-24, 1e1); axt.set_ylabel('power at the faces')
        else:
            tt, rt_ = _td(o['tL'], ETA); _, rf = _td(np.exp(1j * wr * Ls), ETA)
            if mode == 'T':
                Tt, Tc = (np.abs(rt_[sh:]) / np.abs(rf[sh:])) ** 2, abs(oc['tauL'][0]) ** 2
                xt = tt[sh:] - Ls
            else:
                Tt, Tc, xt = np.abs(rt_) ** 2, abs(oc['tL'][0]) ** 2, tt
            mk = xt > 1                                # skip the 0/0 before the light arrives
            axt.semilogy(xt[mk], Tt[mk], color=C.C_T, lw=C.LW_THIN, label=r'TMM $\times$ FFT')
            if mode == 'T':
                tq, qc, fname = _comsol_ratio(tgt.imag, Ls)
                sel = (tq >= 1) & (np.abs(np.round(tq / 5) * 5 - tq) < 1e-6)
                axt.plot(tq[sel], qc[sel] ** 2, marker='o', mfc='none', mec='k', ms=3, mew=0.6,
                         ls='none', label='COMSOL')
            axt.axhline(Tc, color=C.C_T, lw=C.LW_GUIDE, ls=':')
            axt.set_ylim(1e-3, 1e6)
            i60 = np.argmin(abs(xt - 60))
            print(f'B3s[{mode}] {tgt}: pole/w_r={pole:.4f}, T(w_c)={Tc:.4g}, TMMxFFT at t=60: {Tt[i60]:.4g}'
                  + (f', COMSOL ({fname}) at t=60: {np.interp(60, tq, qc) ** 2:.4g}' if mode == 'T' else ''))
        axt.text(0.97, 0.06, lab, transform=axt.transAxes, fontsize=6, ha='right')
        axt.set_xlim(0, 60)
        axt.set_xlabel(r'time after turn-on ($T_r$)')
        axt.legend(loc='lower left' if mode == 'raw' else 'upper left', fontsize=5)
    ax_a.set_ylabel(r"Im $\omega/\omega_r$")
    if mode != 'raw':
        ax_c.set_ylabel(r'$T(t)$' if mode == 'T' else r'$|t(t)|^2$')
    _cbar_right(fig, ax_b, im, r'$\log_{10}T(\omega)$', extend='max').set_ticks([-2, 0, 2, 4, 6])
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c'), (ax_d, 'd')])
    C.save(fig, os.path.join(OUT, {'T': 'B3s_slab_loss_vs_gain.png', 't': 'B3s2_no_tfree.png',
                                   'raw': 'B3s3_raw_power.png'}[mode]))


def _comsol_raw(chi_im, Ls=1.0):
    """COMSOL (finest step) exit-face field E_out(t) itself, |E_out|^2, on the B3s time axis."""
    import glob
    files = sorted(glob.glob(os.path.join(_HERE, 'data', 'b3a_comsol_dt*_rho0.75.npz')),
                   key=lambda f: int(os.path.basename(f).split('_dt')[1].split('_')[0]))
    R = np.load(files[-1])
    k = [i for i in range(0, len(R['combos']), 2) if np.isclose(R['combos'][i, 0], chi_im)][0]
    ec = R['e_exit'][k] - 1j * R['e_exit'][k + 1]
    return R['t'] - T0 - Ls, np.abs(ec) ** 2


def fig_slab_wide():
    """B3s 2x3 (slide layout, 2026-11-16): rows = loss slab (1 + 0.3i), gain slab (1 - 0.3i).
    col 1: log10 T(w) over complex w (x = material pole, dashed = drive line, star = w_c).
    col 2: the raw powers at the exit face, no division: |E_out(t)|^2 (blue, COMSOL circles) and
           what vacuum would deliver there, |E_inc(t - L/c)|^2 (gray dashed). (The dotted driven-term
           line T(w_c)|E_inc(t - L/c)|^2 was removed 2026-11-17 for the talk.)
    col 3: T(t) = |E_out(t)|^2 / |E_inc(t - L/c)|^2 = col 2 blue / col 2 gray (B3s c, d).
    Time axis: t - L/c after turn-on (when the first light can reach the exit face)."""
    sys.path.insert(0, os.path.join(_HERE, 'comsol'))
    wc = cf.omega_c(ETA); a, b = wc.real, abs(wc.imag)
    gap = 0.08 + 0.08 + C.YLABEL_W_IN + C.YLABEL_W_IN + C.CENTER_GAP_IN
    fig, axs, _ = C.panel_grid(2, 3, width_in=7.5, margin_l=0.56, col_gap=gap,
                               margin_t=C.LAYOUT_PAD_IN + 0.16)
    # only column 1 carries a colourbar: pull column 3 back to the normal gap, shrink the figure
    W0, H0 = fig.get_size_inches(); dx = gap - (C.YLABEL_W_IN + C.CENTER_GAP_IN)
    boxes = [(ax, ax.get_position()) for ax in axs.ravel()]
    fig.set_size_inches(W0 - dx, H0)
    for ax, p in boxes:
        x0 = p.x0 * W0 - (dx if ax in axs[:, 2] else 0)
        ax.set_position([x0 / (W0 - dx), p.y0, p.width * W0 / (W0 - dx), p.height])
    wr = cf.rfft_omega(NT, DT)
    xr = a * np.linspace(0.9, 1.1, 241); yi = a * np.linspace(-0.12, 0.006, 190)
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    Ls, nl = 1.0, 40; d = np.full(nl, Ls / nl)
    sh = int(round(Ls / DT))
    t = np.arange(NT) * DT
    _, vac = cf.complex_response(np.exp(1j * wr * Ls), t, ETA, T_ON, T0)   # E_inc(t - L/c)
    for (tgt, g, lab), (axm, axp, axt) in zip(SLAB_CASES, axs):
        p = cf.lorentz_design_gamma(np.array([tgt - 1]), wc, g * 2 * b); assert p['ok']
        e = lambda w: np.repeat(cf.eps_lorentz(w, p), nl, axis=1)
        Z = 2 * np.log10(np.abs(cf.rt(e(W), d, W)['tauL'])).reshape(yi.size, xr.size)
        im = _tmap(axm, xr / a, yi / a, Z)
        _drive_marks(axm, ETA, line='w', star='c')
        axm.text(1.005, -ETA + 0.004, r'$\omega_c$', color='c', fontsize=6)
        pole = lorentz_poles(p); pole = pole[pole.real > 0][0] / a
        axm.plot(pole.real, pole.imag, marker='x', color='w', ms=4, mew=0.9, ls='none')
        axm.text(0.03, 0.06, rf'$\epsilon(\omega_c)={tgt.real:.0f}{tgt.imag:+.1f}i$, '
                 rf'$\gamma/2|\omega^{{\prime\prime}}|={g}$', transform=axm.transAxes, color='w', fontsize=6)
        axm.set_ylabel(r"Im $\omega/\omega_r$")
        _cbar_right(fig, axm, im, r'$\log_{10}T(\omega)$', extend='max').set_ticks([-2, 0, 2, 4, 6])

        Tc = abs(cf.rt(e([wc]), d, [wc])['tauL'][0]) ** 2
        _, out = cf.complex_response(cf.rt(e(wr), d, wr)['tL'], t, ETA, T_ON, T0)
        x = t - T0; Po, Pv = np.abs(out) ** 2, np.abs(vac) ** 2     # x = t since turn-on
        ok = (x > Ls) & (Pv > 1e-28)                                # nothing reaches the exit before L/c
        axp.semilogy(x[ok], Pv[ok], color='gray', ls='--', lw=C.LW_THIN, label=r'$|E_{\rm inc}(t-L/c)|^2$')
        axp.semilogy(x[ok], Po[ok], color=C.C_T, lw=C.LW_THIN, label=r'$|E_{\rm out}(t)|^2$')
        tq, pq = _comsol_raw(tgt.imag, Ls); tq = tq + Ls
        sel = (tq >= 1) & (np.abs(np.round(tq / 5) * 5 - tq) < 1e-6)
        axp.plot(tq[sel], pq[sel], marker='o', mfc='none', mec='k', ms=3, mew=0.6, ls='none', label='COMSOL')
        axp.set_ylim(1e-20, 1e1); axp.set_ylabel('power at the exit face')

        Tt = Po[ok] / Pv[ok]; xt = x[ok]
        mk = xt > Ls + 0.5                       # skip the 0/0 at the arrival of the first light
        axt.semilogy(xt[mk], Tt[mk], color=C.C_T, lw=C.LW_THIN, label=r'TMM $\times$ FFT')
        tq2, qc, _ = _comsol_ratio(tgt.imag, Ls); tq2 = tq2 + Ls
        sel = (tq2 >= 1) & (np.abs(np.round(tq2 / 5) * 5 - tq2) < 1e-6)
        axt.plot(tq2[sel], qc[sel] ** 2, marker='o', mfc='none', mec='k', ms=3, mew=0.6, ls='none',
                 label='COMSOL')
        axt.axhline(Tc, color=C.C_T, lw=C.LW_GUIDE, ls=':')
        axt.set_ylim(1e-3, 1e6); axt.set_ylabel(r'$T(t)$')
        i60 = np.argmin(abs(xt - 60)); j60 = np.argmin(abs(tq - 60))
        print(f'B3s wide {tgt}: T(w_c)={Tc:.4g}; t=60: T={Tt[i60]:.4g}, |E_out|^2 TMM={Po[ok][i60]:.3e} '
              f'COMSOL={pq[j60]:.3e}, |E_inc(t-L)|^2={Pv[ok][i60]:.3e}')
        for ax in (axp, axt):
            ax.set_xlim(0, 60)
        axt.text(0.97, 0.06, lab, transform=axt.transAxes, fontsize=6, ha='right')
    for ax in axs[1]:
        ax.set_xlabel(r'time after turn-on ($T_r$)')
    axs[1][0].set_xlabel(r"Re $\omega/\omega_r$"); axs[0][0].set_xlabel(r"Re $\omega/\omega_r$")
    axs[0][1].legend(loc='lower left', fontsize=5); axs[0][2].legend(loc='upper left', fontsize=5)
    for ax, ttl in zip(axs[0], (r'$T(\omega)$ over complex $\omega$', 'raw powers',
                                r'$T(t)=|E_{\rm out}(t)|^2/|E_{\rm inc}(t-L/c)|^2$')):
        ax.set_title(ttl, fontsize=7)
    C.label_panels(fig, [(ax, l) for ax, l in zip(axs[:, :2].ravel(), 'abde')], dx_in=0.54)
    C.label_panels(fig, [(axs[0][2], 'c'), (axs[1][2], 'f')], dx_in=0.42)
    C.save(fig, os.path.join(OUT, 'B3s4_slab_2x3.png'))


def fig_drive_delay():
    """B3s0 (explainer for B3s4): the drive at the entry face E_inc(t) and the same drive delayed by
    the vacuum transit time, E_inc(t - L/c) = what vacuum would deliver to the exit face at t
    (L = 1 lam_r, so L/c = 1 T_r). a: the fields, linear; b: their powers, log. Because the drive
    decays, the delayed copy is ALWAYS larger, by e^{|w''|L/c} = 1.37 (power 1.87)."""
    Ls = 1.0
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.47)
    ax_a, ax_b = axs[0]
    t = np.arange(NT) * DT; wr = cf.rfft_omega(NT, DT)
    dcx, vac = cf.complex_response(np.exp(1j * wr * Ls), t, ETA, T_ON, T0)
    x = t - T0
    m = (x > 12) & (x < 20)
    ax_a.plot(x[m], dcx[m].real, color='0.75', lw=C.LW_THIN)
    ax_a.plot(x[m], np.abs(dcx[m]), color='0.45', lw=C.LW_MAIN, label=r'$E_{\rm inc}(t)$')
    ax_a.plot(x[m], np.abs(vac[m]), color='0.45', lw=C.LW_MAIN, ls='--', label=r'$E_{\rm inc}(t-L/c)$')
    h = np.interp(15.0, x, np.abs(dcx))                       # same height, one transit time apart
    ax_a.annotate('', xy=(16.0, h), xytext=(15.0, h),
                  arrowprops=dict(arrowstyle='<->', lw=0.7, color='k', shrinkA=0, shrinkB=0))
    ax_a.text(15.5, h - 0.003, r'$L/c$', ha='center', va='top', fontsize=6)
    ax_a.set_xlim(12, 20); ax_a.set_ylim(-0.04, 0.04)
    ax_a.set_xlabel(r'time after turn-on ($T_r$)'); ax_a.set_ylabel('field')
    ax_a.legend(loc='lower left', fontsize=5)
    ok = x > Ls + 0.5
    ratio = np.abs(vac[ok]) ** 2 / np.abs(dcx[ok]) ** 2
    ax_b.plot(x[ok], ratio, color='0.45', lw=C.LW_MAIN)
    ax_b.axhline(np.exp(2 * cf.TWO_PI * ETA * Ls), color='k', lw=C.LW_GUIDE, ls=':')
    ax_b.text(59, np.exp(2 * cf.TWO_PI * ETA * Ls) + 0.05, r'$e^{2|\omega^{\prime\prime}|L/c}=1.87$',
              ha='right', va='bottom', fontsize=6)
    i = np.argmin(abs(x[ok] - 40))
    print(f'B3s0: |E_inc(t-L/c)|^2/|E_inc(t)|^2 at t=40: {ratio[i]:.4f} '
          f'(e^(2|w"|L/c) = {np.exp(2*cf.TWO_PI*ETA*Ls):.4f})')
    ax_b.set_xlim(0, 60); ax_b.set_ylim(0, 2.5)
    ax_b.set_xlabel(r'time after turn-on ($T_r$)')
    ax_b.set_ylabel(r'$|E_{\rm inc}(t-L/c)|^2\,/\,|E_{\rm inc}(t)|^2$')
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B3s0_drive_delay.png'))


def fig_lens():
    """B3L: Long et al.'s own perfect lens (Table 1 media, 1-D normal incidence, eta = 0.32).
    a: log10|t(w)| of the lens over complex w; x = material poles of eps and mu of medium 1
       (lossy at w_c) and medium 2 (gain at w_c); dashed = their drive line.
    b: the time domain (= B3 panel d): it never settles to the w_c values (dashed)."""
    gap = C.YLABEL_W_IN + 0.08 + 0.08 + C.YLABEL_W_IN + C.CENTER_GAP_IN
    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.47, col_gap=gap)
    ax_a, ax_b = axs[0]
    wcL = cf.omega_c(LENS_ETA); a = wcL.real
    # 2026-11-16: window widened to Re 0.6-2.0, Im down to -1.0 so medium 1's eps pole (1.83 - 0.89i)
    # is in view too
    xr = a * np.linspace(0.6, 2.0, 351); yi = a * np.linspace(-1.0, 0.01, 253)
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    Z = _lens_map(xr, yi)                    # |t|^2 (see b)
    im = _tmap(ax_a, xr / a, yi / a, Z)
    _drive_marks(ax_a, LENS_ETA, line='w', star='c')
    # inset: zoom on the drive line around w_r, where medium 2's poles sit just above it
    zx, zy = (0.84, 1.16), (-0.40, -0.24)
    axi = ax_a.inset_axes([0.5, 0.45, 0.47, 0.5])
    mx = (xr / a >= zx[0] - 0.01) & (xr / a <= zx[1] + 0.01)
    yz = a * np.linspace(zy[0], zy[1], 161)
    Wz = (xr[mx][None, :] + 1j * yz[:, None]).ravel()
    Zz = _lens_map(xr[mx], yz)
    _tmap(axi, xr[mx] / a, yz / a, Zz)
    _drive_marks(axi, LENS_ETA, line='w', star='c')
    axi.set_xlim(*zx); axi.set_ylim(*zy)
    axi.tick_params(labelsize=5, color='w', labelcolor='w', length=2, pad=1)
    for sp in axi.spines.values():
        sp.set_color('w')
    ax_a.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec='w',
                                 lw=C.LW_GUIDE))
    names = {'e1': r'$\epsilon_1$', 'm1': r'$\mu_1$', 'e2': r'$\epsilon_2$', 'm2': r'$\mu_2$'}
    for k in ('e1', 'm1', 'e2', 'm2'):
        pl = lorentz_poles(LENS_P[k]); pl = pl[pl.real > 0][0] / a
        print(f'B3L material pole {k}: {pl:.4f}  ({"above" if pl.imag > -LENS_ETA else "below"} the drive line)')
        inz = zx[0] < pl.real < zx[1] and zy[0] < pl.imag < zy[1]
        ax = axi if inz else ax_a
        ax.plot(pl.real, pl.imag, marker='x', color='w', ms=4, mew=0.9, ls='none')
        dx, ha = ((-0.006, 'right') if k == 'e2' else (0.006, 'left')) if inz else (0.03, 'left')
        ax.text(pl.real + dx, pl.imag + (0.006 if inz else 0.02), names[k], color='w', fontsize=6, ha=ha)
    axi.text(1.005, -LENS_ETA - 0.018, r'$\omega_c$', color='c', fontsize=6)
    ax_a.text(0.03, 0.04, 'Long et al. lens', transform=ax_a.transAxes, color='w', fontsize=6)
    ax_a.set_xlabel(r"Re $\omega/\omega_r$"); ax_a.set_ylabel(r"Im $\omega/\omega_r$")
    _cbar_right(fig, ax_a, im, r'$\log_{10}|t(\omega)|^2$', extend='max').set_ticks([-2, 0, 2, 4, 6])

    # b: |t(t)|^2, |r(t)|^2 = out / drive at the SAME instant (no L/c shift, so not called T, R):
    # a perfect lens is designed for t = 1 itself (Long et al. Fig. 2c), image = source.
    t, tl, rl, oc = _lens_td()
    # start at L/c (0.37 T_r), the first instant light can reach the exit face; before it the
    # output is zero and the drive is still ~0 (Planck ramp), so the ratio is 0/0 noise
    m = t > LENS_D.sum()
    t, tl, rl = t[m], tl[m], rl[m]
    ax_b.semilogy(t, np.abs(tl) ** 2, color=C.C_T, label=r'$|t(t)|^2$')
    ax_b.semilogy(t, np.abs(rl) ** 2, color=C.C_R, label=r'$|r(t)|^2$')
    ax_b.axhline(abs(oc['tL'][0]) ** 2, color=C.C_T, ls=':', lw=C.LW_GUIDE)
    ax_b.axhline(abs(oc['rL'][0]) ** 2, color=C.C_R, ls=':', lw=C.LW_GUIDE)
    print(f'B3L at w_c: T={abs(oc["tL"][0])**2:.4f} R={abs(oc["rL"][0])**2:.2e}; at t=40: '
          f'T={np.interp(40, t, np.abs(tl)**2):.3g} R={np.interp(40, t, np.abs(rl)**2):.3g}')
    ax_b.set_xlim(0, 40); ax_b.set_ylim(1e-10, 1e6)
    ax_b.set_xlabel(r'time after turn-on ($T_r$)')
    ax_b.legend(loc='upper left', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    C.save(fig, os.path.join(OUT, 'B3L_long_lens.png'))


def _lens_kx_setup():
    """Lens media with Medium 2 refit so eps2(w_c) = -eps1(w_c), mu2(w_c) = -mu1(w_c) EXACTLY
    (gamma kept; wp, w0 change < 0.2 %), so t(w_c) = 1 at every k_par as in Long Fig. 2c
    (Table 1 rounding alone gives |t| = 0.40 at 1.5 k0)."""
    LP = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in LENS_P.items()}
    wc = cf.omega_c(LENS_ETA)
    ev = lambda k, w: cf.eps_lorentz(np.atleast_1d(w), LP[k])[:, 0]
    for k2, k1 in (('e2', 'e1'), ('m2', 'm1')):
        chi = -ev(k1, wc)[0] - 1
        base = -wc ** 2 - 1j * LP[k2]['gam'][0] * wc
        wp2 = base.imag / np.imag(1 / chi)
        LP[k2]['wp2'] = np.array([wp2]); LP[k2]['w02'] = np.array([(wp2 / chi - base).real])
    return LP, ev


def _q1_down(w, kx, ev, steps=40):
    """q1 = sqrt(w^2 eps1 mu1 - kx^2) in Medium 1: Im q1 > 0 just above the real axis (outgoing),
    continued straight down to Im w (valid while no Medium-1 branch point lies in between)."""
    w = np.asarray(w, complex); prev = None
    for s in np.linspace(0, 1, steps + 1):
        ws = w.real + 1j * (1e-9 - s * (-w.imag + 1e-9))
        q = np.sqrt(ws ** 2 * ev('e1', ws) * ev('m1', ws) - kx ** 2)
        q = np.where(q.imag < 0, -q, q) if prev is None else np.where(np.abs(q - prev) > np.abs(q + prev), -q, q)
        prev = q
    return q


def fig_lens_kx():
    """B3L2 (backup, 2026-11-17; new block because it is a separate slide): the lens at fixed
    real k_par (TM, Medium 1 outside), |t(t = 10 T_r)|^2 vs k_par, next to Medium 1 ALONE (no
    lens, same planes). Frequency domain gives |t(w_c)|^2 = 1 for both. At k_par != 0 Medium 1's
    own light-line branch points (w^2 eps1 mu1 = k_par^2) sit above the drive line, so even the
    background never settles. Each point is computed on two contours (c, 0.75c) and kept only if
    they agree. Range 0-0.7 k0: closer to k0 the branch point reaches the drive itself, and the
    evanescent side (|t| ~ e^{2 kappa lam0} at real w) is beyond double precision."""
    LP, ev = _lens_kx_setup()
    P = np.polynomial.polynomial
    a = cf.TWO_PI; wc = cf.omega_c(LENS_ETA); k0 = cf.TWO_PI / LENS_LAM0
    d = np.array([0.5, 1.0, 0.5]) * LENS_LAM0

    def lens(w, kx):
        w = np.atleast_1d(np.asarray(w, complex))
        e1, m1, e2, m2 = [ev(k, w) for k in ('e1', 'm1', 'e2', 'm2')]
        M = cf._layer_product(np.stack([e1, e2, e1], 1), d, w, kx, np.stack([m1, m2, m1], 1), 'p')
        return cf._rt_from_M(M, _q1_down(w, kx, ev) / e1)[1]

    def medium1(w, kx):
        w = np.atleast_1d(np.asarray(w, complex))
        return np.exp(1j * _q1_down(w, kx, ev) * d.sum())

    def top_branch(kx):          # highest root of w^2 eps1 mu1 = kx^2
        def ND(k):
            D = np.array([LP[k]['w02'][0], -1j * LP[k]['gam'][0], -1])
            return P.polyadd(D, [LP[k]['wp2'][0]]), D
        (Ne, De), (Nm, Dm) = ND('e1'), ND('m1')
        r = P.polyroots(P.polysub(P.polymul([0, 0, 1], P.polymul(Ne, Nm)), P.polymul([kx ** 2], P.polymul(De, Dm))))
        return r[np.abs(r) > 1e-6].imag.max()

    t = np.arange(NT) * DT / 2; t_read = 10.0
    kr = np.round(np.arange(0, 0.701, 0.05), 3)
    res = {}
    for nm, fn in (('lens', lens), ('m1', medium1)):
        vals = []
        for k in kr:
            c = 0.28 * a if k == 0 else min(0.28 * a, 0.9 * abs(top_branch(k * k0)))
            v = []
            for cc in (c, 0.75 * c):
                dcx, o = cf.complex_response_shifted(lambda w: fn(w, k * k0), t, LENS_ETA, 5.0, T0, cc)
                v.append(np.interp(t_read, t - T0, np.abs(o / dcx) ** 2))
            ok = np.all(np.isfinite(v)) and abs(np.log10(v[0] / v[1])) < 0.02
            vals.append(v[0] if ok else np.nan)
            print(f'B3L2 {nm} k/k0={k:.2f}: |t(10)|^2={v[0]:.3g} (check {v[1]:.3g}) {"ok" if ok else "DROPPED"}')
        res[nm] = np.array(vals)
    tc = abs(lens([wc], 0.5 * k0)[0]) ** 2
    fig, axs, _ = C.panel_grid(1, 1, margin_l=0.47)
    ax = axs[0][0]
    ax.semilogy(kr, res['lens'], color=C.C_T, marker='o', ms=3, label='lens')
    ax.semilogy(kr, res['m1'], color='gray', ls='--', marker='s', ms=3, mfc='none', label='Medium 1 alone')
    ax.axhline(tc, color='k', ls=':', lw=C.LW_GUIDE, label=r'$|t(\omega_c)|^2$ (both)')
    ax.set_xlabel(r'$k_\parallel/k_0$'); ax.set_ylabel(r'$|t(t)|^2$ at $t=10\,T_r$')
    ax.set_xlim(0, 0.7)
    ax.legend(loc='upper right', fontsize=5)
    C.save(fig, os.path.join(OUT, 'B3L2_lens_kpar.png'))


def fig_lens_angles():
    """B3L3 (backup, 2026-11-17; new block = separate slide): the vacuum-clad lens hit by a plane
    wave from a FIXED ANGLE theta (TM, k_par = w sin theta at every frequency, as in B9), Medium 2
    refit so t(w_c) = 1 exactly at every angle. |t(t)|^2 = |out(t)/drive(t)|^2 diverges at every
    angle, as at normal incidence. (Fixed REAL k_par instead -- B3L2 -- is confounded: the outside
    medium's own grazing waves, at its light line, outlast the drive even with no lens.)
    Each curve uses the largest contour shift c that two independent shifts (c, c2) confirm; points
    where they differ by > 2 % are dropped, and each curve stops where the shifted drive reaches
    the double-precision floor."""
    LP, ev = _lens_kx_setup()
    a = cf.TWO_PI; d = np.array([0.5, 1.0, 0.5]) * LENS_LAM0

    def lens(w, th):
        w = np.atleast_1d(np.asarray(w, complex)); w = np.where(w == 0, 1e-9, w)
        e1, m1, e2, m2 = [ev(k, w) for k in ('e1', 'm1', 'e2', 'm2')]
        M = cf._layer_product(np.stack([e1, e2, e1], 1), d, w, w * np.sin(np.radians(th)),
                              np.stack([m1, m2, m1], 1), 'p')
        return cf._rt_from_M(M, w * np.cos(np.radians(th)))[1]

    t = np.arange(NT) * DT / 2
    fig, axs, _ = C.panel_grid(1, 1, margin_l=0.47)
    ax = axs[0][0]
    cols = plt.cm.viridis(np.linspace(0, 0.85, 5))
    for th, (c, c2), col in zip((0, 20, 40, 60, 70), ((0.28, 0.20),) * 3 + ((0.12, 0.06),) * 2, cols):
        r = []
        for cc in (c * a, c2 * a):
            dcx, o = cf.complex_response_shifted(lambda w: lens(w, th), t, LENS_ETA, 5.0, T0, cc)
            r.append(np.abs(o / dcx) ** 2)
        x = t - T0
        tmax = min(40.0, 30 / (LENS_ETA * a - c * a))
        ok = (x > LENS_D.sum()) & (x <= tmax) & (np.abs(np.log10(r[0] / r[1])) < np.log10(1.02))
        ax.semilogy(x[ok], r[0][ok], color=col, lw=C.LW_THIN, label=rf'$\theta={th}^\circ$')
        print(f'B3L3 theta={th}: |t(t)|^2 at t=20: {np.interp(20, x, r[0]):.3g}; shown to t={x[ok].max():.1f}')
    ax.axhline(1, color='k', ls=':', lw=C.LW_GUIDE)
    ax.set_xlim(0, 40); ax.set_ylim(1e-3, 1e9)
    ax.set_xlabel(r'time after turn-on ($T_r$)'); ax.set_ylabel(r'$|t(t)|^2$')
    ax.legend(loc='upper left', fontsize=5)
    C.save(fig, os.path.join(OUT, 'B3L3_lens_angles.png'))


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


FIGS = {'freq': fig_freq, 'freq_simple': fig_freq_simple, 'eta_zero': fig_eta_zero, 'trunc': fig_trunc,
        'time': fig_time, 'grow': fig_grow, 'single_layer': fig_single_layer, 'slab_td': fig_slab_td,
        'slab_td_t': lambda: fig_slab_td('t'), 'slab_td_raw': lambda: fig_slab_td('raw'),
        'lens': fig_lens, 'lens_kx': fig_lens_kx, 'lens_angles': fig_lens_angles, 'slab_wide': fig_slab_wide, 'drive_delay': fig_drive_delay}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for nm in names:
        FIGS[nm]()
