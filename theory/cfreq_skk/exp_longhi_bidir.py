"""B10 (2026-11-19): BIDIRECTIONAL invisibility (Longhi, Opt. Lett. 2016, arXiv:1607.04085) realised
with passive two-line layers under a decaying drive, against Long-type single-Lorentz layers.

Target (Longhi Eq. 13, n_b = 1):  eps(x) = 1 - A (x + i sigma)^-2 exp(i Theta x).
  beta(x) = -A (x + i sigma)^-2 is analytic in Im x >= 0 and exp(i Theta x) shifts its spectrum to
  k >= Theta, so with Theta >= 2 k0 there is no Fourier component that can back-scatter a wave from
  EITHER side: R_L = R_R = 0, t = 1 (his theorem). Im eps takes both signs -> gain at real w.
  Longhi's parameters: sigma k0 = 2, A k0^2 = 1, cut at |x| = 30/k0, Theta = 2 k0.
Choices here (k0 = w_r/c = 2 pi, lam_r = 1):
  - Theta = 3 k0, not Longhi's 2 k0. Theta = 2 k0 is the edge of his theorem: the spectral edge sits
    exactly at the back-scattering wavenumber 2k, so truncation leaks straight into R_R (R_R = 2e-3
    even at real w with our window; 1.1e-3 with his hard cut, as in his Fig. 2b). Under the decaying
    drive the leak is amplified by e^{2|k''| |x|} from the far tail (1/x^2 decay only): R_R = 11 at
    Theta = 2 k0, eta = 0.05. Theta = 3 k0 is inside his theorem and puts the edge well past 2k.
    Both numbers are recorded in the meta (theta_scan).
  - the same cos^2 tail window as B8 (cfreq_lib.cos2_window, flat on |x| <= 15/k0), eta = 0.05 as B8.
  - N = 955 layers (lam_r/100); doubling N changes nothing at 3 significant figures.
Writes data/longhi_bidir.npz (+ meta JSON). Figure: fig_stable_gain.py longhi (B10_fano2_longhi_bidir).
  python theory/cfreq_skk/exp_longhi_bidir.py      (~2 min)
"""
import sys, os, json, time, subprocess, warnings
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import cfreq_lib as cf
from exp_stable_gain import DT, NT, T0, T_ON, KEEP, MODELS, material_poles

warnings.filterwarnings('ignore', category=RuntimeWarning)   # ratios divide by the (zero) drive before t0
K0 = 2 * np.pi
LONGHI = dict(sigma=2 / K0, A=1 / K0 ** 2, theta=3 * K0, Y=30 / K0, N=955, eta=0.05)
THETAS = np.concatenate([np.arange(0, 88, 1.0), [88, 89]])


def longhi_target(sigma, A, theta, Y, **_):
    return lambda x: -A * (x + 1j * sigma) ** -2 * np.exp(1j * theta * x) * cf.cos2_window(x, Y)


def td_all(name, p, d, eta):
    """|r_L|, |r_R|, |tau_L| (output / drive) vs time. One TMM pass per medium, reused for all three
    (exp_stable_gain.td_ratios recomputes it per key; with 955 layers that triples the cost).
    Two-line: contour shifted to Im w = -0.9|b| (all its poles lie below -2|b|; checked on the map);
    single Lorentz: unshifted (its poles lie ABOVE w_c)."""
    wc = cf.omega_c(eta); b = abs(wc.imag); L = d.sum()
    t = np.arange(NT) * DT
    c = 0.0 if name == 'lorentz' else 0.9 * b
    w = cf.rfft_omega(NT, DT) - 1j * c
    R = cf.rt_model(MODELS[name], p, d, w)
    out = {}
    for key, resp in (('rL', R['rL']), ('rR', R['rR']), ('tauL', R['tL'] * np.exp(-1j * w * L))):
        dcx, o = cf.complex_response_shifted(lambda _w, r=resp: r, t, eta, T_ON, T0, c)
        out[key] = np.abs(o / dcx)[::KEEP]
    return (t - T0)[::KEEP], out


def logabs_t(name, p, d, xr, yi):
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    return np.log10(np.abs(cf.rt_model(MODELS[name], p, d, W)['tL'])).reshape(yi.size, xr.size)


def stack_poles(name, p, d, xr, yi, Z, ymin, n_max=25):
    """Poles of t(w) of the whole stack in the band Im w > ymin: local maxima of the log10 T map
    (vacuum factor removed, so deep rows don't dominate) with T > 1, the highest n_max of them
    refined by Newton on 1/t. Only the highest poles matter for settling."""
    L = d.sum()
    ZT = 2 * (Z + yi[:, None] * L / np.log(10))
    g = lambda w: 1 / cf.rt_model(MODELS[name], p, d, [w])['tL'][0]
    core = ZT[1:-1, 1:-1]
    pk = ((core > np.max([ZT[:-2, 1:-1], ZT[2:, 1:-1], ZT[1:-1, :-2], ZT[1:-1, 2:]], axis=0))
          & (core > 0) & (yi[1:-1, None] > ymin))
    ii, jj = np.nonzero(pk)
    order = np.argsort(-yi[ii + 1])[:n_max]
    poles = []
    for i, j in zip(ii[order] + 1, jj[order] + 1):
        w = xr[j] + 1j * yi[i]; ok = False
        for _ in range(25):
            h = 1e-7 * abs(w)
            step = g(w) / ((g(w + h) - g(w - h)) / (2 * h)); w = w - step
            if abs(step) < 1e-11 * abs(w):
                ok = True; break
        if ok and not any(abs(w - q) < 1e-7 * abs(w) for q in poles):
            poles.append(w)
    return np.array(poles)


if __name__ == '__main__':
    t_start = time.time()
    s = LONGHI; eta = s['eta']; wc = cf.omega_c(eta); a, b = wc.real, abs(wc.imag)
    meta = dict(longhi={k: float(v) for k, v in s.items()}, dt_kept=DT * KEEP, t0=T0, t_on=T_ON,
                thetas=THETAS.tolist())
    try:
        meta['git'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=_HERE,
                                              text=True).strip()
    except Exception:
        meta['git'] = 'n/a'
    data = {}

    # ---- why Theta = 3 k0: the ideal staircase at w_c (and at real w_r) for Theta = 2, 2.5, 3 k0 ----
    meta['theta_scan'] = {}
    for th in (2.0, 2.5, 3.0):
        q = dict(s, theta=th * K0)
        _, chi_q, d_q = cf.discretize(longhi_target(**q), -s['Y'], s['Y'], s['N'])
        row = {}
        for lab, w in (('w_c', wc), ('w_r', a + 0j)):
            o = cf.rt(1 + chi_q, d_q, [w])
            row[lab] = dict(RL=abs(o['rL'][0]) ** 2, RR=abs(o['rR'][0]) ** 2, tau_m1=abs(o['tauL'][0] - 1))
        meta['theta_scan'][f'{th:g}'] = row
        print(f'Theta = {th:g} k0 (ideal staircase): at w_c R_L={row["w_c"]["RL"]:.2e} R_R={row["w_c"]["RR"]:.2e} '
              f'|tau-1|={row["w_c"]["tau_m1"]:.1e}; at real w_r R_L={row["w_r"]["RL"]:.2e} R_R={row["w_r"]["RR"]:.2e}',
              flush=True)

    # ---- the structure ---------------------------------------------------------------------------
    x, chi, d = cf.discretize(longhi_target(**s), -s['Y'], s['Y'], s['N'])
    L = d.sum()
    D = {'lorentz': cf.lorentz_design(chi, wc, 0.5 * cf.kappa_max(chi, wc)), 'fano2': cf.fano2_design(chi, wc)}
    assert D['lorentz']['ok']
    data['x'] = x; data['chi'] = chi; data['L'] = L
    xf = np.linspace(-s['Y'], s['Y'], 6001); data['xf'] = xf; data['chif'] = longhi_target(**s)(xf)
    data['fano2_eps_real'] = cf.eps_fano2([a], D['fano2'])[0]
    data['lorentz_eps_real'] = cf.eps_lorentz([a], D['lorentz'])[0]
    oi = cf.rt(1 + chi, d, [wc])
    meta['ideal'] = dict(RL=abs(oi['rL'][0]) ** 2, RR=abs(oi['rR'][0]) ** 2, T=abs(oi['tauL'][0]) ** 2,
                         tau_m1=abs(oi['tauL'][0] - 1))
    print(f'L = {L:.3f} lam_r, {s["N"]} layers; ideal at w_c: R_L={meta["ideal"]["RL"]:.2e} '
          f'R_R={meta["ideal"]["RR"]:.2e} |tau-1|={meta["ideal"]["tau_m1"]:.1e}', flush=True)

    xr = a * np.linspace(0.8, 1.2, 201); yi = a * np.linspace(-0.30, 0.006, 307)
    data['map_x'] = xr / a; data['map_y'] = yi / a
    xz = a * np.linspace(0.95, 1.05, 301); yz = a * np.linspace(-0.09, 0.005, 286)
    data['zoom_x'] = xz / a; data['zoom_y'] = yz / a
    meta['media'] = {}
    for name, p in D.items():
        oc = cf.rt(MODELS[name]([wc], p), d, [wc])
        wre = a * np.linspace(0.01, 5, 8001)
        m = dict(RL=abs(oc['rL'][0]) ** 2, RR=abs(oc['rR'][0]) ** 2, T=abs(oc['tauL'][0]) ** 2,
                 min_im_eps_real=float(MODELS[name](wre, p).imag.min()))
        if name == 'fano2':
            m.update(w0_range=[float(p['w0'].min() / a), float(p['w0'].max() / a)],
                     wp1_max=float(np.sqrt(p['f1'].max()) / a), wp2_max=float(np.sqrt(p['f2'].max()) / a),
                     margin=float(p['margin']))
        else:
            m.update(gam_over_2b_min=float((p['gam'] / (2 * b)).min()))
        Z = logabs_t(name, p, d, xr, yi); data[f'map_t_{name}'] = Z
        if name == 'lorentz':     # its stack poles crowd around w_c: find them on a fine zoom map
            data['zoom_t_lorentz'] = Zz = logabs_t(name, p, d, xz, yz)
            sp = stack_poles(name, p, d, xz, yz, Zz, ymin=-0.06 * a)
        else:
            sp = stack_poles(name, p, d, xr, yi, Z, ymin=-0.15 * a)
        act = np.flatnonzero(np.abs(chi) > 1e-3)
        mp = np.concatenate([material_poles(name, p, j) for j in act])
        data[f'matpoles_{name}'] = mp[mp.real > 0] / a
        data[f'stackpoles_{name}'] = sp / a
        m['highest_stack_pole'] = [float(sp[np.argmax(sp.imag)].real / a), float(sp.imag.max() / a)] if sp.size else None
        print(f'{name}: at w_c R_L={m["RL"]:.2e} R_R={m["RR"]:.2e} T={m["T"]:.6f}; min Im eps (real w)='
              f'{m["min_im_eps_real"]:.2e}; highest stack pole: {m["highest_stack_pole"]}', flush=True)

        tt, r = td_all(name, p, d, eta)
        data['t'] = tt
        for k, v in r.items():
            data[f'{name}_{k}'] = v
        for tr in (40.0, 60.0, 85.0):
            i = np.searchsorted(tt, tr)
            m[f'td_t{tr:g}'] = dict(RL=float(r['rL'][i] ** 2), RR=float(r['rR'][i] ** 2), T=float(r['tauL'][i] ** 2))
        print(f'  time domain at t=85: R_L={m["td_t85"]["RL"]:.2e} R_R={m["td_t85"]["RR"]:.2e} T={m["td_t85"]["T"]:.6f}',
              flush=True)
        meta['media'][name] = m

    # ---- angles at w_c (frequency domain): Longhi's claim is every angle, s and p -------------------
    for pol in ('s', 'p'):
        rows = []
        for th in THETAS:
            o1 = cf.rt(1 + chi, d, [wc], theta=th, pol=pol)
            o2 = cf.rt(cf.eps_fano2([wc], D['fano2']), d, [wc], theta=th, pol=pol)
            rows.append([abs(o1['rL'][0]) ** 2, abs(o1['rR'][0]) ** 2, abs(o1['tauL'][0] - 1),
                         abs(o2['rL'][0]) ** 2, abs(o2['rR'][0]) ** 2, abs(o2['tauL'][0] - 1)])
        data[f'ang_{pol}'] = np.array(rows)
        A_ = data[f'ang_{pol}']
        print(f'angles {pol} (two-line, 0-89 deg): max R_L={A_[:, 3].max():.2e} max R_R={A_[:, 4].max():.2e} '
              f'max |tau-1|={A_[:, 5].max():.2e}', flush=True)

    os.makedirs(os.path.join(_HERE, 'data'), exist_ok=True)
    np.savez(os.path.join(_HERE, 'data', 'longhi_bidir.npz'), meta=json.dumps(meta, default=float), **data)
    print(f'saved data/longhi_bidir.npz ({time.time() - t_start:.0f} s)')
