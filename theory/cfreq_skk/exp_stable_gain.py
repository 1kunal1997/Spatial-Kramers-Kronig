"""B7/B8 (2026-10-12): STABLE virtual gain from passive two-pole media, and the one-way-invisible
sKK it makes possible under a decaying (complex-frequency) drive.

B3 showed that Long et al.'s single-Lorentz media reach Im eps(w_c) < 0 only with poles that ring
longer than the drive decays, so the w_c steady state is never reached. Here the same targets are
realised with EIT (lossy bright + LOSSLESS dark oscillator) and two-line (broad minus narrow)
media whose every material pole decays >= 2x faster than the drive (cfreq_lib.eit_design,
fano2_design), and the structures are run in the time domain.

Writes data/stable_gain.npz (+ meta JSON). FDTD cross-checks: fdtd_twopole.py. Run:
  python theory/cfreq_skk/exp_stable_gain.py      (~3 min)
"""
import sys, os, json, time, subprocess, warnings
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import cfreq_lib as cf

warnings.filterwarnings('ignore', category=RuntimeWarning)   # ratios divide by the (zero) drive before t0
DT, NT = 1 / 32, 2 ** 17
T0, T_ON = 2.0, 10.0
EPS_SLAB = 1 - 0.3j                     # B3a's virtual-gain slab target, 1 lam_r thick
SIMPLE = dict(delta=0.2, chimax=0.5, Y=2.0, N=400)               # single 4th-order KHP pole
B1 = dict(n_poles=6, x_span=1.0, delta_range=(0.12, 0.3), chimax=0.8, seed=1, order=4, Y=2.5, N=500)
THETAS = np.concatenate([np.arange(0, 88, 1.0), [88, 89, 89.5]])
KEEP = 8                                # keep every 8th time sample (dt = 1/4 period) in the npz
T_READ = {0.05: 150.0, 0.01: 800.0}     # settled-value readout time per eta (well inside the
                                        # double-precision range of the shifted synthesis)

MODELS = {'lorentz': cf.eps_lorentz, 'eit': cf.eps_eit, 'fano2': cf.eps_fano2}


def designs(chi, wc):
    b = abs(wc.imag)
    lor = (cf.lorentz_design_gamma(chi, wc, 0.5 * 2 * b) if chi.size == 1
           else cf.lorentz_design(chi, wc, 0.5 * cf.kappa_max(chi, wc)))
    return {'lorentz': lor, 'eit': cf.eit_design(chi, wc), 'fano2': cf.fano2_design(chi, wc)}


def simple_target(y):
    s = SIMPLE
    return s['chimax'] * s['delta'] ** 4 / (y + 1j * s['delta']) ** 4 * cf.cos2_window(y, s['Y'])


def b1_target():
    poles, amps = cf.random_khp(B1['n_poles'], B1['x_span'], B1['delta_range'], B1['chimax'],
                                seed=B1['seed'], order=B1['order'])
    return lambda y: cf.pole_profile(y, poles, amps, B1['order']) * cf.cos2_window(y, B1['Y'])


def td_ratios(name, p, d, eta, keys=('rL', 'rR', 'tauL'), **kw):
    """|output / drive| vs time for each key; tauL = transmission / free propagation. Stable media
    use the contour shifted to 0.9|b| (no pole is crossed: all lie below -2|b|); the single-Lorentz
    structure has poles ABOVE w_c, so it uses the unshifted contour."""
    wc = cf.omega_c(eta); b = abs(wc.imag); L = d.sum()
    t = np.arange(NT) * DT
    csh = 0.0 if name == 'lorentz' else 0.9 * b
    out = {}
    for key in keys:
        src = 'tL' if key == 'tauL' else key
        fn = (lambda w, src=src: cf.rt_model(MODELS[name], p, d, w, **kw)[src]
              * (np.exp(-1j * w * L) if src == 'tL' else 1))
        dcx, o = cf.complex_response_shifted(fn, t, eta, T_ON, T0, csh)
        out[key] = np.abs(o / dcx)[::KEEP]
    return (t - T0)[::KEEP], out


def cmap_logabs(name, p, d, key, xr, yi):
    W = (xr[None, :] + 1j * yi[:, None]).ravel()
    return np.log10(np.abs(cf.rt_model(MODELS[name], p, d, W)[key])).reshape(yi.size, xr.size)


def material_poles(name, p, j=0):
    if name == 'lorentz':
        return np.roots([-1, -1j * p['gam'][j], p['w02'][j]])
    if name == 'eit':
        return cf.eit_poles(p['w0'][j], p)
    return np.concatenate([np.roots([-1, -1j * g, p['w0'][j] ** 2]) for g in (p['g1'], p['g2'])])


if __name__ == '__main__':
    t_start = time.time()
    data, meta = {}, dict(dt_kept=DT * KEEP, t0=T0, t_on=T_ON, eps_slab=[EPS_SLAB.real, EPS_SLAB.imag],
                          simple=SIMPLE, b1={k: v for k, v in B1.items()}, thetas=THETAS.tolist())
    try:
        meta['git'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=_HERE,
                                              text=True).strip()
    except Exception:
        meta['git'] = 'n/a'

    # ---- B7: homogeneous 1-lam slab, eps(w_c) = 1 - 0.3i, three media ------------------------
    eta = 0.05; wc = cf.omega_c(eta); a, b = wc.real, abs(wc.imag)
    chi1 = np.array([EPS_SLAB - 1]); d1 = np.array([1.0])
    D = designs(chi1, wc)
    ww = a * np.linspace(0.6, 1.4, 1601)
    xr = a * np.linspace(0.85, 1.15, 241); yi = a * np.linspace(-0.17, 0.006, 177)
    data['slab_w'] = ww / a; data['map_x'] = xr / a; data['map_y'] = yi / a
    meta['slab'] = {}
    for name, p in D.items():
        e = MODELS[name](ww, p)[:, 0]
        data[f'slab_eps_{name}'] = e
        oc = cf.rt(MODELS[name]([wc], p), d1, [wc])
        mp = material_poles(name, p)
        data[f'slab_poles_{name}'] = mp / a
        data[f'slab_map_{name}'] = cmap_logabs(name, p, d1, 'tL', xr, yi)
        t, r = td_ratios(name, p, d1, eta, keys=('tauL',))
        data['slab_t'] = t
        data[f'slab_tau_{name}'] = r['tauL'] / abs(oc['tauL'][0])
        meta['slab'][name] = dict(tau_c=abs(oc['tauL'][0]), min_im_eps_real=float(
            MODELS[name](a * np.linspace(0.01, 5, 20001), p).imag.min()),
            slowest_pole_over_b=float((-mp.imag).min() / b),
            params={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()})
        print(f'slab {name}: slowest material pole / |b| = {(-mp.imag).min() / b:.2f}; '
              f'|tau(t)|/|tau_c| at t=20,40,60,100: '
              + ' '.join(f'{data[f"slab_tau_{name}"][np.searchsorted(t, x)]:.4g}' for x in (20, 40, 60, 100)),
              flush=True)

    # ---- B8: layered one-way-invisible sKK ----------------------------------------------------
    cases = {'simple_eta0.05': (simple_target, SIMPLE['Y'], SIMPLE['N'], 0.05),
             'b1_eta0.01': (b1_target(), B1['Y'], B1['N'], 0.01),
             'simple_eta0.01': (simple_target, SIMPLE['Y'], SIMPLE['N'], 0.01),
             'b1_eta0.05': (b1_target(), B1['Y'], B1['N'], 0.05)}
    meta['skk'] = {}
    for cname, (tgt, Y, N, eta) in cases.items():
        wc = cf.omega_c(eta); a, b = wc.real, abs(wc.imag)
        y, chi, d = cf.discretize(tgt, -Y, Y, N)
        D = designs(chi, wc)
        data[f'{cname}_y'] = y; data[f'{cname}_chi'] = chi
        oi = cf.rt(1 + chi, d, [wc])
        m = dict(ideal=dict(rL=abs(oi['rL'][0]), rR=abs(oi['rR'][0]), tau=abs(oi['tauL'][0]),
                            tau_m1=abs(oi['tauL'][0] - 1)))
        for name, p in D.items():
            oc = cf.rt(MODELS[name]([wc], p), d, [wc])
            t, r = td_ratios(name, p, d, eta)
            data[f'{cname}_t'] = t
            for k, v in r.items():
                data[f'{cname}_{name}_{k}'] = v
            m[name] = dict(rL=abs(oc['rL'][0]), rR=abs(oc['rR'][0]), tau=abs(oc['tauL'][0]),
                           min_im_eps_real=float(MODELS[name](a * np.linspace(0.01, 5, 8001), p).imag.min()),
                           min_margin=float(np.min(p['margin'] if 'margin' in p else p['gam'] / (2 * b))),
                           w0_range=[float(p['w0'].min() / a), float(p['w0'].max() / a)] if 'w0' in p else None)
            if name == 'eit':
                data[f'{cname}_eit_w0'] = p['w0'] / a; data[f'{cname}_eit_wp2'] = p['wp2']
                data[f'{cname}_eit_eps_real'] = cf.eps_eit([a], p)[0]
            i = np.searchsorted(t, T_READ[eta])
            m[name].update(td_rL=float(r['rL'][i]), td_rR=float(r['rR'][i]), td_tau=float(r['tauL'][i]))
            print(f'{cname} {name}: predicted |rL|={m[name]["rL"]:.3g} |rR|={m[name]["rR"]:.3g} '
                  f'|tau|={m[name]["tau"]:.4f}; time domain at t={T_READ[eta]:g}: |rL|={r["rL"][i]:.3g} '
                  f'|rR|={r["rR"][i]:.3g} |tau|={r["tauL"][i]:.4f}', flush=True)
        meta['skk'][cname] = m
        if cname == 'simple_eta0.05':
            xr = a * np.linspace(0.8, 1.2, 201); yi = a * np.linspace(-0.17, 0.006, 177)
            data['skk_map_x'] = xr / a; data['skk_map_y'] = yi / a
            for name in ('lorentz', 'eit'):
                data[f'skk_map_{name}'] = cmap_logabs(name, D[name], d, 'rR', xr, yi)
            # angles: frequency domain at w_c, s and p (ideal staircase vs EIT realisation)
            for pol in ('s', 'p'):
                rows = []
                for th in THETAS:
                    o1 = cf.rt(1 + chi, d, [wc], theta=th, pol=pol)
                    o2 = cf.rt(cf.eps_eit([wc], D['eit']), d, [wc], theta=th, pol=pol)
                    rows.append([abs(o1['rL'][0]) ** 2, abs(o1['rR'][0]) ** 2, abs(o1['tauL'][0] - 1),
                                 abs(o2['rL'][0]) ** 2, abs(o2['rR'][0]) ** 2, abs(o2['tauL'][0] - 1)])
                data[f'ang_{pol}'] = np.array(rows)
            # time-domain stability at oblique incidence (structural poles move with kx)
            meta['oblique_td'] = {}
            for th, pol in ((45.0, 's'), (45.0, 'p'), (70.0, 's')):
                t, r = td_ratios('eit', D['eit'], d, eta, keys=('rL', 'rR'), theta=th, pol=pol)
                oc = cf.rt(cf.eps_eit([wc], D['eit']), d, [wc], theta=th, pol=pol)
                meta['oblique_td'][f'{th:g}{pol}'] = dict(pred_rL=abs(oc['rL'][0]), pred_rR=abs(oc['rR'][0]),
                                                          td_rL=float(r['rL'][np.searchsorted(t, T_READ[eta])]),
                                                          td_rR=float(r['rR'][np.searchsorted(t, T_READ[eta])]),
                                                          td_rL_t100=float(r['rL'][np.searchsorted(t, 100)]))
                print(f'  oblique {th:g} {pol}: predicted |rL|={abs(oc["rL"][0]):.3g} |rR|={abs(oc["rR"][0]):.3g};'
                      f' time domain at t={T_READ[eta]:g}: |rL|={r["rL"][np.searchsorted(t, T_READ[eta])]:.3g}'
                      f' |rR|={r["rR"][np.searchsorted(t, T_READ[eta])]:.3g}', flush=True)

    os.makedirs(os.path.join(_HERE, 'data'), exist_ok=True)
    np.savez(os.path.join(_HERE, 'data', 'stable_gain.npz'), meta=json.dumps(meta, default=float), **data)
    print(f'saved data/stable_gain.npz ({time.time() - t_start:.0f} s)')
