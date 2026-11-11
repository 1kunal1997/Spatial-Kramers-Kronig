"""Independent time-domain check by direct 1-D FDTD (no Fourier synthesis) for layered media made
of up to TWO coupled oscillators per cell (B7, 2026-10-12): Long et al. single Lorentz, the EIT
(bright + dark) medium, and the two-line (broad minus narrow) medium of cfreq_lib.

Per cell, oscillators i = 1, 2 with polarisation-like coordinates P_i, J_i = dP_i/dt:
    dJ_i/dt = -g_i J_i - w0i^2 P_i - kap P_j + F_i E,      P = s1 P1 + s2 P2  (D = E + P)
  single Lorentz: F1 = wp2, osc 2 off;  EIT: F1 = wp2, F2 = 0, s2 = 0, kap = kappa (dark x2);
  two-line: F1 = f1, F2 = -f2, s1 = s2 = 1, kap = 0.
Fields are COMPLEX and the source is the complex drive ramp * e^{-i w_c t}, so one run gives the
complex output (the update has real coefficients, so this equals the cos run - i * the sin run).
Units: c = 1, lam_r = 1, period 1.
  python theory/cfreq_skk/fdtd_twopole.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cfreq_lib as cf

DX = 1 / 400; DT = 0.5 * DX


def cells_from_layers(model, p, d, x0):
    """Per-cell oscillator arrays for a layered structure starting at x0 (layers = whole cells)."""
    n = np.rint(np.asarray(d) / DX).astype(int)
    assert np.allclose(n * DX, d), 'layer thicknesses must be whole FDTD cells'
    rep = lambda v: np.repeat(np.broadcast_to(np.asarray(v, float), len(d)), n)
    z = np.zeros(n.sum())
    if model == 'lorentz':
        c = dict(F1=rep(p['wp2']), w1=rep(p['w02']), g1=rep(p['gam']), F2=z, w2=z, g2=z, kap=z, s2=0.0)
    elif model == 'eit':
        c = dict(F1=rep(p['wp2']), w1=rep(p['w0']) ** 2, g1=rep(p['g1']), F2=z,
                 w2=(rep(p['w0']) + p['det']) ** 2, g2=rep(p['g2']), kap=rep(p['kappa']), s2=0.0)
    elif model == 'fano2':
        c = dict(F1=rep(p['f1']), w1=rep(p['w0']) ** 2, g1=rep(p['g1']), F2=-rep(p['f2']),
                 w2=rep(p['w0']) ** 2, g2=rep(p['g2']), kap=z, s2=1.0)
    c['i0'] = int(round(x0 / DX)); c['n'] = n.sum()
    return c


def run(cells, eta, t_end, x_src, probes, half=None, t_on=10.0, t0=2.0, mirror=False):
    """Complex 1-D FDTD. cells: dict from cells_from_layers (x measured from the grid origin at
    -half) or None for vacuum. The source sits at x_src; probes: list of x positions.
    mirror=True flips the structure (incidence from the right without moving the probes)."""
    half = half or (t_end + t0 + 10)
    X = np.arange(-half, half, DX); n = X.size
    E = np.zeros(n, complex); H = np.zeros(n - 1, complex)
    isrc = np.searchsorted(X, x_src); ip = [np.searchsorted(X, x) for x in probes]
    if cells is not None:
        i0 = np.searchsorted(X, cells['x0'])
        sl = slice(i0, i0 + cells['n'])
        q = {k: (v[::-1] if (mirror and np.ndim(v)) else v) for k, v in cells.items()}
        P1 = np.zeros(cells['n'], complex); J1 = np.zeros_like(P1)
        P2 = np.zeros_like(P1); J2 = np.zeros_like(P1)
        a1 = (1 - 0.5 * q['g1'] * DT) / (1 + 0.5 * q['g1'] * DT); b1 = DT / (1 + 0.5 * q['g1'] * DT)
        a2 = (1 - 0.5 * q['g2'] * DT) / (1 + 0.5 * q['g2'] * DT); b2 = DT / (1 + 0.5 * q['g2'] * DT)
    steps = int((t_end + t0) / DT)
    rec = np.empty((steps, len(ip)), complex)
    tt = np.arange(steps) * DT
    drive = (cf.decaying_drive(tt, eta, t_on, t0) - 1j * cf.decaying_drive(tt, eta, t_on, t0, quadrature=True))
    for k in range(steps):
        H += DT / DX * (E[1:] - E[:-1])
        if cells is not None:
            Es = E[sl]
            J1n = a1 * J1 + b1 * (q['F1'] * Es - q['w1'] * P1 - q['kap'] * P2)
            J2n = a2 * J2 + b2 * (q['F2'] * Es - q['w2'] * P2 - q['kap'] * P1)
            J1, J2 = J1n, J2n
            P1 += DT * J1; P2 += DT * J2
            E[1:-1] += DT / DX * (H[1:] - H[:-1])
            E[sl] -= DT * (J1 + q['s2'] * J2)
        else:
            E[1:-1] += DT / DX * (H[1:] - H[:-1])
        E[isrc] += DT * drive[k]
        rec[k] = E[ip]
    return tt - t0, rec, drive


if __name__ == '__main__':
    import time, json
    eta = 0.05; wc = cf.omega_c(eta)
    dl, CM, Y, N = 0.2, 0.5, 2.0, 400
    tgt = lambda y: CM * dl ** 4 / (y + 1j * dl) ** 4 * cf.cos2_window(y, Y)
    y, chi, d = cf.discretize(tgt, -Y, Y, N)
    designs = {'eit': cf.eit_design(chi, wc), 'fano2': cf.fano2_design(chi, wc),
               'lorentz': cf.lorentz_design(chi, wc, 0.5 * cf.kappa_max(chi, wc))}
    T_END, XS, XP, XQ = 70.0, -2.75, -2.5, 2.5           # source, reflection probe, transmission probe
    t, ref, drv = run(None, eta, T_END, XS, [XP, XQ])
    D = (-Y) - XP                                        # probe -> entry face distance
    shift = int(round(2 * D / DT))
    out = {}
    for name, p in designs.items():
        t_ = time.time()
        cells = cells_from_layers(name, p, d, -Y); cells['x0'] = -Y
        _, rec, _ = run(cells, eta, T_END, XS, [XP, XQ])
        refl = rec[:, 0] - ref[:, 0]
        rL = np.full(t.size, np.nan, complex); rL[shift:] = refl[shift:] / ref[:-shift, 0]
        tau = rec[:, 1] / ref[:, 1]
        oc = cf.rt({'eit': cf.eps_eit, 'fano2': cf.eps_fano2, 'lorentz': cf.eps_lorentz}[name]([wc], p), d, [wc])
        marks = [10, 20, 30, 40, 50, 60]
        sel = [np.searchsorted(t, m) for m in marks]
        print(f'{name}: |rL ratio| at t={marks}: ' + ' '.join(f'{abs(rL[i]):.3g}' for i in sel)
              + f'  (TMM {abs(oc["rL"][0]):.3g});  |tau| ' + ' '.join(f'{abs(tau[i]):.4g}' for i in sel)
              + f'  (TMM {abs(oc["tauL"][0]):.4g})  [{time.time() - t_:.0f}s]', flush=True)
        # time axes aligned to the ENTRY-FACE drive time, as in the Fourier synthesis: the reflection
        # probe sees the face (-Y - XS) + D after the source; the transmission probe XQ - XS after
        out[name] = dict(t_rL=(t - (-Y - XS) - D)[::40], rL=rL[::40], t_tau=(t - (XQ - XS))[::40],
                         tau=tau[::40])
    # B7: homogeneous 1-lam slab with eps(w_c) = 1 - 0.3i (B3a's target), three media
    chi1 = np.array([-0.3j]); d1 = np.array([1.0]); b = abs(wc.imag)
    slab = {'lorentz': cf.lorentz_design_gamma(chi1, wc, 0.5 * 2 * b), 'eit': cf.eit_design(chi1, wc),
            'fano2': cf.fano2_design(chi1, wc)}
    XS1, XQ1 = -1.0, 2.0
    t1, ref1, _ = run(None, eta, T_END, XS1, [XQ1])
    for name, p in slab.items():
        cells = cells_from_layers(name, p, d1, 0.0); cells['x0'] = 0.0
        _, rec, _ = run(cells, eta, T_END, XS1, [XQ1])
        tau_c = cf.rt({'eit': cf.eps_eit, 'fano2': cf.eps_fano2, 'lorentz': cf.eps_lorentz}[name]([wc], p),
                      d1, [wc])['tauL'][0]
        r = np.abs(rec[:, 0] / ref1[:, 0]) / abs(tau_c)
        sel = [np.searchsorted(t1, m) for m in (10, 20, 30, 40, 50, 60)]
        print(f'slab {name}: |tau(t)|/|tau(w_c)| at t=10..60: ' + ' '.join(f'{r[i]:.4g}' for i in sel), flush=True)
        out[f'slab_{name}'] = dict(t=(t1 - (XQ1 - XS1))[::40], tau=r[::40])   # entry-face time
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), exist_ok=True)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fdtd_twopole.npz'),
             **{f'{k}_{q}': v[q] for k, v in out.items() for q in v})
