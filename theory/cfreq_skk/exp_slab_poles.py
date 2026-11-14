"""B7 follow-ups (2026-11-14): slab poles vs material poles for the homogeneous EIT / two-line slab
with eps(w_c) = 1 - 0.3i (B7's slab), eta = 0.05.

  1. 1 lam_r vs 5 lam_r: log10|t(w)| maps and time-domain |t(t)/t_free(t)|. The 5 lam slab has
     Fabry-Perot poles ABOVE the drive line (FINDINGS: "thick virtual-gain slabs lase").
  2. Fabry-Perot order m of each 1 lam slab pole, from the exact symmetric-slab pole condition
     e^{2inkL} = ((n+1)/(n-1))^2  ->  m = (n k L + i ln((n+1)/(n-1))) / pi  (an integer at a pole;
     principal branches, so the absolute index carries the face-reflection-phase convention).
  3. log10|eps(w)| and log10|t(w)| (1 lam) over a taller window that holds every material pole.

Writes data/slab_poles.npz (+ meta). Figures: fig_stable_gain.py slab_thick, eps_map, slab.
Run: python theory/cfreq_skk/exp_slab_poles.py   (~2 min)
"""
import sys, os, json, warnings
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import cfreq_lib as cf
import exp_stable_gain as E

warnings.filterwarnings('ignore', category=RuntimeWarning)
ETA = 0.05
THICK = (1.0, 5.0)
XR, YI = (0.85, 1.15, 361), (-0.17, 0.006, 265)          # B7 map window, finer grid
YI_TALL = (-0.30, 0.006, 460)                             # holds the two-line broad pole (-0.25)
T_MAX = 100.0


def t_of(name, p, L, w):
    """t of a homogeneous slab in vacuum, closed form t = (1 - r^2) e^{iphi} / (1 - r^2 e^{2iphi}),
    r = (n-1)/(n+1), phi = n w L (c = 1); independent of the branch of n. Evaluated in the
    form with no growing exponential, because near a material pole |phi| is huge and the plain TMM
    product overflows or cancels (bright numerical blobs at 5 lam). Agrees with cf.rt elsewhere."""
    w = np.asarray(w, dtype=complex)
    n = np.sqrt(E.MODELS[name](w.ravel(), p)[:, 0]).reshape(w.shape)
    r2 = ((n - 1) / (n + 1)) ** 2
    ph = n * w * L
    ph = np.where(ph.imag < 0, -ph, ph)          # use the branch with |e^{i phi}| <= 1
    r2 = np.where((n * w * L).imag < 0, 1 / r2, r2)
    e1 = np.exp(1j * ph)
    return (1 - r2) * e1 / (1 - r2 * e1 * e1)


def grid(xr, yi, a):
    x = np.linspace(*xr); y = np.linspace(*yi)
    return x, y, (x[None, :] + 1j * y[:, None]) * a


def find_poles(name, p, L, x, y, logt, a, floor=0.3):
    """Local maxima of log|t| on the grid, refined by Newton on 1/t (t = 1/D, D analytic away
    from the material poles). Returns poles / a, highest first."""
    c = logt[1:-1, 1:-1]; m = c > floor
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                m &= c > logt[1 + dy:logt.shape[0] - 1 + dy, 1 + dx:logt.shape[1] - 1 + dx]
    iy, ix = np.nonzero(m)
    out = []
    for z in (x[ix + 1] + 1j * y[iy + 1]) * a:
        for _ in range(40):
            h = 1e-7 * a
            f0 = 1 / t_of(name, p, L, [z])[0]; f1 = 1 / t_of(name, p, L, [z + h])[0]
            dz = f0 / ((f1 - f0) / h); z -= dz
            if abs(dz) < 1e-13 * a:
                break
        if abs(1 / t_of(name, p, L, [z])[0]) < 1e-8:
            out.append(z / a)
    out = np.unique(np.round(np.array(out), 8))
    return out[np.argsort(-out.imag)]


def fp_order(name, p, L, zs, a):
    w = np.asarray(zs) * a
    n = np.sqrt(E.MODELS[name](w, p)[:, 0])
    return (n * w * L + 1j * np.log((n + 1) / (n - 1))) / np.pi


def count_above(name, p, L, lo, a, xr=(0.05, 4.0), top=0.05, n=40000):
    """Argument principle: number of poles of t in Re w/a in xr, lo < Im w/a < top (valid when
    no material pole lies inside, true for EIT/two-line with lo = -eta)."""
    x0, x1 = xr
    z = np.concatenate([np.linspace(x0, x1, n) + 1j * lo, x1 + 1j * np.linspace(lo, top, n // 8),
                        np.linspace(x1, x0, n) + 1j * top, x0 + 1j * np.linspace(top, lo, n // 8)]) * a
    ph = np.unwrap(np.angle(1 / t_of(name, p, L, z)))
    return float(np.round((ph[-1] - ph[0]) / (2 * np.pi), 3))


if __name__ == '__main__':
    wc = cf.omega_c(ETA); a, b = wc.real, abs(wc.imag)
    D = E.designs(np.array([E.EPS_SLAB - 1]), wc)
    data, meta = {}, dict(eta=ETA, thick=list(THICK), t_max=T_MAX)
    x, y, W = grid(XR, YI, a); data['map_x'], data['map_y'] = x, y
    xt, yt, Wt = grid(XR, YI_TALL, a); data['tall_x'], data['tall_y'] = xt, yt
    tt = np.arange(E.NT) * E.DT
    for nm in ('eit', 'fano2'):
        p = D[nm]; meta[nm] = {}
        data[f'mat_poles_{nm}'] = E.material_poles(nm, p) / a
        for L in THICK:
            key = f'{nm}_L{L:g}'
            logt = np.log10(np.abs(t_of(nm, p, L, W)))
            data[f'map_{key}'] = logt
            poles = find_poles(nm, p, L, x, y, logt, a)
            data[f'poles_{key}'] = poles
            data[f'm_{key}'] = fp_order(nm, p, L, poles, a)
            tau_c = abs(cf.rt(E.MODELS[nm]([wc], p), np.array([L]), [wc])['tauL'][0])
            # time domain: the 5 lam slab has poles above the drive, so the contour stays on the
            # real axis (c = 0, always valid: passive on the real axis); 1 lam as in B7 (c = 0.9b)
            c = 0.9 * b if poles[0].imag * a < -b else 0.0
            fn = lambda w, L=L: t_of(nm, p, L, w) * np.exp(-1j * w * L)
            dcx, o = cf.complex_response_shifted(fn, tt, ETA, E.T_ON, E.T0, c)
            keep = (tt - E.T0 <= T_MAX + 1)
            data['td_t'] = (tt - E.T0)[keep][::4]
            data[f'td_{key}'] = np.abs(o / dcx)[keep][::4]       # |t(t)/t_free(t)|
            n_above = count_above(nm, p, L, -ETA, a)
            meta[nm][f'L{L:g}'] = dict(tau_c=tau_c, highest_pole=[poles[0].real, poles[0].imag],
                                       n_poles_above_drive=n_above, contour_shift=c / b)
            print(f'{nm} L={L:g}: |tau(w_c)|={tau_c:.4g}, highest slab pole {poles[0]:.4f}, '
                  f'poles above drive: {n_above}, |t/t_free| at t=20,60,100: '
                  + ' '.join(f'{data[f"td_{key}"][np.searchsorted(data["td_t"], s)]:.4g}'
                             for s in (20, 60, 100)), flush=True)
        ms = data[f'm_{nm}_L1']; pl = data[f'poles_{nm}_L1']
        for z, m in sorted(zip(pl, ms), key=lambda q: q[0].real):
            print(f'   1 lam pole {z.real:.4f}{z.imag:+.4f}i   m = {m.real:+.3f}{m.imag:+.3f}i')
        data[f'tallmap_t_{nm}'] = np.log10(np.abs(t_of(nm, p, 1.0, Wt)))
        data[f'tallmap_eps_{nm}'] = np.log10(np.abs(E.MODELS[nm](Wt.ravel(), p)[:, 0])).reshape(Wt.shape)
    data['meta'] = json.dumps(meta)
    np.savez_compressed(os.path.join(_HERE, 'data', 'slab_poles.npz'), **data)
    print('saved data/slab_poles.npz')
