"""Independent check of the time-domain result by direct 1-D FDTD (Lorentz ADE), no Fourier
synthesis. Homogeneous 1-lambda slab of ONE passive Lorentz medium designed (Long et al.
Theorem 1) for eps(w_c) = target; decaying drive with the same Planck turn-on. Reports the
transmitted/free ratio |tau(t)| / |tau(w_c)| at a probe behind the slab, from two runs (cos and
sin drive) with and without the slab. Units: c = 1, lam_r = 1, period 1.
  python theory/cfreq_skk/fdtd_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cfreq_lib as cf

DX = 1 / 200; DT = 0.5 * DX
X = np.arange(-70, 70, DX)
I_SRC, I_PRB = np.searchsorted(X, -3.0), np.searchsorted(X, 2.0)
SLAB = (X >= 0) & (X < 1.0)


def run(p, eta, quad, t_end=60.0, t_on=10.0, t0=2.0):
    n = X.size
    E = np.zeros(n); H = np.zeros(n - 1); P = np.zeros(n); J = np.zeros(n)
    wp2 = np.where(SLAB, p['wp2'][0], 0.0) if p else np.zeros(n)
    w02 = np.where(SLAB, p['w02'][0], 0.0) if p else np.zeros(n)
    gam = np.where(SLAB, p['gam'][0], 0.0) if p else np.zeros(n)
    steps = int((t_end + t0 + 6) / DT)
    rec = np.empty(steps)
    for k in range(steps):
        t = k * DT
        H += DT / DX * (E[1:] - E[:-1])
        # ADE (semi-implicit for the damping term): dJ/dt = -gam J - w0^2 P + wp^2 E, dP/dt = J
        J = ((1 - 0.5 * gam * DT) * J + DT * (wp2 * E - w02 * P)) / (1 + 0.5 * gam * DT)
        P += DT * J
        E[1:-1] += DT / DX * (H[1:] - H[:-1]) - DT * J[1:-1]
        E[I_SRC] += DT * cf.decaying_drive(np.array([t]), eta, t_on, t0, quadrature=quad)[0]
        rec[k] = E[I_PRB]
    return np.arange(steps) * DT, rec


if __name__ == '__main__':
    eta = 0.05; wc = cf.omega_c(eta); b = abs(wc.imag)
    d = np.full(40, 1.0 / 40)
    for tgt, g in [(1 - 0.3j, 0.5), (1 + 0.3j, 1.5)]:
        p = cf.lorentz_design_gamma(np.array([tgt - 1]), wc, g * 2 * b)
        tau_c = cf.rt(np.repeat(cf.eps_lorentz([wc], p), 40, 1), d, [wc])['tauL'][0]
        t, sc = run(p, eta, False); _, ss = run(p, eta, True)
        _, rc = run(None, eta, False); _, rs = run(None, eta, True)
        ratio = np.abs((sc - 1j * ss) / (rc - 1j * rs)) / abs(tau_c)
        tt = t - 2.0 - 5.0                    # source -> probe delay (5 lam) removed
        out = ' '.join(f'{x:g}:{ratio[np.searchsorted(tt, x)]:.3f}' for x in [5, 10, 20, 30, 40, 50, 60])
        print(f'FDTD eps(wc)={tgt} gam/2|b|={g}: |tau(t)|/|tau(wc)| at t = {out}')
