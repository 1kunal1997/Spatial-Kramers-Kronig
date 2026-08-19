"""handA: independent check that the 4-atom staircase (lam-family) has eps'' >= 0.

INDEPENDENCE CHOICES (deliberately different from the rest of this directory):
  * smoothing kernel = Gaussian CDF (erf) step, NOT raised cosine, NOT tanh.
  * standalone: imports ONLY numpy/scipy + production tmm_helper.ht_derivative.
  * both grid conventions tested (endpoint-excluded periodic vs np.linspace).
"""
import sys, json
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from scipy.special import erf
import tmm_helper as tmm_h

L = 1.0
T = np.array([-0.40, -0.20, 0.20, 0.90])


def weights(lam):
    return np.array([lam * 0.8968, (1 - lam) * 0.5, (1 - lam) * 0.5, lam * 0.1032])


def M1(lam):
    return float(np.sum(weights(lam) * np.tan(np.pi * T / (2 * L))))


def grid(N, kind):
    if kind == "periodic":          # endpoint EXCLUDED, spacing 2L/N
        return -L + 2.0 * L * np.arange(N) / N
    elif kind == "linspace":        # np.linspace(-L, L, N): endpoint DUPLICATED under FFT
        return np.linspace(-L, L, N)
    raise ValueError(kind)


def eps_re(x, lam, s):
    """Descending staircase: erf-smoothed downward steps of height w_i at t_i."""
    w = weights(lam)
    out = np.zeros_like(x)
    for ti, wi in zip(T, w):
        out -= wi * 0.5 * (1.0 + erf((x - ti) / (s * np.sqrt(2.0))))
    return out


def analyse(lam, N, kind, s, excl, edge=0.9999):
    x = grid(N, kind)
    e_im = tmm_h.ht_derivative(x, eps_re(x, lam, s))
    keep = np.abs(x) <= edge * L
    for ti in T:
        keep &= np.abs(x - ti) > excl
    if not keep.any():
        return None
    xk, yk = x[keep], e_im[keep]
    j = int(np.argmin(yk))
    return dict(min=float(yk[j]), argmin=float(xk[j]), npts=int(keep.sum()),
                max=float(yk.max()))


if __name__ == "__main__":
    print("== M1 = sum_i w_i tan(pi t_i / 2L) ==")
    for lam in (0.1, 0.2, 0.5, 1.0):
        print(f"  lam={lam:<4} weights={np.array2string(weights(lam), precision=6)}"
              f"  M1={M1(lam): .6e}")

    Ns = [1024, 4096, 16384, 10000, 6250]   # factor 16 span + two non-powers-of-two
    s_default, excl = 0.01, 0.06
    rows = []
    print(f"\n== eps'' minimum (erf smoothing s={s_default}, "
          f"step exclusion |x-t_i|>{excl}, |x|<=0.9999L) ==")
    hdr = "  lam       N      grid       min eps''    argmin   max eps''"
    print(hdr); print("-" * len(hdr))
    for lam in (0.1, 0.2, 0.5, 1.0):
        for N in Ns:
            for kind in ("periodic", "linspace"):
                r = analyse(lam, N, kind, s_default, excl)
                r.update(lam=lam, N=N, kind=kind, s=s_default, excl=excl)
                rows.append(r)
                print(f"{lam:>5} {N:>7} {kind:>9} {r['min']:>13.3e} "
                      f"{r['argmin']:>9.4f} {r['max']:>10.3e}")

    # smoothing-width sensitivity at lam=0.2 (does the sign survive sharper steps?)
    print(f"\n== lam=0.2, smoothing-width sweep (periodic grid N=16384, excl={excl}) ==")
    for s in (0.02, 0.01, 0.005, 0.0025):
        r = analyse(0.2, 16384, "periodic", s, excl)
        rows.append(dict(r, lam=0.2, N=16384, kind="periodic", s=s, excl=excl))
        print(f"  s={s:<7} min={r['min']:>12.3e} at x={r['argmin']:>8.4f}")

    # exclusion-radius sensitivity: is any negativity just the log spike leaking in?
    print("\n== lam=0.2, exclusion-radius sweep (periodic N=16384, s=0.01) ==")
    for ex in (0.02, 0.04, 0.06, 0.10, 0.15):
        r = analyse(0.2, 16384, "periodic", 0.01, ex)
        print(f"  excl={ex:<6} min={r['min']:>12.3e} at x={r['argmin']:>8.4f}")

    # edge sensitivity: separate genuine interior sign from |x|->L grid noise
    print("\n== lam=0.2, edge-cut sweep (periodic N=16384, s=0.01, excl=0.06) ==")
    for ed in (0.9999, 0.99, 0.98, 0.95):
        r = analyse(0.2, 16384, "periodic", 0.01, 0.06, edge=ed)
        print(f"  |x|<={ed:<7} min={r['min']:>12.3e} at x={r['argmin']:>8.4f}")

    json.dump(rows, open("/Users/kunal/Documents/Spatial KK Project/research/results/"
                         "cycle3_c25_hardening/handA_results.json", "w"), indent=1)
