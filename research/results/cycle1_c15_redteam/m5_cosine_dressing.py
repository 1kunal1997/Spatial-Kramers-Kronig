"""M5: cosine dressing (claim C14 / potential_criterion.py part_F).

Prior-session evidence (theory/passivity_2026Aug/potential_criterion.py,
part_F) claimed: dressing a passive logistic base with a*cos(m*pi*x/L),
localised near (not at) the edge via a smooth window, then re-balancing M1
exactly with a centred compensator bump, gives min eps'' ~ -0.20
INDEPENDENT of m for m in {8,16,32,64}, while low-order Legendre moments of
eps' change by < 1e-4 by m=64 (Riemann-Lebesgue) -- i.e. finite smooth-moment
batteries are evaded.

We reproduce this exactly (same base k=4.0/w=0.25, same window, same
compensator, same amplitude a=0.25) and EXTEND to m=4 and m=128 as asked,
at both the original N=4096 and our project N=2**14 for a resolution check.
"""
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from common import tmm_h, OUTDIR
import os

L = 2.5
NB = 1.7
EPS_B, EPS_AIR = NB**2, 1.0
D_FULL = EPS_B - EPS_AIR


def run(N, ms, a=0.25):
    xx = np.linspace(-L, L, N)

    def dihiti(e):
        v = np.imag(hilbert(np.gradient(e, xx)))
        c = cumulative_trapezoid(v, xx, initial=0)
        return c - np.linspace(c[0], c[-1], len(e))

    def M1(e):
        g = -np.gradient(e, xx)
        return np.trapezoid(g * np.tan(np.pi * xx / (2 * L)), xx)

    def sigma_mean(e):
        g = -np.gradient(e, xx)
        return np.trapezoid(g * np.log(2 * np.cos(np.pi * xx / (2 * L))), xx)

    def legendre_moments(e, K=6):
        u = xx / L
        return np.array([np.trapezoid(e * np.polynomial.legendre.Legendre.basis(k)(u), xx)
                          for k in range(K)])

    e0 = tmm_h.logistic(xx, 4.0, NB)   # w = 1/k = 0.25
    win = np.exp(-((np.abs(xx) - 1.9) / 0.25)**2)
    win *= 0.5 * (1 + np.cos(np.pi * np.clip((np.abs(xx) - 2.15) / 0.2, 0, 1)))
    m10, sig0 = M1(e0), sigma_mean(e0)
    comp = np.exp(-(xx / 0.6)**2)
    lm0 = legendre_moments(e0)

    ei0 = dihiti(e0)
    mask_int = np.abs(xx) <= 0.95 * L

    rows = []
    rows.append(dict(m=0, a=0.0, min_full=float(ei0.min()), min_interior=float(ei0[mask_int].min()),
                      dM1=0.0, dsigma=0.0, dLegendre=0.0))
    for m in ms:
        pert = a * np.cos(m * np.pi * xx / L) * win
        e = e0 + pert
        e = e - (M1(e) - m10) / (M1(e0 + comp) - m10) * comp
        ei = dihiti(e)
        dlm = float(np.abs(legendre_moments(e) - lm0).max())
        rows.append(dict(m=m, a=a, min_full=float(ei.min()), min_interior=float(ei[mask_int].min()),
                          dM1=float(M1(e) - m10), dsigma=float(sigma_mean(e) - sig0), dLegendre=dlm))
    return rows


ms = [4, 8, 16, 32, 64, 128]

print("=== N = 4096 (prior-session resolution) ===")
rows_4096 = run(4096, ms)
for r in rows_4096:
    print(f"  m={r['m']:>4} a={r['a']:.2f}: min eps''(full)={r['min_full']:+.5f}  "
          f"min eps''(interior)={r['min_interior']:+.5f}  dM1={r['dM1']:+.1e}  "
          f"dsigma={r['dsigma']:+.1e}  maxdLegendre={r['dLegendre']:.1e}")

print("\n=== N = 2**14 (project resolution check) ===")
rows_16k = run(2**14, ms)
for r in rows_16k:
    print(f"  m={r['m']:>4} a={r['a']:.2f}: min eps''(full)={r['min_full']:+.5f}  "
          f"min eps''(interior)={r['min_interior']:+.5f}  dM1={r['dM1']:+.1e}  "
          f"dsigma={r['dsigma']:+.1e}  maxdLegendre={r['dLegendre']:.1e}")

vals_full = [r['min_full'] for r in rows_4096 if r['m'] != 0]
vals_int = [r['min_interior'] for r in rows_4096 if r['m'] != 0]
spread_full = max(vals_full) - min(vals_full)
spread_int = max(vals_int) - min(vals_int)
print(f"\n[N=4096] spread of min eps''(full) across m: {spread_full:.5f}  "
      f"(values {vals_full})")
print(f"[N=4096] spread of min eps''(interior) across m: {spread_int:.5f}  "
      f"(values {vals_int})")

vals_full_16k = [r['min_full'] for r in rows_16k if r['m'] != 0]
vals_int_16k = [r['min_interior'] for r in rows_16k if r['m'] != 0]
print(f"[N=16384] spread of min eps''(full) across m: {max(vals_full_16k)-min(vals_full_16k):.5f}  "
      f"(values {vals_full_16k})")
print(f"[N=16384] spread of min eps''(interior) across m: {max(vals_int_16k)-min(vals_int_16k):.5f}  "
      f"(values {vals_int_16k})")

with open(os.path.join(OUTDIR, 'm5_summary.txt'), 'w') as f:
    f.write("N=4096 (prior-session resolution):\n")
    for r in rows_4096:
        f.write(f"  m={r['m']:>4} a={r['a']:.2f}: min_full={r['min_full']:+.5f} "
                f"min_interior={r['min_interior']:+.5f} dM1={r['dM1']:+.1e} "
                f"dsigma={r['dsigma']:+.1e} maxdLegendre={r['dLegendre']:.1e}\n")
    f.write(f"spread(min_full)={spread_full:.5f}  spread(min_interior)={spread_int:.5f}\n\n")
    f.write("N=2**14:\n")
    for r in rows_16k:
        f.write(f"  m={r['m']:>4} a={r['a']:.2f}: min_full={r['min_full']:+.5f} "
                f"min_interior={r['min_interior']:+.5f} dM1={r['dM1']:+.1e} "
                f"dsigma={r['dsigma']:+.1e} maxdLegendre={r['dLegendre']:.1e}\n")
    f.write(f"spread(min_full)={max(vals_full_16k)-min(vals_full_16k):.5f}  "
            f"spread(min_interior)={max(vals_int_16k)-min(vals_int_16k):.5f}\n")

print("\nwrote m5_summary.txt")
