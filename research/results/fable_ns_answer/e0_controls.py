"""E0 — controls. Every number later in this session rests on these evaluators.

C0.1  Cor 4.A threshold: bisect b at fixed a on the exact node test -> expect a+b* = L.
C0.2  Threshold depth law (C40): slope of min eps'' vs (a+b-L) -> -(D/2) sin(pi a/L)/L.
      Reference values (red-team e12): (.5,.5) .50000, (.25,.75) .35355, (.1,.9) .15451.
C0.3  Node test vs PRODUCTION ht_derivative on smoothed staircases (passive + gain cases),
      N = 2^15, 2^17 and non-power-of-two 60000, both grid conventions.
C0.4  Red-team lambda* reproduction: corner pair (-0.4,+0.9) + bulk pair (+-0.1):
      expect lambda* ~ 0.262; bulk (+-0.2): ~ 0.278 (handA). Independent route:
      closed-form mixture minimization, no bisection on full evaluators.
"""
import numpy as np
from common import (Atomic, balanced_pair, pair_eta_star, pair_gap_star,
                    exact_sup_gap, exact_min_eps2, production_min_eps2)

print("=" * 70)
print("C0.1 Cor 4.A threshold by bisection on the node test")
for a in (0.25, 0.10, 0.40):
    lo, hi = 0.5, 1.0 - 1e-9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        sup, _ = exact_sup_gap(balanced_pair(a, mid))
        if sup > 0:
            hi = mid
        else:
            lo = mid
    print(f"  a={a:.2f}: b* = {0.5*(lo+hi):.9f}   a+b* = {a+0.5*(lo+hi):.9f}  (expect 1)")

print("=" * 70)
print("C0.2 threshold depth slope vs C40 closed form")
for (a, b) in [(0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]:
    eps = 1e-4
    m = balanced_pair(a * (1), b + eps)  # push excess entirely into b
    v, _ = exact_min_eps2(m)
    slope = v / eps
    print(f"  (a,b)=({a},{b}): measured {slope:+.5f}   analytic {-0.5*np.sin(np.pi*a):+.5f}")

print("=" * 70)
print("C0.3 node test vs production ht_derivative")
cases = {
    "bad pair (0.3,0.8)  GAIN": balanced_pair(0.3, 0.8),
    "good pair (0.3,0.6) PASS": balanced_pair(0.3, 0.6),
    "rt mix lam=0.5 GAIN": balanced_pair(0.4, 0.9, 0.5).mix(balanced_pair(0.2, 0.2, 0.5)),
    "rt mix lam=0.2 PASS": balanced_pair(0.4, 0.9, 0.2).mix(balanced_pair(0.2, 0.2, 0.8)),
}
for name, m in cases.items():
    ex, xex = exact_min_eps2(m)
    row = f"  {name:26s} exact {ex:+.6e} @x={xex:+.4f} | prod:"
    for N, grid in [(2**15, "linspace"), (2**17, "linspace"), (60000, "periodic")]:
        pv, px = production_min_eps2(m, N=N, grid=grid)
        row += f" {pv:+.3e}@{px:+.3f}"
    print(row)
print("  (production min for PASS cases is a small negative quadrature floor -> see C42;")
print("   the decision quantity is the exact node value, production corroborates scale+argmin)")

print("=" * 70)
print("C0.4 lambda* via closed-form mixture minimization (independent of red-team route)")

def lam_star(corner, bulk, eta_grid=None):
    """Largest lam with lam*Dc(eta) + (1-lam)*Db(eta) <= 0 for all eta."""
    if eta_grid is None:
        eta_grid = np.tan(np.pi * 0.5 * np.linspace(-0.999999, 0.999999, 400001))
    Dc = corner.Delta(eta_grid)
    Db = bulk.Delta(eta_grid)
    mask = Dc > 0
    lam = (-Db[mask]) / (Dc[mask] - Db[mask])
    return float(np.min(lam))

corner = balanced_pair(0.4, 0.9)
for bp in (0.1, 0.2):
    bulk = balanced_pair(bp, bp)
    print(f"  corner(-0.4,+0.9) + bulk(+-{bp}): lambda* = {lam_star(corner, bulk):.4f}"
          f"   (red-team: 0.262 @ 0.1, handA: ~0.278 @ 0.2)")
