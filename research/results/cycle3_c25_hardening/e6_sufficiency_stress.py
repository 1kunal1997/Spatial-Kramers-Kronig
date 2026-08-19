"""E6 -- adversarial stress of the SURVIVING (box-quantified) sufficiency half:
   g>=0, M1=0, supp g in [-a,b], a+b <= L   ==>   eps'' >= 0.
Since eps'' is LINEAR in g and F(g)=min_x eps''(x;g) is therefore CONCAVE in g, the
minimum of F over the polytope {g>=0, mass=1, M1=0} is attained at an extreme point.
So random interior measures are weak tests; the sharp test is many-atom measures and
especially measures loaded at the corners.  We do both, plus a direct LP-style search.
"""
import numpy as np
from common import *
rng = np.random.default_rng(20260819)
L = 1.0
xg = periodic_grid(1 << 17, L)

def balanced_random(a, b, natoms, rng):
    """random g>=0 with M1=0, supp in [-a,b]: draw atoms, solve for weights >=0."""
    for _ in range(200):
        tm = -rng.uniform(1e-3, a, size=natoms)
        tp = rng.uniform(1e-3, b, size=natoms)
        wm = rng.uniform(0.05, 1, size=natoms)
        wp = rng.uniform(0.05, 1, size=natoms)
        Sm = (wm * np.tan(np.pi * tm / 2)).sum()      # negative
        Sp = (wp * np.tan(np.pi * tp / 2)).sum()      # positive
        c = -Sm / Sp
        atoms = [(t, w) for t, w in zip(tm, wm)] + [(t, c * w) for t, w in zip(tp, wp)]
        s = sum(w for _, w in atoms)
        return [(t, w / s) for t, w in atoms]

print("=== A. random many-atom balanced measures inside boxes with a+b <= L ===")
worst = 0.0; worst_cfg = None
for (a, b) in [(0.5,0.5),(0.25,0.75),(0.1,0.9),(0.4,0.6),(0.75,0.25),(0.02,0.98),
               (0.499,0.499),(0.3,0.7),(0.05,0.95),(0.9,0.1)]:
    mn_all = []
    for natoms in (1,2,3,5,10,25):
        for trial in range(60):
            atoms = balanced_random(a, b, natoms, rng)
            y = exact_kernel(xg, atoms, L)
            m, xm = interior_min(xg, y, L, 0.99999, [t for t,_ in atoms], 2e-3)
            mn_all.append(m)
            if m < worst: worst, worst_cfg = m, (a,b,natoms,trial)
    mn_all = np.array(mn_all)
    print(f" box [-{a},{b}] a+b={a+b:.3f}  n=360  min over trials = {mn_all.min():+.3e}  "
          f"#negative = {(mn_all < -1e-12).sum()}")
print(f" GLOBAL worst = {worst:+.3e}  cfg={worst_cfg}")

print("\n=== B. corner-loaded: ALL mass at the two corners (the extreme point) ===")
def corner_pair(a, b):
    wm, wp = np.tan(np.pi*b/2), np.tan(np.pi*a/2)
    s = wm+wp
    return [(-a, wm/s), (b, wp/s)]
for (a,b) in [(0.5,0.5),(0.25,0.75),(0.1,0.9),(0.01,0.99),(0.001,0.999),(0.4,0.6)]:
    y = exact_kernel(xg, corner_pair(a,b), L)
    m, xm = interior_min(xg, y, L, 0.99999, [-a,b], 2e-3)
    print(f" a={a:.3f} b={b:.3f} a+b={a+b:.4f}  min eps''={m:+.6e} at x={xm:+.5f}")

print("\n=== C. E8: the boundary a+b = L EXACTLY -- passive or not? ===")
print("  Cor 4.A predicts Delta(eta*) = 0 exactly => min eps'' = 0 at an INTERIOR point.")
for (a,b) in [(0.5,0.5),(0.25,0.75),(0.1,0.9),(0.02,0.98),(0.001,0.999)]:
    at = corner_pair(a, b)
    y = exact_kernel(xg, at, L)
    m, xm = interior_min(xg, y, L, 0.99999, [-a,b], 2e-3)
    A, B = np.pi*b/2, np.pi*a/2
    eta = np.tan(A-B); xstar = 2*np.arctan(eta)/np.pi
    print(f"  a={a:.3f} b={b:.3f}: min={m:+.3e} at x={xm:+.5f}   predicted x*={xstar:+.5f}"
          f"   eps''(x*)={exact_kernel(np.array([xstar]),at,L)[0]:+.3e}")

print("\n=== D. a+b = L + eps : how small a violation is still detectable? ===")
for eps in (1e-1,1e-2,1e-3,1e-4,1e-5,1e-6):
    a = 0.5; b = 0.5+eps
    at = corner_pair(a,b)
    y = exact_kernel(xg, at, L)
    m,xm = interior_min(xg, y, L, 0.99999, [-a,b], 2e-3)
    print(f"  a+b = L+{eps:.0e}:  min eps'' = {m:+.4e}   (scaling ~ eps^? )")
