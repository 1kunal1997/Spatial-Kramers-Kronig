"""E1 -- KILL THE LITERAL 'iff'.

C25 as registered: "g >= 0, M1 = 0, supp g in [-a,b]  ==>  passive  IFF  a+b <= L."
Read as a per-profile statement this asserts, in particular, the NECESSITY direction:
   supp g in [-a,b] and a+b > L  ==>  NOT passive.
That is false.  Exhibit: a balanced measure whose support TOUCHES BOTH CORNERS -a and +b
of a box with a+b = 1.3 L, which is nevertheless passive.

Construction (linearity of eps'' in g is exact -- C21):
   P = lam * (balanced corner pair at -a, +b)  +  (1-lam) * (balanced pair at -s, +s)
Both pieces have M1 = 0, both are >= 0, so P is admissible; supp P = {-a,-s,s,b}, whose
convex hull is exactly [-a, b] -- the EFFECTIVE support equals the nominal box.
"""
import numpy as np
from common import *

L = 1.0
def balanced_pair(tm, tp):
    """atoms at -tm and +tp with M1 = 0, total mass 1."""
    wm = np.tan(np.pi * tp / 2); wp = np.tan(np.pi * tm / 2)
    s = wm + wp
    return [(-tm, wm / s), (tp, wp / s)]

a, b = 0.40, 0.90          # a + b = 1.30 L  -- WELL over threshold
s     = 0.20               # middle pair, 2s = 0.40 << L
corner = balanced_pair(a, b)
middle = balanced_pair(s, s)
xg = periodic_grid(1 << 18, L)

print("corner pair alone:", [(round(t,3), round(w,4)) for t,w in corner],
      " M1 =", f"{M1_atoms(corner):.2e}")
mc, xc = interior_min(xg, exact_kernel(xg, corner, L), L, 0.9999, [t for t,_ in corner], 1e-3)
print(f"   min eps'' = {mc:+.6f} at x = {xc:+.4f}   -> {'FAIL(gain)' if mc<0 else 'PASS'}")
mm, xm = interior_min(xg, exact_kernel(xg, middle, L), L, 0.9999, [t for t,_ in middle], 1e-3)
print(f"middle pair alone: min eps'' = {mm:+.6f} at x = {xm:+.4f}")

print("\nlam   min eps'' (exact kernel, |x|<=0.9999L, atoms excised)   verdict")
for lam in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
    P = [(t, lam * w) for t, w in corner] + [(t, (1 - lam) * w) for t, w in middle]
    y = exact_kernel(xg, P, L)
    m, xm_ = interior_min(xg, y, L, 0.9999, [t for t, _ in P], 1e-3)
    print(f"{lam:5.3f}  {m:+.6e}  at x={xm_:+.4f}   M1={M1_atoms(P):+.1e}   "
          f"{'FAIL' if m < 0 else 'PASS  <-- counterexample' if lam>0 else ''}")

# ---- confirm the winner through the PRODUCTION pipeline, not just the closed form ----
lam = 0.01
P = [(t, lam * w) for t, w in corner] + [(t, (1 - lam) * w) for t, w in middle]
print("\nPRODUCTION ht_derivative check on the lam=0.01 measure (smoothed steps):")
for N in (1 << 15, 1 << 17, 1 << 19):
    x, e = dihiti_of_atoms(P, N=N, L=L)
    ww = 40.0 * (2 * L / N)
    m, xm_ = interior_min(x, e, L, 0.9999, [t for t, _ in P], 3 * ww)
    print(f"   N={N:8d}  step width={ww:.2e}   min eps'' = {m:+.6e} at x={xm_:+.4f}"
          f"   -> {'PASS (passive)' if m >= 0 else 'FAIL'}")
print("\nnominal box  [-a,b] = [-0.40, 0.90],  a+b = 1.30 L > L")
print("effective support = convex hull{-0.40,-0.20,+0.20,+0.90} = [-0.40,+0.90]  (identical)")
