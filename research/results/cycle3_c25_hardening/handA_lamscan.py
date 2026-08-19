"""handA: lam-threshold scan + independent closed-form cross-check.

Second evaluator, derived by hand (NOT read off the numerics): for a single
downward unit step at t0 the DiHiTI partner is the periodic-log kernel
   K(x;t0) = (1/pi) * ln[ cos(pi t0/2L) / |sin(pi (x-t0)/2L)| ],
so for atoms {(t_i,w_i)}:  eps''(x) = sum_i w_i K(x;t_i).
The additive gauge is fixed by eps''(+-L)=0, which the ht_derivative detrend
also enforces.  Agreement of the two routes is the cross-check.
"""
import sys
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from handA_staircase import L, T, weights, grid, eps_re, M1
import tmm_helper as tmm_h


def kernel(x, lam):
    w = weights(lam)
    out = np.zeros_like(np.asarray(x, float))
    with np.errstate(divide='ignore'):
        for ti, wi in zip(T, w):
            out += wi * (np.log(np.cos(np.pi*ti/(2*L)))
                         - np.log(np.abs(np.sin(np.pi*(x-ti)/(2*L)))))
    return out / np.pi


def interior_min(lam, N=16384, excl=0.06, edge=0.99, use='ht', s=0.01):
    x = grid(N, "periodic")
    y = tmm_h.ht_derivative(x, eps_re(x, lam, s)) if use == 'ht' else kernel(x, lam)
    keep = np.abs(x) <= edge*L
    for ti in T:
        keep &= np.abs(x-ti) > excl
    xk, yk = x[keep], y[keep]
    j = int(np.argmin(yk))
    return float(yk[j]), float(xk[j])


print("== cross-check: production ht_derivative vs hand-derived log kernel ==")
print("   (periodic N=16384, s=0.01, |x-t_i|>0.06, |x|<=0.99)")
for lam in (0.1, 0.2, 0.3, 0.5, 1.0):
    a, xa = interior_min(lam, use='ht')
    b, xb = interior_min(lam, use='kern')
    print(f"  lam={lam:<4} ht: {a:>11.4e} @ {xa:>8.4f}   kernel: {b:>11.4e} @ {xb:>8.4f}"
          f"   |diff|={abs(a-b):.2e}")

print("\n== lam scan (both evaluators, interior min over |x|<=0.99) ==")
print("   lam    min(ht)       min(kernel)   argmin(kernel)")
for lam in np.round(np.arange(0.05, 1.001, 0.05), 3):
    a, _ = interior_min(lam, use='ht')
    b, xb = interior_min(lam, use='kern')
    flag = "  <-- NEGATIVE" if b < 0 else ""
    print(f"  {lam:>4.2f}  {a:>12.4e}  {b:>12.4e}  {xb:>8.4f}{flag}")

# bisect the sign-change threshold using the (grid-free) kernel on a dense probe
xs = np.linspace(-0.999, 0.999, 400001)
mask = np.ones_like(xs, bool)
for ti in T:
    mask &= np.abs(xs-ti) > 1e-3
def kmin(lam):
    return kernel(xs[mask], lam).min()
lo, hi = 0.05, 1.0
assert kmin(lo) > 0 >= kmin(hi), (kmin(lo), kmin(hi))
for _ in range(60):
    mid = 0.5*(lo+hi)
    if kmin(mid) > 0: lo = mid
    else: hi = mid
print(f"\n== sign-change threshold (dense kernel probe): lam* = {0.5*(lo+hi):.6f} ==")
print(f"   support of the drop measure: [a,b] = [{T.min()}, {T.max()}], "
      f"a+b (as |t_min|+t_max) = {abs(T.min())+T.max():.2f}  vs  L = {L}")
