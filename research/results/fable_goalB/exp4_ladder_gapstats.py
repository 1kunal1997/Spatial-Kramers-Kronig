"""Goal B experiment 4 — the sector ladder and quantified gap (B3).

 (a) Ladder witness: m1*S(0.6) + m2*S(2.2)  (0.6*2.2=1.32>1: not a good pair set,
     2.2 != 1/0.6: not a reciprocal pair). Find the exact passive range in m2/m1.
     Any passive point with m2>0 lies OUTSIDE cone(good pairs) AND outside the
     reciprocal sector -> the generator ladder does not terminate at S+.
 (b) Theorem S-tau control: inversion-symmetric even measures with all radii in
     [1/4, 3/4] (tau-middle-band) must be 100% passive (proved via E1' + mixing).
     Control: a pure symmetric pair at radius < 1/4 must fail.
 (c) Gap statistics on a random even atomic ensemble:
       - among N-satisfying (outer+centre clean): fraction with gain  (N-side gap)
       - among passive: fraction certified by S = cone(good balanced pairs)
         [atomic profiles have no plateau component]                  (S-side gap)
"""
import sys
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/results/fable_ns_answer")
from common import Atomic, balanced_pair, exact_sup_gap, critical_etas, xi_of_t, t_of_xi

rng = np.random.default_rng(2468)
TOL = 1e-11

print("=" * 72)
print("(a) ladder witness: m1*S(0.6) + m2*S(2.2)")
print("=" * 72)
u1, u2 = t_of_xi(0.6), t_of_xi(2.2)
lo, hi = 0.0, 2.0
for m2 in np.linspace(0.0, 1.2, 13):
    m = Atomic([u1, -u1, u2, -u2], [0.5, 0.5, m2 / 2, m2 / 2]) if m2 > 0 else \
        Atomic([u1, -u1], [0.5, 0.5])
    s, e = exact_sup_gap(m)
    print(f"  m2={m2:.2f}: sup={s:+.4e} at eta={e:+.4f}  {'PASSIVE' if s<=1e-10 else 'GAIN'}")
# bisect the threshold
a, b = 0.0, 1.2
f = lambda m2: exact_sup_gap(Atomic([u1, -u1, u2, -u2],
                                    [0.5, 0.5, m2 / 2, m2 / 2]))[0]
if f(1.2) > 0:
    while b - a > 1e-10:
        mid = 0.5 * (a + b)
        if f(mid) <= 1e-14:
            a = mid
        else:
            b = mid
    print(f"  exact threshold: m2* = {0.5*(a+b):.8f} (passive for m2 <= m2*)")
print(f"  note: 0.6*2.2 = 1.32 > 1 (bad cross pair), 1/0.6 = {1/0.6:.4f} != 2.2 (not reciprocal)")
print(f"  => every passive point with m2 > 0 is outside cone(good pairs U reciprocal pairs)")

print("=" * 72)
print("(b) S-tau control: inversion-symmetric, radii in [1/4, 3/4]")
print("=" * 72)
n_bad = 0
for trial in range(2000):
    K = rng.integers(1, 5)
    u = rng.uniform(0.25, 0.5, K)      # inner radius of each inversion pair
    v = rng.dirichlet(np.ones(K))
    tt, ww = [], []
    for uk, vk in zip(u, v):
        for r in (uk, 1.0 - uk):       # inversion partner: u <-> 1-u, equal mass
            tt += [r, -r]; ww += [vk / 4, vk / 4]
    m = Atomic(tt, ww)
    s, e = exact_sup_gap(m)
    if s > 1e-9:
        n_bad += 1
        if n_bad <= 3:
            print(f"  VIOLATION sup={s:+.3e} u={np.round(u,4)} v={np.round(v,4)}")
print(f"  violations: {n_bad}/2000  [expect 0 — proved via E1' + mixtures]")
for u0, expect in [(0.24, "GAIN"), (0.25, "boundary~0"), (0.26, "PASSIVE")]:
    tt = [u0, -u0, 1 - u0, -(1 - u0)]
    m = Atomic(tt, [0.25] * 4)
    s, e = exact_sup_gap(m)
    print(f"  pure inversion pair u={u0}: sup={s:+.4e}  expect {expect}")

print("=" * 72)
print("(c) gap statistics, random even atomic ensemble")
print("=" * 72)

def region_sups(m: Atomic):
    etas = critical_etas(m)
    vals = m.Delta(etas) if len(etas) else np.array([])
    out = list(vals[np.abs(etas) >= 1.0]) + list(m.Delta(np.array([1.0, -1.0])))
    mid = vals[np.abs(etas) < 1.0]
    sup_out = float(np.max(out))
    sup_mid = float(mid.max()) if len(mid) else -np.inf
    d0 = float(m.Delta(np.array([0.0]))[0]) if np.min(np.abs(m.t)) > 1e-9 else -np.inf
    return sup_out, sup_mid, d0

def in_pair_cone(m: Atomic):
    """LP: is the atomic measure a nonneg combination of PASSIVE balanced pairs on its
    own atoms (xi_i < 0 < xi_j with |xi_i|*xi_j <= 1), plus delta_0 if present?"""
    xi, w = m.xi, m.w
    neg = np.where(xi < -1e-12)[0]
    pos = np.where(xi > 1e-12)[0]
    zer = np.where(np.abs(xi) <= 1e-12)[0]
    cols = []
    for i in neg:
        for j in pos:
            if abs(xi[i]) * xi[j] <= 1.0 + 1e-12:
                beta, gam = xi[j], -xi[i]
                col = np.zeros(len(xi))
                col[j] = gam / (beta + gam)
                col[i] = beta / (beta + gam)
                cols.append(col)
    for k in zer:
        col = np.zeros(len(xi)); col[k] = 1.0
        cols.append(col)
    if not cols:
        return False
    A = np.array(cols).T
    res = linprog(c=np.zeros(A.shape[1]), A_eq=A, b_eq=w,
                  bounds=[(0, None)] * A.shape[1], method="highs")
    return res.status == 0

nN = n_N_gain = n_pass = n_pass_inS = 0
tried = 0
for trial in range(12000):
    K = rng.integers(2, 6)
    u = rng.uniform(0.02, 0.97, K)
    for i in range(K):
        while np.min(np.abs(np.delete(u, i) - u[i])) < 1e-5:
            u[i] += 1e-4
    v = rng.dirichlet(np.ones(K) * rng.uniform(0.3, 2.0))
    try:
        m = Atomic(np.concatenate([u, -u]), np.concatenate([v / 2, v / 2]))
        so, smid, d0 = region_sups(m)
        s, _ = exact_sup_gap(m)
    except Exception:
        continue
    tried += 1
    if so <= TOL and d0 <= TOL:
        nN += 1
        if s > 1e-9:
            n_N_gain += 1
    if s <= 1e-10:
        n_pass += 1
        if in_pair_cone(m):
            n_pass_inS += 1
print(f"  tried {tried} random even atomic measures")
print(f"  N-satisfying: {nN};  of those with GAIN: {n_N_gain}  "
      f"({100.0*n_N_gain/max(nN,1):.1f}% — the N-side gap)")
print(f"  passive: {n_pass};  certified by S (pair cone): {n_pass_inS}  "
      f"({100.0*n_pass_inS/max(n_pass,1):.1f}% — S-side coverage)")
