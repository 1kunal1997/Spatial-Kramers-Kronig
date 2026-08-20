"""Goal B experiment 3 — does {outer-clean + centre-clean} imply passivity for EVEN g?

The asymmetric case is already dead (exp2 C: 1583 counterexamples). Even case open.
Probes:
  (1) The exp1-E2 profile 1/2 delta_0 + lambda|outer (known middle gain at eta~0.75):
      is it outer-clean? If yes -> even reduction DEAD immediately.
  (2) Random even measures: outer<=0, Delta(0)<=0, middle sup>0?
  (3) Adversarial: maximize middle bulge at eta0 in (0,1) over small even atomic
      parametrizations subject to outer+centre clean (penalty method + polish).
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/results/fable_ns_answer")
from common import Atomic, exact_sup_gap, critical_etas, xi_of_t, t_of_xi

rng = np.random.default_rng(99)
TOL = 1e-11


def even_from_radial(u, v, w0=0.0):
    u = np.asarray(u, float); v = np.asarray(v, float)
    t = np.concatenate([u, -u] + ([[0.0]] if w0 > 0 else []))
    w = np.concatenate([v / 2, v / 2] + ([[w0]] if w0 > 0 else []))
    return Atomic(t, w)


def region_sups(m: Atomic):
    """(sup over closed outer |eta|>=1, sup over open middle |eta|<1 excl. centre, Delta(0)).
    Outer sup includes the boundary points eta=+-1 and the tail limit 0- is NOT included
    (limit value 0 is approached from below iff leading coefficient sign; we report crit
    values and the boundary value)."""
    etas = critical_etas(m)
    vals = m.Delta(etas) if len(etas) else np.array([])
    out = list(vals[np.abs(etas) >= 1.0]) + list(m.Delta(np.array([1.0, -1.0])))
    mid = vals[np.abs(etas) < 1.0]
    sup_out = float(np.max(out))
    sup_mid = float(mid.max()) if len(mid) else -np.inf
    d0 = float(m.Delta(np.array([0.0]))[0]) if np.min(np.abs(m.t)) > 1e-9 else -np.inf
    return sup_out, sup_mid, d0


print("=" * 72)
print("(1) E2 profile: 1/2 delta_0 + lambda|outer  — outer-clean?")
print("=" * 72)
K = 120
uu = 0.5 + (np.arange(K) + 0.5) / (2 * K)
m = even_from_radial(uu, np.full(K, 0.5 / K), w0=0.5)
# continuous-formula check on a dense outer grid (atoms discretize the band; between-atom
# spikes are discretization artifacts, so also evaluate the continuous integral)
try:
    so, sm, d0 = region_sups(m)
    print(f"  atomic K={K}: sup_out={so:+.4e} sup_mid={sm:+.4e} Delta(0)={d0}")
except Exception as ex:
    print(f"  atomic K={K}: node test overflow ({ex}); relying on continuous evaluation")
# continuous: Delta(eta) = 0.5*ln|eta| + int_{1/2}^{1} ln|eta^2 - tan(pi u/2)^2| du*0.5...
from scipy.integrate import quad
def Delta_cont(eta):
    y = eta * eta
    def integrand(u):
        return 0.5 * np.log(abs(y - np.tan(np.pi * u / 2) ** 2))
    val, _ = quad(integrand, 0.5, 1.0, limit=400,
                  points=[t_of_xi(abs(eta))] if abs(eta) > 1 else None)
    return 0.5 * np.log(abs(eta)) + val - 0.5 * np.log1p(y)
outer_grid = np.concatenate([np.linspace(1.0, 8.0, 141), [10, 20, 50, 100]])
dv = np.array([Delta_cont(e) for e in outer_grid])
print(f"  continuous outer sup on grid: {dv.max():+.5e} at eta={outer_grid[np.argmax(dv)]:.3f}")
mid_grid = np.linspace(0.01, 0.999, 199)
dm = np.array([Delta_cont(e) for e in mid_grid])
print(f"  continuous middle sup on grid: {dm.max():+.5e} at eta={mid_grid[np.argmax(dm)]:.3f}")

print("=" * 72)
print("(2) random even measures: outer+centre clean, middle gain?")
print("=" * 72)
found = []
tried = 0
for trial in range(30000):
    K = rng.integers(2, 7)
    u = rng.uniform(0.02, 0.97, K)
    for i in range(K):
        while np.min(np.abs(np.delete(u, i) - u[i])) < 1e-5:
            u[i] += 1e-4
    v = rng.dirichlet(np.ones(K) * rng.uniform(0.3, 2.0))
    w0 = rng.uniform(0, 0.6) if rng.random() < 0.4 else 0.0
    tot = v.sum() + w0
    v, w0 = v / tot, w0 / tot
    try:
        m = even_from_radial(u, v, w0)
        so, sm, d0 = region_sups(m)
    except Exception:
        continue
    tried += 1
    if so <= TOL and d0 <= TOL and sm > 1e-8:
        found.append((sm, so, d0, u.copy(), v.copy(), w0))
print(f"  tried {tried}; even counterexamples: {len(found)}")
if found:
    found.sort(key=lambda z: -z[0])
    sm, so, d0, u, v, w0 = found[0]
    print(f"  strongest: mid={sm:+.4e} outer={so:+.4e} D0={d0:+.4e}")
    print(f"    u={np.round(u,5)} v={np.round(v,5)} w0={w0:.5f}")

print("=" * 72)
print("(3) adversarial polish (maximize middle bulge, even, outer+centre clean)")
print("=" * 72)
from scipy.optimize import minimize

def pack(u, logv):
    return np.concatenate([u, logv])

def objective(z, K, ret=False):
    u = np.clip(z[:K], 0.01, 0.98)
    v = np.exp(z[K:]); v = v / v.sum()
    try:
        m = even_from_radial(u, v)
        so, sm, d0 = region_sups(m)
    except Exception:
        return 1e6
    if ret:
        return so, sm, d0, u, v
    pen = 1e4 * max(so, 0.0) ** 2 + 1e4 * max(d0, 0.0) ** 2
    return -(sm if np.isfinite(sm) else -10.0) + pen

best = (None, -np.inf)
for K in (3, 4, 5):
    for start in range(40):
        u0 = np.sort(rng.uniform(0.02, 0.97, K))
        z0 = pack(u0, rng.normal(0, 1, K))
        res = minimize(objective, z0, args=(K,), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12})
        so, sm, d0, u, v = objective(res.x, K, ret=True)
        if so <= 1e-9 and d0 <= 1e-9 and sm > best[1]:
            best = ((u, v, so, d0), sm)
print(f"  best middle bulge with outer+centre clean: {best[1]:+.4e}")
if best[0] is not None:
    u, v, so, d0 = best[0]
    print(f"    u={np.round(u,5)} v={np.round(v,5)} outer={so:+.2e} D0={d0:+.2e}")
