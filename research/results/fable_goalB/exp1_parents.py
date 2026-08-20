"""Goal B (V8) experiment 1 — stress-test the two candidate parent conditions.

Candidate S (sufficient parent, SYMMETRIC form):
    g even, >= 0, radial CDF F(u) = nu([0,u]) of |t| under g/W satisfies
        F(u) >= u   for all u in [1/2, 1]
    ==> passive.
    (Recovers middle-half support: F == 1 on [1/2,1]; recovers Khinchin plateaus:
     F(u) = min(u/s,1) >= u. Constant 1/2 = inf_eta of kernel crossing point.)

Candidate N (necessary parent) redundancy probe:
    Is {Delta <= 0 on |eta| >= 1} AND {Delta(0) <= 0} sufficient for passivity?
    (If yes, the middle half minus centre is redundant -> two-sided, suspicious.
     Expect counterexample: concave bulge in an off-centre atom-free component.)

All verdicts via the audited exact node test (grid-free). No FFT, no linspace.
"""
import sys, os
import numpy as np

sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/results/fable_ns_answer")
from common import Atomic, balanced_pair, exact_sup_gap, critical_etas, xi_of_t, t_of_xi

rng = np.random.default_rng(20260820)
TOL = 1e-11   # sup <= TOL counts as passive (machine-zero boundary touches)


def even_from_radial(u, v, w0=0.0):
    """Even atomic measure: mass v_k split at +-u_k, plus optional atom at 0."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    t = np.concatenate([u, -u] + ([[0.0]] if w0 > 0 else []))
    w = np.concatenate([v / 2, v / 2] + ([[w0]] if w0 > 0 else []))
    return Atomic(t, w)


def radial_cdf_ok(u, v, w0=0.0, lo=0.5):
    """Check F(u) >= u for all u in [lo, 1] for the radial measure (atoms u_k mass v_k,
    plus mass w0 at 0), total mass normalized to 1. F is right-continuous; the binding
    checks are at u = lo and just below each atom in (lo, 1]."""
    W = v.sum() + w0
    order = np.argsort(u)
    us, vs = u[order], v[order]
    cum = w0 + np.concatenate([[0.0], np.cumsum(vs)])  # F just below each atom / at end
    # F(x) for x in [lo,1]: F(lo) >= lo, and F(u_k^-) >= u_k for atoms u_k in (lo,1]
    ok = True
    # F(lo): mass strictly... use <= lo (closed): F(lo) = w0 + sum(vs[us <= lo])
    Flo = (w0 + vs[us <= lo].sum()) / W
    if Flo < lo - 1e-12:
        ok = False
    for k, uk in enumerate(us):
        if lo < uk <= 1.0:
            if cum[k] / W < uk - 1e-12:   # F(uk^-) = cum[k]
                ok = False
    return ok


def margin_radial(u, v, w0=0.0, lo=0.5):
    """min over binding points of F(u) - u  (>=0 iff condition holds)."""
    W = v.sum() + w0
    order = np.argsort(u)
    us, vs = u[order], v[order]
    cum = w0 + np.concatenate([[0.0], np.cumsum(vs)])
    m = (w0 + vs[us <= lo].sum()) / W - lo
    for k, uk in enumerate(us):
        if lo < uk <= 1.0:
            m = min(m, cum[k] / W - uk)
    return m


def classify_sup(m: Atomic):
    """sup Delta split by region: outer (|eta|>=1), middle (|eta|<1), centre value."""
    etas = critical_etas(m)
    vals = m.Delta(etas) if len(etas) else np.array([])
    out = vals[np.abs(etas) >= 1.0]
    mid = vals[np.abs(etas) < 1.0]
    sup_out = float(out.max()) if len(out) else 0.0   # tail limit is 0 (attained at inf)
    sup_out = max(sup_out, 0.0) if False else sup_out # tail is 0 exactly but sign(0)=0: keep crit only
    sup_mid = float(mid.max()) if len(mid) else -np.inf
    d0 = float(m.Delta(np.array([0.0]))[0]) if np.min(np.abs(m.t)) > 0 else -np.inf
    return sup_out, sup_mid, d0, etas, vals


print("=" * 72)
print("E0 CONTROLS (each must fire in the known direction)")
print("=" * 72)
# C0a: balanced pair threshold a+b=1
for a, b, expect in [(0.4, 0.55, "PASS"), (0.4, 0.6, "PASS-boundary"), (0.4, 0.65, "GAIN")]:
    p = balanced_pair(a, b)
    s, e = exact_sup_gap(p)
    print(f"  pair a={a} b={b}: sup={s:+.3e} at eta={e:+.4f}  expect {expect}")
# C0b: E1' bifurcation at beta = sqrt(2)-1
for beta, expect in [(0.5, "PASS"), (np.sqrt(2) - 1 + 1e-6, "PASS~0"), (0.35, "GAIN")]:
    u1, u2 = t_of_xi(beta), t_of_xi(1 / beta)
    m = Atomic([u1, -u1, u2, -u2], [0.5] * 4)
    s, e = exact_sup_gap(m)
    print(f"  E1' beta={beta:.6f}: sup={s:+.3e} at eta={e:+.4f}  expect {expect}")
# C0c: uniform staircase (discretized equilibrium) -> sup should be small +, shrink with K
for K in (50, 100, 200, 400):
    uu = (np.arange(K) + 0.5) / K   # radii, uniform in u
    m = even_from_radial(uu, np.full(K, 1.0 / K))
    s, e = exact_sup_gap(m)
    print(f"  uniform K={K}: sup={s:+.3e} (discretization floor)")
# C0d: S(1.2) in xi: gain at centre only, outer clean
u0 = t_of_xi(1.2)
m = even_from_radial([u0], [1.0])
so, sm, d0, _, _ = classify_sup(m)
print(f"  S(xi=1.2): sup_out={so:+.3e} sup_mid={sm:+.3e} Delta(0)={d0:+.3e} "
      f"expect out<=0, mid=ln(1.2)={np.log(1.2):+.3e}")

print("=" * 72)
print("E1 THEOREM-S STRESS TEST (random even measures satisfying the radial-CDF cond.)")
print("=" * 72)
n_ok, n_gain, worst = 0, 0, (0.0, None)
for trial in range(4000):
    K = rng.integers(1, 7)
    u = rng.uniform(0.02, 0.995, K)
    v = rng.dirichlet(np.ones(K))
    w0 = 0.0
    if rng.random() < 0.3:
        w0 = rng.uniform(0, 0.5)
        tot = v.sum() + w0
        v, w0 = v / tot, w0 / tot
    if not radial_cdf_ok(u, v, w0):
        continue
    m = even_from_radial(u, v, w0)
    s, e = exact_sup_gap(m)
    n_ok += 1
    if s > TOL:
        n_gain += 1
        if s > worst[0]:
            worst = (s, (u.copy(), v.copy(), w0, e))
print(f"  measures satisfying condition: {n_ok};  with gain: {n_gain}")
if worst[1] is not None:
    u, v, w0, e = worst[1]
    print(f"  WORST violation sup={worst[0]:+.3e} at eta={e:+.4f}")
    print(f"    u={np.round(u,4)} v={np.round(v,4)} w0={w0:.4f}, margin={margin_radial(u,v,w0):+.3e}")

print("  -- boundary-hugging construction (F(u)-u ~ eps on several outer atoms) --")
n_b, n_bg, worstb = 0, 0, (0.0, None)
for trial in range(2000):
    Ko = rng.integers(1, 5)           # outer atoms in (1/2, 1)
    uo = np.sort(rng.uniform(0.5, 0.995, Ko))
    eps = 10.0 ** rng.uniform(-4, -1.3)
    # place inner mass so that F(uo_k^-) = uo_k + eps exactly (tight backing)
    # cumulative just below uo_0 must be uo_0+eps: put it at random inner radii
    Ki = rng.integers(1, 4)
    ui = rng.uniform(0.02, 0.5, Ki)
    vi = rng.dirichlet(np.ones(Ki)) * min(uo[0] + eps, 0.9)
    need = []
    cum = vi.sum()
    uu, vv = list(ui), list(vi)
    ok = True
    for k in range(Ko):
        tgt = uo[k] + eps
        if k + 1 < Ko:
            nxt = uo[k + 1] + eps
        else:
            nxt = 1.0
        # mass of atom at uo_k: brings cumulative from tgt to nxt is done by NEXT...
        # atom k mass = (next target) - (current cumulative). Simplest: after atom k,
        # cumulative = uo[k+1]+eps (or 1 at end).
        mass_k = nxt - cum
        if mass_k <= 0:
            ok = False
            break
        uu.append(uo[k]); vv.append(mass_k)
        cum += mass_k
    if not ok:
        continue
    u, v = np.array(uu), np.array(vv)
    v = v / v.sum()
    if not radial_cdf_ok(u, v):
        continue
    m = even_from_radial(u, v)
    s, e = exact_sup_gap(m)
    n_b += 1
    if s > TOL:
        n_bg += 1
        if s > worstb[0]:
            worstb = (s, (u.copy(), v.copy(), e))
print(f"  boundary-hugging measures: {n_b};  with gain: {n_bg}")
if worstb[1] is not None:
    u, v, e = worstb[1]
    print(f"  WORST sup={worstb[0]:+.3e} at eta={e:+.4f} u={np.round(u,4)} v={np.round(v,4)}")

print("=" * 72)
print("E2 EXACT BOUNDARY CASE: 1/2 delta_0 + lambda|outer  (F(u) == u on [1/2,1])")
print("=" * 72)
for K in (50, 100, 200, 400, 800):
    uu = 0.5 + (np.arange(K) + 0.5) / (2 * K)     # uniform radii in [1/2, 1]
    m = even_from_radial(uu, np.full(K, 0.5 / K), w0=0.5)
    s, e = exact_sup_gap(m)
    print(f"  K={K}: sup={s:+.4e} at eta={e:+.5f} (t={t_of_xi(e):+.5f})")
print("  (compare with C0c uniform floor at same K; if sup -> positive const, S is DEAD)")

print("=" * 72)
print("E3 N-REDUNDANCY PROBE: outer-clean + centre-clean, gain in middle?")
print("=" * 72)
found = []
for trial in range(20000):
    K = rng.integers(2, 6)
    t = rng.uniform(-0.995, 0.995, K)
    w = rng.dirichlet(np.ones(K))
    m0 = Atomic(t, w)
    # balance: add the balancing atom to kill M1 (keep positivity)
    M1 = m0.M1()
    tb = -t_of_xi(np.sign(M1) * min(abs(M1) / 0.2, 50.0)) if abs(M1) > 1e-14 else None
    # simpler: pair each atom into balanced pairs instead
    # build from balanced pairs directly:
    npairs = rng.integers(2, 5)
    mm = None
    for _ in range(npairs):
        a, b = rng.uniform(0.02, 0.98, 2)
        p = balanced_pair(a, b, mass=rng.uniform(0.1, 1.0))
        mm = p if mm is None else mm.mix(p)
    so, sm, d0, _, _ = classify_sup(mm)
    if so <= TOL and d0 <= TOL and sm > 1e-9:
        found.append((sm, so, d0, mm))
print(f"  counterexamples found (outer<=0, Delta(0)<=0, middle gain): {len(found)}")
if found:
    found.sort(key=lambda z: -z[0])
    sm, so, d0, mm = found[0]
    print(f"  strongest: middle sup={sm:+.3e}, outer sup={so:+.3e}, Delta(0)={d0:+.3e}")
    print(f"    atoms t={np.round(mm.t,4)}")
    print(f"    weights={np.round(mm.w,4)}  (M1={mm.M1():+.2e})")
else:
    print("  NONE in 20000 random balanced-pair mixtures — needs adversarial follow-up")
