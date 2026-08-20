"""Goal B experiment 2 — closure operations, N-redundancy probe, and the B3 gap map.

 (A) CONTROL: tau-space convolution form of the symmetric criterion
     Delta(eta)/W = phi*nu~(tau) - ln(2cosh(tau/2)) + m1/2,  tau = ln(eta^2),
     phi(s) = ln(2 sinh(|s|/2)), nu~ = distribution of 2 ln|xi| under g/W.
 (B) CONTROL: contraction closure. Identity
     Delta_{mu_c}(eta) = Delta_mu(eta/c) + (W/2) ln((c^2+eta^2)/(1+eta^2)),
     and passivity preservation for c <= 1 on random passive measures.
 (C) N-redundancy probe (fixed): outer-clean + centre-clean but middle gain?
 (D) N single-pair threshold: outer-clean for S(beta) iff beta <= sqrt(3).
 (E) B3 gap map: family x*S(beta) + (1-x)*S(1/beta), beta in (0,1), x in [0,1]:
     exact passive region vs N region (outer+centre) vs S (join) region {x=1, beta<=1}.
 (F) Is a contracted plateau outside the plain join S2? (outer radial density
     must be non-increasing for S2 membership)
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/results/fable_ns_answer")
from common import Atomic, balanced_pair, exact_sup_gap, critical_etas, xi_of_t, t_of_xi

rng = np.random.default_rng(7)
TOL = 1e-11

print("=" * 72)
print("(A) tau-space convolution form — identity control")
print("=" * 72)
# random even atomic measure; compare Delta(eta) with the tau-form
u = np.array([0.15, 0.42, 0.66, 0.83]); v = np.array([0.3, 0.25, 0.25, 0.2])
m = Atomic(np.concatenate([u, -u]), np.concatenate([v / 2, v / 2]))
tau_atoms = 2.0 * np.log(np.abs(xi_of_t(u)))
m1 = np.sum(v * tau_atoms)
for eta in (0.3, 0.9, 1.7, 4.0):
    tau = np.log(eta**2)
    phi_conv = np.sum(v * np.log(2.0 * np.sinh(np.abs(tau - tau_atoms) / 2.0)))
    tau_form = phi_conv - np.log(2.0 * np.cosh(tau / 2.0)) + m1 / 2.0
    direct = m.Delta(np.array([eta]))[0]
    print(f"  eta={eta}: direct={direct:+.12f} tau-form={tau_form:+.12f} "
          f"diff={abs(direct-tau_form):.2e}")

print("=" * 72)
print("(B) contraction closure control")
print("=" * 72)
def contract(m: Atomic, c):
    xi_new = c * m.xi
    return Atomic(t_of_xi(xi_new), m.w.copy())
# identity check
c = 0.7
mc = contract(m, c)
for eta in (0.25, 1.1, 3.0):
    lhs = mc.Delta(np.array([eta]))[0]
    rhs = m.Delta(np.array([eta / c]))[0] + m.W / 2 * np.log((c**2 + eta**2) / (1 + eta**2))
    print(f"  eta={eta}: Delta_c={lhs:+.12f} identity RHS={rhs:+.12f} diff={abs(lhs-rhs):.2e}")
# preservation on random passive measures (and NON-preservation for c>1 must appear)
n_pass, n_bad, n_exp_pass, n_exp_bad = 0, 0, 0, 0
for _ in range(800):
    mm = None
    for _ in range(rng.integers(2, 5)):
        a, b = rng.uniform(0.05, 0.9, 2)
        if a + b > 0.99:
            s = 0.99 * rng.uniform(0.5, 1.0) / (a + b); a, b = a * s, b * s
        p = balanced_pair(a, b, mass=rng.uniform(0.2, 1.0))
        mm = p if mm is None else mm.mix(p)
    s0, _ = exact_sup_gap(mm)
    if s0 > TOL:
        continue
    n_pass += 1
    cc = rng.uniform(0.2, 1.0)
    sc, _ = exact_sup_gap(contract(mm, cc))
    if sc > 1e-9:
        n_bad += 1
    ce = rng.uniform(1.05, 3.0)
    se, _ = exact_sup_gap(contract(mm, ce))
    n_exp_pass += 1
    if se > 1e-9:
        n_exp_bad += 1
print(f"  passive seeds: {n_pass}; contraction (c<=1) violations: {n_bad}  [expect 0]")
print(f"  expansion (c>1) violations: {n_exp_bad}/{n_exp_pass}  [expect >0: control must fire]")

print("=" * 72)
print("(C) N-redundancy probe: outer<=0 AND Delta(0)<=0 AND middle gain?")
print("=" * 72)
def classify_sup(m: Atomic):
    etas = critical_etas(m)
    vals = m.Delta(etas) if len(etas) else np.array([])
    out = vals[np.abs(etas) >= 1.0]
    mid = vals[np.abs(etas) < 1.0]
    sup_out = float(out.max()) if len(out) else -np.inf
    sup_mid = float(mid.max()) if len(mid) else -np.inf
    d0 = float(m.Delta(np.array([0.0]))[0]) if np.min(np.abs(m.t)) > 1e-9 else -np.inf
    return sup_out, sup_mid, d0

found = []
tried = 0
for trial in range(30000):
    mm = None
    for _ in range(rng.integers(2, 5)):
        a, b = rng.uniform(0.02, 0.96, 2)
        p = balanced_pair(a, b, mass=rng.uniform(0.1, 1.0))
        mm = p if mm is None else mm.mix(p)
    # jitter exact duplicates
    tt, ww = mm.t.copy(), mm.w.copy()
    for i in range(len(tt)):
        while np.min(np.abs(np.delete(tt, i) - tt[i])) < 1e-6:
            tt[i] += 1e-5 * rng.standard_normal()
    mm = Atomic(np.clip(tt, -0.97, 0.97), ww)
    try:
        so, sm, d0 = classify_sup(mm)
    except Exception:
        continue
    tried += 1
    if so <= TOL and d0 <= TOL and sm > 1e-8:
        found.append((sm, so, d0, mm))
print(f"  tried {tried}; counterexamples: {len(found)}")
if found:
    found.sort(key=lambda z: -z[0])
    sm, so, d0, mm = found[0]
    print(f"  strongest: mid sup={sm:+.4e}, outer sup={so:+.4e}, Delta(0)={d0:+.4e}")
    print(f"    t={np.round(mm.t,5)}")
    print(f"    w={np.round(mm.w,5)} (M1={mm.M1():+.1e})")

print("=" * 72)
print("(D) N threshold for single symmetric pair: outer-clean iff beta <= sqrt(3)")
print("=" * 72)
for beta in (1.5, np.sqrt(3.0), 1.7405, 2.0):
    u0 = t_of_xi(beta)
    m1p = Atomic([u0, -u0], [0.5, 0.5])
    so, sm, d0 = classify_sup(m1p)
    # outer sup from critical etas may miss: for the pair, sup on |eta|>=1 is at eta=1 boundary
    edge = m1p.Delta(np.array([1.0]))[0]
    print(f"  beta={beta:.4f}: crit-outer={so:+.3e} Delta(1)={edge:+.4e} mid={sm:+.4e}")

print("=" * 72)
print("(E) B3 gap map on the reciprocal-pair family x*S(beta)+(1-x)*S(1/beta)")
print("=" * 72)
betas = np.linspace(0.05, 0.999, 120)
xs = np.linspace(0.0, 1.0, 101)
passive = np.zeros((len(betas), len(xs)), bool)
Ncond = np.zeros_like(passive)
for i, be in enumerate(betas):
    u1, u2 = t_of_xi(be), t_of_xi(1.0 / be)
    for j, x in enumerate(xs):
        w = np.array([x / 2, x / 2, (1 - x) / 2, (1 - x) / 2])
        keep = w > 1e-12
        tt = np.array([u1, -u1, u2, -u2])[keep]
        mm = Atomic(tt, w[keep])
        s, _ = exact_sup_gap(mm)
        passive[i, j] = s <= 1e-10
        so, smid, d0 = classify_sup(mm)
        Ncond[i, j] = (so <= 1e-10) and (d0 <= 1e-10)
np.savez("gapmap_reciprocal.npz", betas=betas, xs=xs, passive=passive, Ncond=Ncond)
area_p = passive.mean(); area_N = Ncond.mean()
false_N = (Ncond & ~passive).mean()
print(f"  area(passive)={area_p:.3f}  area(N-cond)={area_N:.3f}  area(N but gain)={false_N:.3f}")
print(f"  S(join) region in this family is the single line x=1, beta<=1 (measure ~0)")
# equal-mass slice must reproduce E1': passive iff beta >= sqrt(2)-1
jmid = np.argmin(np.abs(xs - 0.5))
bcut = betas[passive[:, jmid]].min()
print(f"  x=0.5 slice: passive down to beta={bcut:.4f} (E1' predicts {np.sqrt(2)-1:.4f})")

print("=" * 72)
print("(F) contracted plateau vs plain join S2 (outer density monotone?)")
print("=" * 72)
# plateau Unif[0,s] contracted by c: radial density rho~(u~) on outer half
for s, c in [(1.0, 0.5), (0.9, 0.6), (1.0, 0.85)]:
    uu = np.linspace(0.501, min(1.0, t_of_xi(c * xi_of_t(s)) - 1e-6), 200)
    # inverse image u of u~: tan(pi u/2) = tan(pi u~/2)/c
    u_pre = t_of_xi(xi_of_t(uu) / c)
    ok = np.all(u_pre <= s + 1e-12)
    dens = np.gradient(u_pre, uu) / s   # pushforward density (1/s)*du/du~
    mono = np.all(np.diff(dens) <= 1e-9)
    print(f"  s={s} c={c}: image outer edge={t_of_xi(c*xi_of_t(s)):.4f}, "
          f"outer density monotone-decreasing: {mono}; "
          f"dens range [{dens.min():.3f},{dens.max():.3f}]")
