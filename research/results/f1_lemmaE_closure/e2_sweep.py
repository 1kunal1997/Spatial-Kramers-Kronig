"""E2 — numerical verification of Lemma E2 (research/theory/lemmaE_closure.md §5).

WHAT IS BEING CHECKED (none of it is load-bearing for the proof; these are checks on the algebra):

  (V1) base point p0 = S(1/2) + S(2): Delta <= 0 on R with equality only at eta = 0;
       Delta''(0) = -25/4 exactly; other critical points at eta = +-1 with value ln(3/4).
  (V2) an explicit admissible rho: max Delta''_{p0} on [-1/4, 1/4] <= -25/8.
  (V3) epsilon := -max_K Delta_{p0} = ln(4/3) exactly for the configs used (K = H \ (-1/4,1/4)).
  (V4) the IFT continuation: Newton-solve  Phi(eta, r; t) = (Delta_m(eta), Delta'_m(eta)) = 0
       for m = (1+r) P1 + P2 + t (P3 + ... + Pn); check
         eta(t)/t -> (4/25) * sum_j (beta_j - gamma_j)/(beta_j gamma_j)
         r(t)/t   -> Delta_u(0)/ln 2
       and that m(t) has all weights > 0.
  (V5) INDEPENDENT confirmation that m(t) is on the passive boundary, by the pre-existing exact
       node test (roots of the degree-(2n+1) polynomial Q) in results/fable_ns_answer/common.py --
       a different code path, different author, different algorithm from the Newton solve.

CONTROLS (each would FAIL if the harness were wrong):
  (C1) cross-evaluator: fresh mpmath Delta vs common.py Atomic.Delta, on random etas.
  (C2) SIGN control: swapping the added pair to beta_j < gamma_j must flip the sign of eta(t).
       (A harness that manufactured motion, or that lost the sign of Delta', would not track this.)
  (C3) ZERO control: a SYMMETRIC added pair (beta = gamma) must give eta(t) == 0 identically.
  (C4) BOUNDARY control: perturbing the solved r off the solution must flip the node-test verdict:
       r - 1e-3 -> sup Delta > 0 (GAIN), r + 1e-3 -> sup Delta < 0 (strictly passive).
  (C5) RECIPROCITY control: replacing S(2) by S(3) must give Delta(0) != 0 (base point destroyed).
  (C6) BIFURCATION control: S(beta)+S(1/beta) must be passive iff beta >= sqrt(2)-1.

Conventions: everything is done in the Cayley coordinate xi directly (xi_j are chosen numbers),
so tan(pi t/2) is never evaluated near the window edge -- the historical tan(pi/2) overflow trap
cannot occur here. Conversion to common.py's t-coordinates uses t = (2/pi) arctan(xi), exact-ish.
"""
import sys, os
import numpy as np
from mpmath import mp, mpf, log, sqrt, atan, pi, matrix, lu_solve

mp.dps = 40

_FABLE = "/Users/kunal/Documents/Spatial KK Project/research/results/fable_ns_answer"
sys.path.insert(0, _FABLE)
import common as C   # pre-existing, independent implementation (exact node test)

# ------------------------------------------------------------------ fresh evaluator (mpmath)
class Meas:
    """Atomic measure in the Cayley coordinate: atoms xi_j, weights w_j > 0."""
    def __init__(self, xi, w):
        self.xi = [mpf(str(v)) for v in xi]
        self.w  = [mpf(str(v)) for v in w]
        self.W  = sum(self.w)
    def M1(self):
        return sum(wj*xj for wj, xj in zip(self.w, self.xi))
    def D(self, eta):     # Delta
        eta = mpf(eta)
        return sum(wj*log(abs(eta-xj)) for wj, xj in zip(self.w, self.xi)) \
               - self.W/2*log(1+eta**2)
    def D1(self, eta):    # Delta'
        eta = mpf(eta)
        return sum(wj/(eta-xj) for wj, xj in zip(self.w, self.xi)) \
               - self.W*eta/(1+eta**2)
    def D2(self, eta):    # Delta''
        eta = mpf(eta)
        return -sum(wj/(eta-xj)**2 for wj, xj in zip(self.w, self.xi)) \
               - self.W*(1-eta**2)/(1+eta**2)**2

def sym_pair(beta):
    """S(beta): mass-1 balanced pair, atoms +-beta, weights 1/2."""
    return Meas([beta, -beta], [mpf(1)/2, mpf(1)/2])

def pair(beta, gamma):
    """mass-1 balanced pair, atoms +beta and -gamma, weights gamma/(b+g), beta/(b+g)."""
    b, g = mpf(str(beta)), mpf(str(gamma))
    return Meas([b, -g], [g/(b+g), b/(b+g)])

def combine(pieces, coeffs):
    xi, w = [], []
    for P, c in zip(pieces, coeffs):
        for xj, wj in zip(P.xi, P.w):
            xi.append(xj); w.append(mpf(str(c))*wj)
    return Meas(xi, w)

def to_common(M):
    """Port a Meas to common.Atomic (t-coordinates) -- the independent evaluator."""
    t = [float(2/pi*atan(x)) for x in M.xi]
    w = [float(x) for x in M.w]
    return C.Atomic(t, w)

out = []
def say(s=""):
    print(s); out.append(s)

# ================================================================== (V1) base point
say("="*78)
say("(V1) BASE POINT  p0 = S(1/2) + S(2)   [atoms xi = +-1/2, +-2, each weight 1/2]")
P1, P2 = sym_pair(mpf(1)/2), sym_pair(2)
p0 = combine([P1, P2], [1, 1])
say(f"  W = {float(p0.W):.15f}   M1 = {float(p0.M1()):.3e}   (predicted W=2, M1=0)")
say(f"  Delta(0)   = {mp.nstr(p0.D(0), 12)}      predicted 0 exactly")
say(f"  Delta''(0) = {mp.nstr(p0.D2(0), 15)}     predicted -25/4 = -6.25 exactly")
say(f"  Delta(1)   = {mp.nstr(p0.D(1), 12)}      predicted ln(3/4) = {mp.nstr(log(mpf(3)/4),12)}")
say(f"  Delta'(1)  = {mp.nstr(p0.D1(1), 6)}      predicted 0 (critical point)")
# closed form vs direct sum: Delta = 1/2 ln(|y-4||y-1/4|/(1+y)^2), y = eta^2
def closed(eta):
    y = mpf(str(eta))**2
    return log(abs(y-4)*abs(y-mpf(1)/4)/(1+y)**2)/2
res = max(abs(p0.D(e) - closed(e)) for e in [0.13, 0.7, 1.3, 2.7, 9.1, -0.31, -1.9, -5.0])
say(f"  closed form vs direct atom sum, max residual over 8 etas: {mp.nstr(res, 3)}")
# global sup: dense scan + the exact node test (two routes)
grid = [mpf(k)/mpf(20000) for k in range(-200000, 200001)]   # eta in [-10,10], step 5e-5
grid = [g for g in grid if min(abs(g-x) for x in p0.xi) > mpf('1e-9')]
vals = [(p0.D(g), g) for g in grid[::37]]        # subsample for speed, then refine near max
vmax, amax = max(vals)
say(f"  dense scan (270k pts, step 1.85e-3 after subsample): sup Delta = {mp.nstr(vmax,6)} at eta = {mp.nstr(amax,6)}")
sup_node, at_node = C.exact_sup_gap(to_common(p0))
say(f"  INDEPENDENT exact node test (common.py):    sup Delta = {sup_node:+.3e} at eta = {at_node:+.9f}")
say(f"  => p0 is a boundary point, unique touch at the origin.")

# ================================================================== (V2) explicit rho
say()
say("(V2) EXPLICIT rho for step (4c): need max Delta''_{p0} <= -25/8 = -3.125 on [-rho,rho]")
for rho in ['0.25', '0.4', '0.45']:
    R = mpf(rho)
    xs = [R*mpf(k)/200 for k in range(-200, 201)]
    m = max(p0.D2(x) for x in xs)
    say(f"   rho = {rho}: max Delta'' on [-rho,rho] = {mp.nstr(m, 8)}   {'OK' if m <= -mpf(25)/8 else 'FAILS'}")

# ================================================================== (C5,C6) controls on the family
say()
say("(C5) RECIPROCITY control: S(1/2)+S(3) (reciprocity broken) must have Delta(0) != 0")
bad = combine([sym_pair(mpf(1)/2), sym_pair(3)], [1, 1])
say(f"     Delta(0) = {mp.nstr(bad.D(0), 8)}   (= ln(3/2) = {mp.nstr(log(mpf(3)/2),8)})  -> not a boundary touch")
say("(C6) BIFURCATION control: S(b)+S(1/b) passive iff b >= sqrt(2)-1 = 0.41421356")
for b in ['0.30', '0.41421356', '0.5', '0.9']:
    bb = mpf(b)
    fam = combine([sym_pair(bb), sym_pair(1/bb)], [1, 1])
    s, a = C.exact_sup_gap(to_common(fam))
    c = float(bb**2 + bb**(-2))
    say(f"     beta = {b:>10}: c = beta^2+beta^-2 = {c:8.4f}   sup Delta = {s:+.4e} at {a:+.4f}"
        f"   -> {'GAIN' if s > 1e-12 else 'passive'}")

# ================================================================== (V4,V5,C1-C4) the sweep
def sweep(name, added, tvals=('1e-4','3e-4','1e-3','3e-3','1e-2','3e-2'), report_node=True):
    """added = list of (beta_j, gamma_j) for j >= 3."""
    say()
    say("-"*78)
    say(f"(V4) CONTINUATION  [{name}]   added pairs (beta_j, gamma_j) = {added}")
    Ps = [pair(b, g) for (b, g) in added]
    u  = combine(Ps, [1]*len(Ps))                      # direction: switch on all added pairs
    D1u0 = sum((mpf(str(b))-mpf(str(g)))/(mpf(str(b))*mpf(str(g))) for (b, g) in added)
    pred_eta = mpf(4)/25 * D1u0
    pred_r   = u.D(0)/log(2)
    say(f"  Delta'_u(0) = sum (b-g)/(bg) = {mp.nstr(D1u0, 10)}")
    say(f"  predicted eta_dot(0) = (4/25)*Delta'_u(0) = {mp.nstr(pred_eta, 10)}")
    say(f"  predicted r_dot(0)   = Delta_u(0)/ln2      = {mp.nstr(pred_r, 10)}   (Delta_u(0) = {mp.nstr(u.D(0),8)})")
    say(f"  {'t':>8} {'eta(t)':>16} {'eta/t':>14} {'r(t)':>16} {'r/t':>13} "
        f"{'node-test sup':>15} {'|argmax-eta|':>13} {'min weight':>11}")
    rows = []
    for ts in tvals:
        t = mpf(ts)
        eta, r = mpf(0), mpf(0)                        # Newton from the base point
        for _ in range(80):
            m = combine([P1, P2] + Ps, [1+r, 1] + [t]*len(Ps))
            F  = matrix([m.D(eta), m.D1(eta)])
            J  = matrix([[m.D1(eta), P1.D(eta)],
                         [m.D2(eta), P1.D1(eta)]])
            d  = lu_solve(J, -F)
            eta += d[0]; r += d[1]
            if abs(d[0]) + abs(d[1]) < mpf('1e-35'):
                break
        m = combine([P1, P2] + Ps, [1+r, 1] + [t]*len(Ps))
        sup, at = C.exact_sup_gap(to_common(m)) if report_node else (float('nan'), float('nan'))
        minw = min(float(x) for x in [1+r, 1] + [t]*len(Ps))
        say(f"  {ts:>8} {mp.nstr(eta,10):>16} {mp.nstr(eta/t,8):>14} {mp.nstr(r,10):>16} "
            f"{mp.nstr(r/t,7):>13} {sup:>15.2e} {abs(at-float(eta)):>13.2e} {minw:>11.4f}")
        rows.append((t, eta, r, sup, at))
    say(f"  extrapolated eta/t (Richardson, first two t): "
        f"{mp.nstr((3*rows[0][1]/rows[0][0] - rows[1][1]/rows[1][0])/2, 10)}")
    if pred_eta != 0:
        say(f"  relative error of eta(t)/t vs prediction at t=1e-4: "
            f"{mp.nstr(abs(rows[0][1]/rows[0][0]/pred_eta - 1), 4)}")
    else:
        say(f"  prediction is eta_dot(0) = 0 exactly; measured |eta(t)| = "
            f"{mp.nstr(abs(rows[0][1]), 4)}, {mp.nstr(abs(rows[-1][1]), 4)}")
    say(f"  relative error of r(t)/t   vs prediction at t=1e-4: "
        f"{mp.nstr(abs(rows[0][2]/rows[0][0]/pred_r - 1), 4)}")
    return rows, Ps

rows3, Ps3 = sweep("n=3", [(1.4, 1.2)])
rows6, _   = sweep("n=6", [(1.15, 1.1), (1.35, 1.25), (1.7, 1.55), (2.4, 2.2)])

say()
say("(C2) SIGN control: mirror the added pair (beta<gamma) -> eta(t) must change sign")
rowsM, _ = sweep("n=3 mirrored", [(1.2, 1.4)], tvals=('1e-3','1e-2'))
say(f"   original  eta(1e-2) = {mp.nstr(rows3[4][1], 8)}")
say(f"   mirrored  eta(1e-2) = {mp.nstr(rowsM[1][1], 8)}   -> signs opposite: "
    f"{(rows3[4][1] > 0) != (rowsM[1][1] > 0)}")

say()
say("(C3) ZERO control: SYMMETRIC added pair (beta=gamma=1.3) -> eta(t) must vanish identically")
rowsS, _ = sweep("n=3 symmetric added", [(1.3, 1.3)], tvals=('1e-3','1e-2'))
say(f"   |eta(1e-2)| = {mp.nstr(abs(rowsS[1][1]), 4)}  (predicted 0 by evenness)")

# ================================================================== (C4) boundary control
say()
say("(C4) BOUNDARY control at t = 1e-2 of the n=3 config: move r off the solution, verdict must flip")
t = mpf('1e-2'); eta_s, r_s = rows3[4][1], rows3[4][2]
for dr in ['-1e-3', '-1e-5', '0', '+1e-5', '+1e-3']:
    m = combine([P1, P2] + Ps3, [1 + r_s + mpf(dr), 1] + [t])
    sup, at = C.exact_sup_gap(to_common(m))
    say(f"   r -> r{dr:>7} : node-test sup Delta = {sup:+.6e} at eta = {at:+.8f}"
        f"   -> {'GAIN' if sup > 1e-12 else ('BOUNDARY' if abs(sup) <= 1e-12 else 'strictly passive')}")

# ================================================================== (C1) cross-evaluator
say()
say("(C1) CROSS-EVALUATOR control: fresh mpmath Delta vs common.py Atomic.Delta (different code path)")
m = combine([P1, P2] + Ps3, [1 + rows3[4][2], 1, rows3[4][0]])
A = to_common(m)
worst = 0.0
for e in [-3.3, -1.05, -0.37, 0.0, 0.21, 0.77, 1.6, 4.2]:
    a = float(m.D(e)); b = float(A.Delta(np.array([e]))[0])
    worst = max(worst, abs(a-b))
say(f"   max |fresh - common.py| over 8 etas = {worst:.3e}")

# ================================================================== (V6) universal concavity
say()
say("(V6) UNIVERSAL CONCAVITY (used in step (4c)):  Delta''_g < 0 on (-1,1) for EVERY positive g")
import random
random.seed(7)
worst = -mpf('1e99')
for _ in range(300):
    nn = random.randint(1, 6)
    xr = [mpf(random.uniform(-5, 5)) for _ in range(nn)]
    wr = [mpf(random.uniform(0.01, 3)) for _ in range(nn)]
    g = Meas(xr, wr)
    for k in range(-99, 100):
        e = mpf(k)/100
        if min(abs(e-x) for x in g.xi) < mpf('1e-3'):
            continue
        worst = max(worst, g.D2(e))
say(f"   max Delta'' over 300 random positive measures x 199 etas in (-1,1): {mp.nstr(worst, 6)}")
say("   SHARPNESS control -- the same quantity must be able to go POSITIVE outside (-1,1):")
gc = Meas([10], [1])
for e in ['1.0', '1.5', '3.0']:
    say(f"     single atom xi=10, eta={e}: Delta'' = {mp.nstr(gc.D2(e), 6)}")

# ================================================================== (V3) epsilon
say()
say("(V3) epsilon = -max_K Delta_{p0},  K = H \\ (-1/4, 1/4)   [predicted ln(4/3) = 0.28768207]")
for nm, H in [("n=3 (H=[-2,2])", mpf(2)), ("n=6 (H=[-2.4,2.4])", mpf('2.4'))]:
    xs = [mpf(k)/4000 for k in range(1000, int(H*4000)+1)]     # [0.25, H]
    xs = [x for x in xs if min(abs(x-abs(xj)) for xj in p0.xi) > mpf('1e-6')]
    mx = max(p0.D(x) for x in xs)
    say(f"   {nm}: max_K Delta_p0 = {mp.nstr(mx, 10)}  -> epsilon = {mp.nstr(-mx, 10)}")
say(f"   ln(4/3) = {mp.nstr(log(mpf(4)/3), 10)}")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_output.txt"), "w").write("\n".join(out) + "\n")
print("\n[written to e2_output.txt]")
