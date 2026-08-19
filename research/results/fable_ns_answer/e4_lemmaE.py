"""E4 — Lemma E: the active-point set A* contains an interval (envelope construction).

Three pairs: bad1, bad2, good. For each target theta, solve the 2x3 homogeneous system
    [ Delta_1(theta)  Delta_2(theta)  Delta_g(theta) ] m = 0
    [ Delta'_1(theta) Delta'_2(theta) Delta'_g(theta) ] m = 0
for the 1-dim nullspace m(theta). If m(theta) can be normalized componentwise-positive
AND the global sup of Delta_{m(theta)} equals 0 attained at theta (checked by the exact
node test), then theta is the active point of a genuine boundary mixture: theta in A*.

Sweep theta and report the window where this holds. Any subinterval => Lemma E.

Control: at each accepted theta, sup(Delta) must be 0 to solver precision AND the
argmax must equal theta to 1e-6 — two independent consistency checks on the envelope.
"""
import numpy as np
from common import balanced_pair, exact_sup_gap, Atomic

def pair_Dprime(pair, eta):
    """Closed-form Delta'(eta) for a unit-mass balanced pair (atoms xi = beta, -gam)."""
    beta = pair.xi[0]; gam = -pair.xi[1]
    return (eta * (1 + beta * gam) + (gam - beta)) / \
           ((eta - beta) * (eta + gam) * (1 + eta**2))

bad1 = balanced_pair(0.25, 0.90)   # sum 1.15
bad2 = balanced_pair(0.40, 0.80)   # sum 1.20
good = balanced_pair(0.15, 0.45)   # sum 0.60
pairs = [bad1, bad2, good]

def mix(m):
    t = np.concatenate([p.t for p in pairs])
    w = np.concatenate([mi * p.w for mi, p in zip(m, pairs)])
    return Atomic(t, w)

accepted = []
thetas = np.linspace(0.2, 3.0, 281)
for th in thetas:
    row0 = np.array([p.Delta(np.array([th]))[0] for p in pairs])
    row1 = np.array([pair_Dprime(p, th) for p in pairs])
    A = np.vstack([row0, row1])
    _, _, Vt = np.linalg.svd(A)
    m = Vt[-1]
    if np.all(m < 0):
        m = -m
    if np.all(m > 1e-10):
        g = mix(m)
        sup, at = exact_sup_gap(g)
        if abs(sup) < 1e-10 and abs(at - th) < 1e-6:
            # nondegeneracy of the touching maximum (needed by the thickening step)
            h = 1e-5
            d2 = (g.Delta(np.array([th - h]))[0] - 2 * sup
                  + g.Delta(np.array([th + h]))[0]) / h**2
            accepted.append((th, m / m[2], d2))

if accepted:
    ths = np.array([a[0] for a in accepted])
    # report the largest contiguous run
    print(f"accepted theta values: {len(accepted)} of {len(thetas)}")
    print(f"contiguous window(s): [{ths.min():.3f}, {ths.max():.3f}] "
          f"(gaps: {np.any(np.diff(ths) > 1.5*(thetas[1]-thetas[0]))})")
    d2s = np.array([a[2] for a in accepted])
    print(f"touching-max nondegeneracy: Delta'' in [{d2s.min():.3f}, {d2s.max():.3f}] "
          f"(all < 0: {np.all(d2s < 0)})")
    for th, m, d2 in accepted[:: max(1, len(accepted)//8)]:
        print(f"  theta = {th:.3f}: masses (bad1,bad2,good)/good = "
              f"({m[0]:.4f}, {m[1]:.4f}, 1)   Delta''={d2:+.3f}")
else:
    print("no theta accepted — construction failed for this pair triple")
