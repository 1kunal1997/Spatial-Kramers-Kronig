"""T1: worst-case gain over ext{unimodal about phi_m, M1=0} via differential evolution."""
import numpy as np, sys
from scipy.optimize import differential_evolution
from t1_fast import cl2, phi_arc, m1_arc, PI
TH = np.linspace(-PI, PI, 3001)

def gain_of(p1, q1, p2, q2):
    w1, w2 = (q1-p1)/2, (q2-p2)/2
    if w1 < 1e-8 or w2 < 1e-8: return 0.0
    c1, c2 = (p1+q1)/2, (p2+q2)/2
    a, b = m1_arc(c1, w1), m1_arc(c2, w2)
    if abs(a-b) < 1e-14: return 0.0
    lam = -b/(a-b)
    if not (0.0 <= lam <= 1.0): return 0.0
    P = lam*phi_arc(TH, c1, w1) + (1-lam)*phi_arc(TH, c2, w2)
    Pi = lam*phi_arc(np.array(PI), c1, w1) + (1-lam)*phi_arc(np.array(PI), c2, w2)
    return float(P.max() - Pi)

def worst(phim, seed=0, iters=400):
    E = 1e-7
    def neg(u):
        # u in [0,1]^4 -> p1,q1,p2,q2
        p1 = -PI+E + u[0]*(phim + PI - E)
        q1 = phim + u[1]*(PI-E-phim)
        p2 = -PI+E + u[2]*(phim + PI - E)
        q2 = phim + u[3]*(PI-E-phim)
        return -gain_of(p1, q1, p2, q2)
    r = differential_evolution(neg, [(0,1)]*4, seed=seed, maxiter=iters, popsize=30,
                               tol=1e-12, polish=True)
    u = r.x
    p1 = -PI+E + u[0]*(phim+PI-E); q1 = phim + u[1]*(PI-E-phim)
    p2 = -PI+E + u[2]*(phim+PI-E); q2 = phim + u[3]*(PI-E-phim)
    return -r.fun, (p1, q1, p2, q2)

if __name__ == '__main__':
    print(f"{'phi_m':>7} {'phi_m/pi':>9} {'max gain':>12}   worst arcs (p1,q1 | p2,q2)")
    for phim in [float(v) for v in sys.argv[1:]] or [0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0]:
        g, cfg = max((worst(phim, s) for s in (0,1,2)), key=lambda z: z[0])
        print(f"{phim:7.3f} {phim/PI:9.3f} {g:12.6f}   ({cfg[0]:+.4f},{cfg[1]:+.4f} | {cfg[2]:+.4f},{cfg[3]:+.4f})")
