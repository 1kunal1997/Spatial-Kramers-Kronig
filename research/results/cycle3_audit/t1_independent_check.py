"""Fully independent, closed-form/high-precision check of C35 (skew tolerance = 0),
using an exact point-mass (delta) atom + uniform-arc mixture, NOT the theorist's
narrow-rectangle atom construction (t1_verify.py), and mpmath's built-in Clausen
function (not the trigamma-DFT route used by t3_clausen.py / t1_arcs.py).
"""
import mpmath as mp
mp.mp.dps = 40
PI = mp.pi

def Cl2(theta):
    return mp.clsin(2, theta)

def Phi_arc(theta, p, q):
    return (Cl2(theta-q) - Cl2(theta-p)) / (q-p)

def M1_arc(p, q):
    return mp.log(mp.cos(p/2)/mp.cos(q/2)) / ((q-p)/2)

def Phi_delta(theta, c):
    return mp.log(abs(2*mp.sin((theta-c)/2)))

def M1_delta(c):
    return mp.tan(c/2)

def analyze(cf, qf, verbose=True):
    c = cf*PI; q = qf*PI; p = c   # arc starts exactly at the atom location, as in t1_verify
    a = M1_delta(c)
    b = M1_arc(p, q)
    lam = -b/(a-b)
    if not (0 <= lam <= 1):
        print(f"c={cf}pi q={qf}pi: infeasible lam={lam}")
        return None
    def Phi_mu(theta):
        return lam*Phi_delta(theta, c) + (1-lam)*Phi_arc(theta, p, q)
    M1_mu = lam*a + (1-lam)*b   # should be EXACTLY 0 by construction (closed form)
    # find max over theta in (-pi,pi) excluding the two singular points c and q (log -> -inf there, not max)
    # scan finely then refine with golden-section / mp.findroot on derivative
    n = 4000
    thetas = [-PI + 2*PI*mp.mpf(i)/n for i in range(1, n)]
    vals = []
    for th in thetas:
        if abs(th-c) < mp.mpf('1e-6') or abs(th-q) < mp.mpf('1e-6'):
            continue
        vals.append((Phi_mu(th), th))
    vals.sort(key=lambda t: -t[0])
    best_val, best_th = vals[0]
    # local refine via ternary search around best_th
    lo, hi = best_th - 2*PI/n, best_th + 2*PI/n
    for _ in range(60):
        m1_ = lo + (hi-lo)/3; m2_ = hi - (hi-lo)/3
        if Phi_mu(m1_) < Phi_mu(m2_):
            lo = m1_
        else:
            hi = m2_
    th_star = (lo+hi)/2
    Phi_star = Phi_mu(th_star)
    Phi_pi = Phi_mu(PI)
    gain = Phi_star - Phi_pi
    print(f"c={cf:+.3f}pi q={qf:.5f}pi  lam={float(lam):.6f}  M1(exact)={mp.nstr(M1_mu,6)}  "
          f"theta*/pi={float(th_star/PI):+.5f}  gain=Phi(theta*)-Phi(pi) = {mp.nstr(gain,10)}")
    return gain

print("=== c=-pi/2, varying q -> pi ===")
for qf in [0.90, 0.97, 0.995, 0.999, 0.9999, 0.99999]:
    analyze(-0.5, qf)

print()
print("=== other mode positions ===")
for cf, qf in [(-0.25, 0.995), (-0.75, 0.99), (-0.1, 0.999)]:
    analyze(cf, qf)

print()
print("=== does gain appear for ARBITRARILY SMALL skew c, as q -> pi? (tests 'tolerance IS zero') ===")
for cf in [-0.30, -0.10, -0.03, -0.01, -0.003]:
    print(f"--- c = {cf}pi ---")
    for qf in [0.999, 0.9999, 0.99999, 0.999999]:
        analyze(cf, qf)
