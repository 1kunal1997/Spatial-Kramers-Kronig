"""Fast independent check: for very small skew c, does gain appear for q close enough to pi?
Uses derivative root-finding instead of brute scan (much faster), dps=25."""
import mpmath as mp
mp.mp.dps = 25
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

def analyze(cf, qf, guess_frac=0.7):
    c = cf*PI; q = qf*PI; p = c
    a = M1_delta(c); b = M1_arc(p, q)
    lam = -b/(a-b)
    if not (0 <= lam <= 1):
        return None
    def Phi_mu(theta):
        return lam*Phi_delta(theta, c) + (1-lam)*Phi_arc(theta, p, q)
    def dPhi(theta):
        d1 = mp.mpf('0.5')*mp.cot((theta-c)/2)
        d2 = mp.log(abs(mp.sin((theta-p)/2))/abs(mp.sin((theta-q)/2)))/(q-p)
        return lam*d1 + (1-lam)*d2
    # coarse scan for sign change of dPhi away from singularities c,q
    n = 300
    thetas = [-PI + 2*PI*mp.mpf(i)/n for i in range(1,n)]
    best = None
    prev = None; prevth=None
    for th in thetas:
        if abs(th-c)<mp.mpf('1e-3') or abs(th-q)<mp.mpf('1e-3'):
            prev=None; continue
        v = dPhi(th)
        if prev is not None and prev>0 and v<0:
            # bracket a local max
            try:
                r = mp.findroot(dPhi, (prevth+th)/2)
                val = Phi_mu(r)
                if best is None or val>best[0]:
                    best=(val, r)
            except Exception:
                pass
        prev=v; prevth=th
    if best is None:
        return None
    Phi_star, th_star = best
    gain = Phi_star - Phi_mu(PI)
    print(f"c={cf:+.4f}pi q={qf:.6f}pi lam={float(lam):.5f} theta*/pi={float(th_star/PI):+.4f} gain={mp.nstr(gain,8)}")
    return gain

print("=== small skew c, q -> pi ===")
for cf in [-0.30, -0.10, -0.03, -0.01, -0.003, -0.001]:
    print(f"--- c={cf}pi ---")
    for qf in [0.999, 0.9999, 0.99999]:
        analyze(cf, qf)
