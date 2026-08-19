import sys, numpy as np
from scipy.interpolate import CubicSpline
import mpmath as mp
mp.mp.dps = 40
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory/passivity_2026Aug")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/theory")
import tmm_helper as tmm_h
import profile_generator as pg
from t3_clausen import gap_clausen

NB, k = mp.mpf('1.7'), mp.mpf('4.0')
L = mp.mpf('2.5')
A = NB**2 - 1
def g(t):
    e = mp.e**(k*t)
    return A*k*e/(1+e)**2
def kernel_true(t, xpt, L):
    return mp.log(abs(mp.cos(mp.pi*t/(2*L)) / mp.sin(mp.pi*(xpt-t)/(2*L))))
def eps_im_true(xpt, L):
    xpt = mp.mpf(xpt)
    f = lambda t: g(t)*kernel_true(t, xpt, L)
    val = mp.quad(f, [-L, xpt, L])
    return val/mp.pi

test_x = [-1.5, -0.7, -0.2, 0.3, 0.9, 1.7]
truth = {xp: float(eps_im_true(xp, L)) for xp in test_x}

print("=== non-power-of-two N: convergence order check ===")
prev=None
for N in [501, 1001, 2003, 4001]:   # M = N-1: 500,1000,2002,4000 -- not powers of 2
    x = np.linspace(-2.5, 2.5, N)
    e_re = tmm_h.logistic(x, 4.0, 1.7)
    dom, D = gap_clausen(e_re, x)
    cs = CubicSpline(x, dom)
    errs = [abs(float(cs(xp)) - truth[xp]) for xp in test_x]
    m = max(errs)
    ratio = (prev/m) if prev else float('nan')
    print(f"N={N:6d} M={N-1:6d}  max err = {m:.4e}  ratio~{ratio}")
    prev=m

print()
print("=== corpus verdict agreement at non-pow2 N ===")
for N in [501, 1001, 2003, 4001, 4999]:
    rng = np.random.default_rng(7)
    agree=0; total=0; dis=[]
    names = sorted(pg.PROFILES)
    for _ in range(200):
        nm = names[rng.integers(len(names))]
        kw = {}
        for p,(a,b) in pg.RANGES.get(nm,{}).items():
            v = rng.uniform(a,b)
            if p=='sign': v = 1.0 if v>0 else -1.0
            kw[p]=v
        s = pg.make(nm, N=N, **kw)
        total+=1
        x=s['x']; ei=s['e_im']
        dom,Dd = gap_clausen(s['e_re'], x)
        scale=max(ei.max(),1e-12)
        v = dom.min() >= -1e-3*scale
        if v==s['passive']: agree+=1
        else: dis.append((nm,kw))
    print(f"N={N:6d}  AGREEMENT {agree}/{total}", "disagree:" if dis else "")
    for d in dis[:5]: print("   ", d)
