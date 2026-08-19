import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
sys.path.insert(0, os.path.join(_ROOT,'research','theory'))
import profile_generator as pg
from t3_clausen import gap_clausen

rng = np.random.default_rng(7)
samples = list(pg.sample(240, rng=rng))
rows=[]
for i,s in enumerate(samples):
    x=s['x']; ei=s['e_im']
    g = -np.gradient(s['e_re'], x)
    D = np.trapezoid(g,x)
    L = x[-1]
    xi = np.tan(np.pi*x/(2*L))
    # M1 in eta-space per C8: int g(t) xi(t) dt  (moment w.r.t. tan)
    M1 = np.trapezoid(g*xi, x)
    scale = max(ei.max(),1e-12)
    M1_norm = abs(M1)/max(abs(D),1e-12)
    argmin_frac = abs(x[np.argmin(ei)])/L
    rows.append((i, s['name'], M1, M1_norm, D, argmin_frac, ei.min(), s['passive']))

rows.sort(key=lambda r: r[3])
print("10 smallest |M1|/|D| (closest to knife edge):")
for r in rows[:15]:
    print(f"  i={r[0]:3d} {r[1]:14s} M1={r[2]:+.4e} |M1|/|D|={r[3]:.4e} D={r[4]:+.3f} argmin|x|/L={r[5]:.5f} min_eim={r[6]:+.4e} passive={r[7]}")
