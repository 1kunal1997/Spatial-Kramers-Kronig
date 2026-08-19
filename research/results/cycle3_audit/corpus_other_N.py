import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
sys.path.insert(0, os.path.join(_ROOT,'research','theory'))
import profile_generator as pg
from t3_clausen import gap_clausen

for N in [513, 1025, 2049, 4097, 8193]:
    rng = np.random.default_rng(7)
    agree=0; dis=[]; total=0
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
        dom,D = gap_clausen(s['e_re'], x)
        scale=max(ei.max(),1e-12)
        v = dom.min() >= -1e-3*scale
        if v==s['passive']: agree+=1
        else: dis.append((nm, kw, round(float(ei.min()),6), round(float(dom.min()),6)))
    print(f"N={N:6d}  AGREEMENT {agree}/{total}")
    for d in dis[:6]:
        print("   ", d)
