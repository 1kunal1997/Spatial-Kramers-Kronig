import sys, os, time
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
sys.path.insert(0, os.path.join(_ROOT,'research','theory'))
import profile_generator as pg
from t3_clausen import gap_clausen

rng = np.random.default_rng(7)
samples = list(pg.sample(240, rng=rng))
t0=time.time()
flips=[]; big=[]
for i,s in enumerate(samples):
    nm=s['name']; kw=s['params']
    s_fine = pg.make(nm, N=32768, **kw)
    scale_fine = max(s_fine['e_im'].max(),1e-12)
    lblfine = bool(s_fine['e_im'].min() >= -1e-3*scale_fine)
    if lblfine != s['passive']:
        flips.append((i,nm,kw,s['e_im'].min(),s_fine['e_im'].min()))
    if s_fine['e_im'].min() < s['e_im'].min() - 1e-2*scale_fine:
        big.append((i,nm,s['e_im'].min(),s_fine['e_im'].min()))
print(f"N=32768 DiHiTI refine, {time.time()-t0:.1f}s: flips={len(flips)} big_gap={len(big)}")
for f in flips: print("  FLIP", f)
for b in big[:20]: print("  BIG", b)
