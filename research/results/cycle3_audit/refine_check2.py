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
    s_fine = pg.make(nm, N=131072, **kw)
    scale_fine = max(s_fine['e_im'].max(),1e-12)
    lblfine = bool(s_fine['e_im'].min() >= -1e-3*scale_fine)
    if lblfine != s['passive']:
        flips.append((i,nm,kw,s['e_im'].min(),s_fine['e_im'].min()))
    if s_fine['e_im'].min() < s['e_im'].min() - 1e-2*scale_fine:
        big.append((i,nm,s['e_im'].min(),s_fine['e_im'].min()))
print(f"N=131072 refine, {time.time()-t0:.1f}s: flips={len(flips)} big_gap={len(big)}")
for f in flips: print("  FLIP", f)
for b in big[:20]: print("  BIG", b)

# also check gap_clausen at N=131072 vs corpus's own gap_clausen at N=2048
print()
print("gap_clausen verdict robustness under refinement:")
flips2=[]
for i,s in enumerate(samples):
    nm=s['name']; kw=s['params']
    x2 = s['x']
    dom2048,_ = gap_clausen(s['e_re'], x2)
    x_fine = pg.grid(pg.L_DEF, 131072)
    fn,defaults = pg.PROFILES[nm]
    kwall = dict(defaults); kwall.update(kw)
    e_re_fine = np.asarray(fn(x_fine, **kwall), float)
    domfine,_ = gap_clausen(e_re_fine, x_fine)
    scale = max(s['e_im'].max(),1e-12)
    v2048 = dom2048.min() >= -1e-3*scale
    vfine = domfine.min() >= -1e-3*max(scale,1e-12)
    if v2048 != vfine:
        flips2.append((i,nm,kw,dom2048.min(),domfine.min()))
print(f"gap_clausen flips under refinement: {len(flips2)}")
for f in flips2: print("  ", f)
