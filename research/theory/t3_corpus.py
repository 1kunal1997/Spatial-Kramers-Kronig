import sys, os, time
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
from t3_clausen import gap_clausen

rng = np.random.default_rng(7); samples=list(pg.sample(240,rng=rng))
agree=0; dis=[]; rels=[]
t0=time.time()
for i,s in enumerate(samples):
    x=s['x']; ei=s['e_im']
    dom,D = gap_clausen(s['e_re'], x)
    scale=max(ei.max(),1e-12)
    rels.append(np.abs(dom-ei).max()/scale)
    v = dom.min() >= -1e-3*scale
    if v==s['passive']: agree+=1
    else: dis.append((i,s['name'],s['params'],round(float(ei.min()),6),round(float(dom.min()),6),round(float(scale),4),s['passive']))
rels=np.array(rels)
print(f"AGREEMENT {agree}/240   ({time.time()-t0:.1f}s)")
print(f"pointwise rel err vs DiHiTI: median {np.median(rels):.2e}  p90 {np.percentile(rels,90):.2e}  max {rels.max():.2e}")
for d in dis: print("  DISAGREE",d)
