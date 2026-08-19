import sys, os, time
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
sys.path.insert(0, os.path.join(_ROOT,'research','theory'))
import profile_generator as pg
from t3_clausen import gap_clausen

for seed in [0,1,2,3,7,11,42,999]:
    rng = np.random.default_rng(seed)
    samples = list(pg.sample(400, rng=rng))
    agree=0; dis=[]
    for i,s in enumerate(samples):
        x=s['x']; ei=s['e_im']
        dom,D = gap_clausen(s['e_re'], x)
        scale=max(ei.max(),1e-12)
        v = dom.min() >= -1e-3*scale
        if v==s['passive']: agree+=1
        else: dis.append((s['name'],s['params'],round(float(ei.min()),6),round(float(dom.min()),6)))
    print(f"seed={seed:5d}  AGREEMENT {agree}/400  disagreements={len(dis)}")
    for d in dis[:5]:
        print("   ", d)
