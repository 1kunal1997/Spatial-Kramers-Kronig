import sys, os, time
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
from t3_clausen import gap_clausen

tot=0; ag=0; alld=[]
for seed in [1,2,3,7,11,23]:
    rng=np.random.default_rng(seed); a=0; n=0; dis=[]
    for s in pg.sample(400, rng=rng):
        x=s['x']; ei=s['e_im']; dom,_=gap_clausen(s['e_re'],x)
        sc=max(ei.max(),1e-12)
        # blind indicator: use the same relative tolerance the labeller uses
        v=dom.min()>=-1e-3*sc
        n+=1
        if v==s['passive']: a+=1
        else: dis.append((s['name'],round(float(ei.min()),6),round(float(dom.min()),6),round(float(sc),4)))
    print(f"seed {seed}: {a}/{n}", "" if not dis else dis)
    tot+=n; ag+=a; alld+=dis
print(f"TOTAL {ag}/{tot}")
