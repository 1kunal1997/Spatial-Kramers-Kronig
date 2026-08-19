import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
rng = np.random.default_rng(7); samples=list(pg.sample(240,rng=rng))
for i in [41,46,72,89,162,170,199]:
    s=samples[i]; x=s['x']; L=x[-1]; ei=s['e_im']
    j=np.argmin(ei)
    print(f"{i:4d} {s['name']:15s} min={ei.min():+.6f} at x/L={x[j]/L:+.5f}  ei[0]={ei[0]:.2e} ei[-1]={ei[-1]:.2e}")
    # where is ei negative?
    neg=np.nonzero(ei< -1e-3*max(ei.max(),1e-12))[0]
    if len(neg): print(f"      negative region x/L in [{x[neg[0]]/L:+.5f},{x[neg[-1]]/L:+.5f}], count {len(neg)}")
