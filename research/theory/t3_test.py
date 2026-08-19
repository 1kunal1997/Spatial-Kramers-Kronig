import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
from t3_clausen import gap_clausen

rng = np.random.default_rng(7); samples=list(pg.sample(240,rng=rng))
for i in [46,72,170,41,0,1]:
    s=samples[i]; x=s['x']; ei=s['e_im']
    dom,D = gap_clausen(s['e_re'], x)
    err = dom-ei
    print(f"{i:4d} {s['name']:15s} scale={ei.max():9.4f}  max|err|={np.abs(err).max():.3e} rel={np.abs(err).max()/max(ei.max(),1e-12):.2e}"
          f"  dihiti_min={ei.min():+.6f} dom_min={dom.min():+.6f}")
