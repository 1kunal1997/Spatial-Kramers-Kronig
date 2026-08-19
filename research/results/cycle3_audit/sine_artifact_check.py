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
print("checking rel=1e9-ish entries (scale==1e-12 floor hit)")
for i,s in enumerate(samples):
    x=s['x']; ei=s['e_im']
    dom,D = gap_clausen(s['e_re'], x)
    scale = max(ei.max(), 1e-12)
    rel = np.abs(dom-ei).max()/scale
    if rel > 1e6:
        abs_err = np.abs(dom-ei).max()
        print(f"i={i:3d} name={s['name']:14s} eim_max={ei.max():.3e} eim_min={ei.min():.4f} "
              f"dom_max={dom.max():.3e} dom_min={dom.min():.4f} abs_err={abs_err:.4e} rel={rel:.2e} passive={s['passive']}")
