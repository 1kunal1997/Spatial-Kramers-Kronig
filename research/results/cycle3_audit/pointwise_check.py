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
rms_all = []
for s in samples:
    x=s['x']; ei=s['e_im']
    dom,D = gap_clausen(s['e_re'], x)
    interior = np.abs(x) < 0.95*x[-1]
    rms = np.sqrt(np.mean((dom[interior]-ei[interior])**2))
    scale = max(np.abs(ei[interior]).max(), 1e-9)
    rms_all.append(rms/scale)
rms_all = np.array(rms_all)
print("full-array interior RMS rel err across corpus:")
print(" median", np.median(rms_all), " p90", np.percentile(rms_all,90), " max", rms_all.max())

# interior-relabel robustness (addendum claim)
print()
print("relabel by interior min (|x|<0.97L) vs original whole-domain label:")
changed=0
for s in samples:
    x=s['x']; ei=s['e_im']
    interior = np.abs(x) < 0.97*x[-1]
    scale = max(float(ei.max()),1e-12)
    lbl_interior = bool(ei[interior].min() >= -1e-3*scale)
    if lbl_interior != s['passive']:
        changed += 1
print(f"changed labels: {changed}/240")
