"""Direct test of the coordinator's concern: does refining N reveal a deeper near-seam
minimum that N=2048 (the corpus default) misses, for ANY of the 240 corpus profiles?
Compares DiHiTI min_eim and gap_clausen min at N=2048 (labeling N) vs N=32768 (fine)."""
import sys, os, time
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
sys.path.insert(0, os.path.join(_ROOT,'research','theory'))
import profile_generator as pg
from t3_clausen import gap_clausen

rng = np.random.default_rng(7)
samples = list(pg.sample(240, rng=rng))   # N=2048 default, this defines corpus labels

t0=time.time()
flips = []
big_gap = []
for i, s in enumerate(samples):
    nm = s['name']; kw = s['params']
    s_fine = pg.make(nm, N=16384, **{k_:v for k_,v in kw.items()})
    scale2048 = max(s['e_im'].max(), 1e-12)
    scale_fine = max(s_fine['e_im'].max(), 1e-12)
    min2048 = s['e_im'].min()
    minfine = s_fine['e_im'].min()
    lbl2048 = s['passive']
    lblfine = bool(minfine >= -1e-3*scale_fine)
    if lbl2048 != lblfine:
        flips.append((i, nm, kw, min2048, minfine, lbl2048, lblfine))
    # also track cases where refinement reveals meaningfully deeper minimum (even if verdict same)
    if minfine < min2048 - 1e-2*scale_fine:
        big_gap.append((i, nm, min2048, minfine))
print(f"done in {time.time()-t0:.1f}s")
print(f"LABEL FLIPS (N=2048 -> N=16384): {len(flips)}")
for f in flips:
    print("  ", f)
print(f"\nProfiles where finer N reveals notably deeper min (>1% scale, same verdict or not): {len(big_gap)}")
for b in big_gap[:20]:
    print("  ", b)
