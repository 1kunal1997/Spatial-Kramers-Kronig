import sys, os
_PROJECT_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import tmm_helper as tmm_h

nb = 1.7
k_steep = 4.0
L = 2.5
N = 40001
xx = np.linspace(-L, L, N)

window = ((1 + np.cos(np.pi * xx / L)) / 2)**3
sigma_b = 0.5
bump_L_loc = np.exp(-((xx + 1.0) / sigma_b)**2) * window
bump_R_loc = np.exp(-((xx - 1.0) / sigma_b)**2) * window
bump_L_loc /= bump_L_loc.max()
bump_R_loc /= bump_R_loc.max()

def ht_d1(x_, e_re_):
    u = np.gradient(e_re_, x_)
    v = np.imag(hilbert(u))
    e_im_ = cumulative_trapezoid(v, x_, initial=0)
    e_im_ -= np.linspace(e_im_[0], e_im_[-1], len(e_re_))
    return e_im_

for ao in [0.0, 0.3, 0.6, 1.0, 1.5]:
    e_re_opp = tmm_h.logistic(xx, k_steep, nb) + ao * (bump_L_loc - bump_R_loc)
    e_im_opp = ht_d1(xx, e_re_opp)
    print(f"alpha={ao:4.2f}  min eps''={e_im_opp.min():+.4f}  argmin x={xx[np.argmin(e_im_opp)]:+.4f}")

print("\n--- symmetry check ---")
for ao in [1.0, 1.5]:
    e_re_opp = tmm_h.logistic(xx, k_steep, nb) + ao * (bump_L_loc - bump_R_loc)
    e_im_opp = ht_d1(xx, e_re_opp)
    asym = np.max(np.abs(e_im_opp - e_im_opp[::-1]))
    print(f"alpha={ao}: max|eps''(x)-eps''(-x)| = {asym:.3e}, min={e_im_opp.min():.5f}")
    i1 = np.argmin(e_im_opp)
    i2 = np.argmin(e_im_opp[::-1])
    print(f"   val at argmin={e_im_opp[i1]:.5f} (x={xx[i1]:.3f}); mirrored point val={e_im_opp[N-1-i1]:.5f}")
