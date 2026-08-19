import sys, os
_PROJECT_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import tmm_helper as tmm_h

nb = 1.7; k_steep = 4.0; L = 2.5; N = 40001
xx = np.linspace(-L, L, N)
window = ((1 + np.cos(np.pi * xx / L)) / 2)**3
sigma_b = 0.5
bump_L_loc = np.exp(-((xx + 1.0) / sigma_b)**2) * window
bump_R_loc = np.exp(-((xx - 1.0) / sigma_b)**2) * window
bump_L_loc /= bump_L_loc.max()
bump_R_loc /= bump_R_loc.max()

def eps_total(alpha):
    return tmm_h.logistic(xx, k_steep, nb) + alpha * (bump_L_loc - bump_R_loc)

def g_of(alpha):
    e = eps_total(alpha)
    return -np.gradient(e, xx)

for alpha in [0.0, 0.6, 1.0, 1.5]:
    g = g_of(alpha)
    even_err = np.max(np.abs(g - g[::-1]))
    # print g at sampled s values (x>=0 half)
    idx_half = xx >= 0
    s = xx[idx_half]; h = g[idx_half]
    # find local extrema in h(s)
    dh = np.gradient(h, s)
    sign_changes = np.sum(np.diff(np.sign(dh)) != 0)
    print(f"alpha={alpha}: even_err={even_err:.2e}, g(0)={g[N//2]:.4f}, "
          f"h(L)={h[-1]:.4f}, #sign-changes-in-h'={sign_changes}, "
          f"h_min={h.min():.4f} at s={s[np.argmin(h)]:.3f}, h_max={h.max():.4f} at s={s[np.argmax(h)]:.3f}")
