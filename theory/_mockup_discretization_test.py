"""Clinch the cause of the large-d/lambda kick-up: refine discretization and
watch R_sKK in the kick-up region. If it's a staircase (discretization)
artifact, finer delta -> more layers -> lower R there. Throwaway."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import tmm_helper as tmm_h
from tmm_helper import _make_c_list

nb = 1.7; T = 5.0; k = 40.0 / T
dx = 1 / (100 * k); xmin = -20 / k; xmax = -xmin
nx = 1 + int(np.floor((xmax - xmin) / dx))
xx = np.linspace(xmin, xmax, nx)
e_re = tmm_h.logistic(xx, k, nb)
e_im = tmm_h.ht_derivative(xx, e_re)
ee = e_re + 1j * e_im

deltas = [0.02, 0.01, 0.005, 0.0025, 0.00125]
dlam_vals = [10, 20, 50, 100]

print(f"{'delta':>8} {'Nlay':>6} {'max d_lay/lam@dl=100':>20} | " +
      " ".join(f"R_sKK(d/l={r})".rjust(14) for r in dlam_vals) +
      " | " + " ".join(f"inc@{r}".rjust(7) for r in dlam_vals))
for delta in deltas:
    nc, dc = tmm_h.discretize_profile(xx, ee, delta=delta)
    n_t = [nb] + list(nc) + [1.0]
    d_t = [np.inf] + list(dc) + [np.inf]
    Nlay = len(nc)
    max_dlay = max(dc)                      # thickest layer (um)
    Rs, incs = [], []
    for r in dlam_vals:
        lam = T / r
        _, R, _ = tmm_h.TRA(n_t, d_t, lamb=lam, angle=0.0, pol='s')
        Rs.append(R)
        c = _make_c_list(n_t, d_t, lam, 0.0)
        incs.append(c[1:-1].count('i'))
    # max electrical layer thickness at d/lam=100 (lam=0.05)
    max_elec = max(np.real(n) * d / (T / 100) for n, d in zip(nc, dc))
    print(f"{delta:>8.5f} {Nlay:>6d} {max_elec:>20.3f} | " +
          " ".join(f"{R:>14.3e}" for R in Rs) +
          " | " + " ".join(f"{i:>7d}" for i in incs))

print("\nIf R_sKK at d/l=20,50,100 falls as delta shrinks -> discretization artifact confirmed.")
print("(inc columns = # incoherent coating layers; max d_lay/lam@100 = thickest layer in wavelengths at d/l=100)")
