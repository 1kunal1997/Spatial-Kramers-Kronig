import sys, os
_PROJECT_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
import tmm_helper as tmm_h
import json, subprocess, datetime

nb = 1.7; k_steep = 4.0; L = 2.5

def ht_d1(x_, e_re_):
    u = np.gradient(e_re_, x_)
    v = np.imag(hilbert(u))
    e_im_ = cumulative_trapezoid(v, x_, initial=0)
    e_im_ -= np.linspace(e_im_[0], e_im_[-1], len(e_re_))
    return e_im_

def build(alpha, N):
    xx = np.linspace(-L, L, N)
    window = ((1 + np.cos(np.pi * xx / L)) / 2)**3
    sigma_b = 0.5
    bump_L_loc = np.exp(-((xx + 1.0) / sigma_b)**2) * window
    bump_R_loc = np.exp(-((xx - 1.0) / sigma_b)**2) * window
    bump_L_loc /= bump_L_loc.max()
    bump_R_loc /= bump_R_loc.max()
    e_re = tmm_h.logistic(xx, k_steep, nb) + alpha * (bump_L_loc - bump_R_loc)
    return xx, e_re

def interior_min(alpha, N, edge_excl=0.2):
    """min of eps'' excluding a strip of width edge_excl near each boundary,
    where the de-tilt forces eps''(+/-L)=0 exactly (that's a separate,
    always-zero feature, not the interior dip we're tracking)."""
    xx, e_re = build(alpha, N)
    ei = ht_d1(xx, e_re)
    mask = np.abs(xx) <= (L - edge_excl)
    return ei[mask].min()

# scan first to confirm true (global, whole-domain) min behavior + interior-only behavior
print("global min (whole domain) vs interior-only min (|x|<=L-0.2):")
for a in np.linspace(0.60, 0.70, 11):
    xx, e_re = build(a, 80001)
    ei = ht_d1(xx, e_re)
    gmin = ei.min()
    imin = interior_min(a, 80001)
    print(f"  a={a:.3f}  global_min={gmin:+.6e}  interior_min={imin:+.6e}")

print("\nBisecting true onset of interior negativity (excludes the always-0 endpoint pin):")
results = {}
for N in [10001, 20001, 40001, 80001, 160001]:
    f = lambda a: interior_min(a, N)
    ac = brentq(f, 0.60, 0.70, xtol=1e-12, rtol=1e-13)
    results[N] = ac
    print(f"  N={N}: alpha_c = {ac:.8f}")

Ns = sorted(results)
print("\nConvergence:")
for i in range(1, len(Ns)):
    print(f"  N={Ns[i-1]}->{Ns[i]}: d(alpha_c) = {results[Ns[i]]-results[Ns[i-1]]:.3e}")

alpha_c_best = results[max(Ns)]
print(f"\nTRUE alpha_c (interior onset of gain) = {alpha_c_best:.6f}")

# also record: at what alpha does the GLOBAL min (incl. forced-zero endpoint)
# first dip below the -0.01 threshold used by mismatch_table.py's PASS/FAIL flag
def global_min(alpha, N):
    xx, e_re = build(alpha, N)
    return ht_d1(xx, e_re).min()
g = lambda a: global_min(a, 160001) + 0.01
alpha_flag01 = brentq(g, 0.60, 1.0, xtol=1e-10)
print(f"alpha where global min crosses the -0.01 PASS/FAIL flag threshold: {alpha_flag01:.6f}")

sha = subprocess.run(["git","-C",_PROJECT_ROOT,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
manifest = dict(
    experiment="H6 alpha_c bisection v2 -- true interior onset of negativity, "
               "separated from the endpoint pin (which is exactly 0 by de-tilt construction "
               "for ALL alpha, and dominates the naive whole-domain min for small alpha)",
    timestamp=datetime.datetime.now().isoformat(),
    git_sha=sha,
    script=os.path.abspath(__file__),
    params=dict(nb=nb, k_steep=k_steep, L=L, sigma_b=0.5, bracket=[0.60,0.70], edge_excl=0.2),
    grid_sizes=Ns,
    alpha_c_by_N={str(k): v for k,v in results.items()},
    alpha_c_true=alpha_c_best,
    alpha_c_naive_global_min_at_flag_minus0p01=alpha_flag01,
    note=("mismatch_table.py's reported values (0.0/0.3/0.6 -> min=0.000 PASS; 1.0 -> -0.117; "
          "1.5 -> -0.302) are reproduced exactly by the WHOLE-DOMAIN min. But that min is "
          "identically 0.0 (to double precision) for alpha<=0.665 because it is realized at the "
          "boundary x=+/-L, which the DiHiTI de-tilt pins to exactly 0 for every alpha "
          "(see H7 result). The Captain's 0.6-1.0 bracket is therefore a coarse sample of a "
          "genuinely sharper transition: the INTERIOR dip (near x=+/-1.3) first crosses zero at "
          f"alpha_c = {alpha_c_best:.4f}, not somewhere loosely between 0.6 and 1.0."),
)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_bisection.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print("wrote manifest_bisection.json")
