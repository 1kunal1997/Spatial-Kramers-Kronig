"""
Second (production-pipeline) route for C49 blind replication:
build the atomic drop measure as a smoothed decreasing staircase eps'(x) and
push it through tmm_helper.ht_derivative (derivative -> Hilbert -> integrate
-> de-tilt), exactly as the project's own scripts do.

This is used as a CROSS-CHECK against the analytic closed-form log-potential
route (analytic_route.py), which is the primary/strongest route since it is
an exact closed form with no grid or smoothing at all.

Deliberate choices made to dodge the stated harness traps:
 - grid: x = -L + 2L*(arange(N)+0.5)/N   (cell-centered, NO endpoint duplication,
   avoids the np.linspace(-L,L,N) periodic-endpoint bug flagged in PROTOCOL.md)
 - tested at N = 4001 (odd, not a power of two) and N = 8192 (power of two) and
   N = 6000 (non-power-of-two, even) to check the verdict is not an idiom artifact
 - staircase jumps are smoothed with erf transitions of width s, s varied
   (0.002, 0.005, 0.01, 0.02) to check verdict stability against smoothing choice
 - battery functionals are NOT recomputed from this noisy grid version; they are
   exact atom sums (see analytic_route.py) since phi_2 = tan(pi t/2) etc. would
   blow up near t->1 if evaluated carelessly on a grid -- avoided here entirely.
"""
import sys, os
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import json
import numpy as np
from scipy.special import erf
import tmm_helper as tmm_h

IN_PATH = "/Users/kunal/Documents/Spatial KK Project/research/results/c49_blind_replication/witness_input.json"
with open(IN_PATH) as f:
    data = json.load(f)
atoms = np.array(data["atoms_t"], dtype=float)
wA = np.array(data["weights_A"], dtype=float)
wB = np.array(data["weights_B"], dtype=float)
L = 1.0

def staircase(x, atoms, w, s):
    """Decreasing staircase: eps'(x) = C - sum_j w_j * smoothstep((x-t_j)/s).
    smoothstep = 0.5*(1+erf(z/sqrt2)) approximates a step of height 1 at t_j,
    width ~ s. As s->0 this -> ideal Heaviside staircase; derivative -> -sum w_j delta(t-t_j).
    """
    out = np.zeros_like(x)
    for tj, wj in zip(atoms, w):
        out = out - wj * 0.5 * (1 + erf((x - tj) / (s * np.sqrt(2))))
    return out

def eps2_analytic_v(x, atoms, w, L=1.0):
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    for tj, wj in zip(atoms, w):
        num = np.cos(np.pi * tj / (2 * L))
        den = np.abs(np.sin(np.pi * (x - tj) / (2 * L)))
        with np.errstate(divide='ignore'):
            term = wj * np.log(num / den)
        total = total + term
    return total / np.pi

def interior_min_from_grid(x, e2, atoms, edge_margin=0.02, atom_excl=0.01):
    mask = (x > -L + edge_margin) & (x < L - edge_margin)
    for tj in atoms:
        mask &= (np.abs(x - tj) > atom_excl)
    xm, em = x[mask], e2[mask]
    idx = np.argmin(em)
    return xm[idx], em[idx]

print(f"{'N':>6} {'s':>7} | {'A min':>10} {'A@x':>8} verdict | {'B min':>10} {'B@x':>8} verdict")
results = []
for N in [4001, 6000, 8192, 16001]:
    x = -L + 2 * L * (np.arange(N) + 0.5) / N   # cell-centered, no endpoint dup
    for s in [0.02, 0.01, 0.005, 0.002]:
        eA = staircase(x, atoms, wA, s)
        eB = staircase(x, atoms, wB, s)
        e2A = tmm_h.ht_derivative(x, eA)
        e2B = tmm_h.ht_derivative(x, eB)
        xA_min, vA = interior_min_from_grid(x, e2A, atoms)
        xB_min, vB = interior_min_from_grid(x, e2B, atoms)
        vdA = "GAIN" if vA < -1e-6 else "PASSIVE"
        vdB = "GAIN" if vB < -1e-6 else "PASSIVE"
        print(f"{N:6d} {s:7.3f} | {vA:10.5f} {xA_min:8.4f} {vdA:7s} | {vB:10.5f} {xB_min:8.4f} {vdB:7s}")
        results.append(dict(N=N, s=s, A_min=vA, A_x=xA_min, A_verdict=vdA,
                             B_min=vB, B_x=xB_min, B_verdict=vdB))

# compare a representative (N=8192, s=0.01) grid-based e2A to the analytic curve directly (RMS)
N = 8192; s = 0.01
x = -L + 2 * L * (np.arange(N) + 0.5) / N
eA = staircase(x, atoms, wA, s)
e2A_grid = tmm_h.ht_derivative(x, eA)
e2A_analytic = eps2_analytic_v(x, atoms, wA)
mask = np.ones_like(x, dtype=bool)
for tj in atoms:
    mask &= (np.abs(x - tj) > 0.02)
mask &= (np.abs(x) < 0.98)
rel_rms = np.sqrt(np.mean((e2A_grid[mask]-e2A_analytic[mask])**2)) / np.sqrt(np.mean(e2A_analytic[mask]**2))
print()
print("Grid-based ht_derivative vs analytic closed form, rel RMS (masked away from atoms/edges):", rel_rms)

import pickle
with open("/private/tmp/claude-501/-Users-kunal-Documents-Spatial-KK-Project/66ff7cc8-80e8-4b98-8d9f-eac225d1f9cd/scratchpad/ht_results.pkl", "wb") as f:
    pickle.dump(results, f)
