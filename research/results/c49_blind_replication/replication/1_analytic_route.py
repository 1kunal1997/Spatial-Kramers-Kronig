"""
Independent analytic route for C49 blind replication.

Route: closed-form logarithmic-potential sum (Wave-2 master identity, quoted
in research/MISSION.md, foundational assumption #3):

    eps''(x) = (1/pi) * integral g(t) * ln[ cos(pi t / 2L) / |sin(pi (x-t)/2L)| ] dt

For an atomic drop measure g = sum_j w_j delta(t - t_j), this collapses to an
EXACT closed-form sum with NO grid, NO FFT, NO smoothing:

    eps''(x) = (1/pi) * sum_j w_j * ln[ cos(pi t_j / 2L) / |sin(pi (x - t_j)/2L)| ]

This is Rank-1 (analytic closed form) and structurally independent of the
FFT-based derivative->Hilbert->integrate machinery in tmm_helper.ht_derivative
(which is used separately, in a second script, as the "production pipeline"
cross-check).
"""
import json
import numpy as np

IN_PATH = "/Users/kunal/Documents/Spatial KK Project/research/results/c49_blind_replication/witness_input.json"

with open(IN_PATH) as f:
    data = json.load(f)

atoms = np.array(data["atoms_t"], dtype=float)
wA = np.array(data["weights_A"], dtype=float)
wB = np.array(data["weights_B"], dtype=float)
L = 1.0

def eps2_analytic(x, atoms, w, L=1.0):
    """Closed-form log-potential eps''(x) for atomic drop measure. x scalar or array,
    must avoid exact atom coincidence (log singularity there)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    for tj, wj in zip(atoms, w):
        num = np.cos(np.pi * tj / (2 * L))
        den = np.abs(np.sin(np.pi * (x[:, None] - tj) / (2 * L))) if x.ndim == 1 else None
    # vectorized properly below
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

# --- battery functionals, direct exact sums over atoms (no grid at all) ---
def battery(atoms, w):
    t = atoms
    phi = [
        np.ones_like(t),                       # phi1 total mass
        np.tan(np.pi * t / 2),                 # phi2 = M1
        1.0 / np.cos(np.pi * t / 2) ** 2,       # phi3 = M2 (sec^2)
        np.log(2 * np.cos(np.pi * t / 2)),      # phi4 = sigma
        t,                                      # phi5
        t ** 2,                                 # phi6
        t ** 3,                                 # phi7
        t ** 4,                                 # phi8
    ]
    return np.array([np.sum(w * p) for p in phi])

battA = battery(atoms, wA)
battB = battery(atoms, wB)

print("Battery A:", battA)
print("Battery B:", battB)
rel_diff = np.abs(battA - battB) / (np.abs(battA) + 1e-30)
print("abs diff:", np.abs(battA - battB))
print("max abs diff:", np.max(np.abs(battA - battB)))

# --- interior minimum of eps'' via dense analytic evaluation ---
# exclude a tiny margin near +-1 (structural zero) and exclude near-atom points
# (log singularity, always a LOCAL MAX not a min for w_j>0, so excluding them
# is safe and conservative for finding the minimum)

def interior_min(atoms, w, L=1.0, n=200001, margin=1e-3, atom_excl=2e-4):
    xs = np.linspace(-L + margin, L - margin, n)
    # mask out points too close to any atom (singularity)
    mask = np.ones_like(xs, dtype=bool)
    for tj in atoms:
        mask &= (np.abs(xs - tj) > atom_excl)
    xs_m = xs[mask]
    vals = eps2_analytic_v(xs_m, atoms, w, L)
    idx = np.argmin(vals)
    return xs_m[idx], vals[idx], xs_m, vals

xA_min, valA_min, xsA, valsA = interior_min(atoms, wA)
xB_min, valB_min, xsB, valsB = interior_min(atoms, wB)

print()
print(f"A: min eps'' (interior) = {valA_min:.6f} at x = {xA_min:.6f}  -> {'PASSIVE' if valA_min >= -1e-9 else 'GAIN'}")
print(f"B: min eps'' (interior) = {valB_min:.6f} at x = {xB_min:.6f}  -> {'PASSIVE' if valB_min >= -1e-9 else 'GAIN'}")

# Convergence check: vary n and margin
for n in [20001, 50001, 200001, 500001]:
    xA_min, valA_min, _, _ = interior_min(atoms, wA, n=n)
    xB_min, valB_min, _, _ = interior_min(atoms, wB, n=n)
    print(f"n={n}: A min={valA_min:.6f} @ {xA_min:.5f} | B min={valB_min:.6f} @ {xB_min:.5f}")

# Check endpoint behavior -> should -> 0
for eps in [1e-2, 1e-4, 1e-6]:
    vA = eps2_analytic_v(np.array([1 - eps, -1 + eps]), atoms, wA)
    vB = eps2_analytic_v(np.array([1 - eps, -1 + eps]), atoms, wB)
    print(f"edge margin {eps}: A(1-eps),A(-1+eps)={vA}  B(1-eps),B(-1+eps)={vB}")

np.savez("/private/tmp/claude-501/-Users-kunal-Documents-Spatial-KK-Project/66ff7cc8-80e8-4b98-8d9f-eac225d1f9cd/scratchpad/analytic_results.npz",
         atoms=atoms, wA=wA, wB=wB, battA=battA, battB=battB,
         xA_min=xA_min, valA_min=valA_min, xB_min=xB_min, valB_min=valB_min)
