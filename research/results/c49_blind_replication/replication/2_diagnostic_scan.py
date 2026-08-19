import json
import numpy as np

IN_PATH = "/Users/kunal/Documents/Spatial KK Project/research/results/c49_blind_replication/witness_input.json"
with open(IN_PATH) as f:
    data = json.load(f)
atoms = np.array(data["atoms_t"], dtype=float)
wA = np.array(data["weights_A"], dtype=float)
wB = np.array(data["weights_B"], dtype=float)
L = 1.0

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

# full fine scan for B, exclude only extremely close to atoms
n = 2_000_001
xs = np.linspace(-1 + 1e-7, 1 - 1e-7, n)
mask = np.ones_like(xs, dtype=bool)
for tj in atoms:
    mask &= (np.abs(xs - tj) > 1e-5)
xs_m = xs[mask]

valsB = eps2_analytic_v(xs_m, atoms, wB, L)
idx = np.argmin(valsB)
print("B global interior min (fine, margin 1e-7):", valsB[idx], "at x=", xs_m[idx])

# print value profile of B near its most negative candidate regions in coarser terms
# scan sign changes
neg_mask = valsB < 0
print("fraction of interior points where B < 0:", neg_mask.mean())
if neg_mask.any():
    neg_xs = xs_m[neg_mask]
    print("neg x range:", neg_xs.min(), neg_xs.max())
    print("min value in neg region:", valsB[neg_mask].min())

valsA = eps2_analytic_v(xs_m, atoms, wA, L)
idxA = np.argmin(valsA)
print("A global interior min (fine, margin 1e-7):", valsA[idxA], "at x=", xs_m[idxA])
neg_maskA = valsA < 0
print("fraction of interior points where A < 0:", neg_maskA.mean())
negA_xs = xs_m[neg_maskA]
print("A neg x range:", negA_xs.min(), negA_xs.max())

# Print B's value at a handful of x positions across domain to see overall shape
sample_x = np.linspace(-0.999, 0.999, 41)
valsB_samp = eps2_analytic_v(sample_x, atoms, wB, L)
for xv, yv in zip(sample_x, valsB_samp):
    print(f"x={xv:+.3f}  epsB''={yv:+.6f}")
