"""
HARNESS CONTROL — required by the dispatch brief before any C49 number is trusted.

Independent known result: a single balanced atomic pair at t=-a, t=+b (both
weights positive, drop measure), with weights chosen so M1 = sum w_j tan(pi t_j/2) = 0,
is PASSIVE iff a+b <= 1 (window half-width L=1).

We test both the analytic closed-form log-potential route and the
tmm_helper.ht_derivative grid route against this known threshold, sweeping
a=b symmetric pairs across a+b = 0.6 .. 1.4, and asymmetric pairs too.
"""
import sys
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from scipy.special import erf
import tmm_helper as tmm_h

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

def analytic_min(atoms, w, n=400001, atom_excl=1e-4, margin=1e-6):
    xs = np.linspace(-L + margin, L - margin, n)
    mask = np.ones_like(xs, dtype=bool)
    for tj in atoms:
        mask &= (np.abs(xs - tj) > atom_excl)
    xs_m = xs[mask]
    vals = eps2_analytic_v(xs_m, atoms, w, L)
    idx = np.argmin(vals)
    return vals[idx], xs_m[idx]

def staircase(x, atoms, w, s):
    out = np.zeros_like(x)
    for tj, wj in zip(atoms, w):
        out = out - wj * 0.5 * (1 + erf((x - tj) / (s * np.sqrt(2))))
    return out

def ht_min(atoms, w, N=8001, s=0.005, edge_margin=0.02, atom_excl=0.01):
    x = -L + 2 * L * (np.arange(N) + 0.5) / N
    e = staircase(x, atoms, w, s)
    e2 = tmm_h.ht_derivative(x, e)
    mask = (x > -L + edge_margin) & (x < L - edge_margin)
    for tj in atoms:
        mask &= (np.abs(x - tj) > atom_excl)
    xm, em = x[mask], e2[mask]
    idx = np.argmin(em)
    return em[idx], xm[idx]

print("=== Symmetric pair a=b, weights=1 each (M1=0 automatically by symmetry) ===")
print(f"{'a+b':>6} | {'analytic min':>13} verdict | {'ht_deriv min':>13} verdict")
for a in [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
    atoms = np.array([-a, a])
    w = np.array([1.0, 1.0])
    vA, xA = analytic_min(atoms, w)
    vH, xH = ht_min(atoms, w)
    vdA = "GAIN" if vA < -1e-6 else "PASSIVE"
    vdH = "GAIN" if vH < -1e-5 else "PASSIVE"
    print(f"{2*a:6.3f} | {vA:13.6f} {vdA:8s}| {vH:13.6f} {vdH:8s}")

print()
print("=== Asymmetric pair, M1=0 enforced: w_b = w_a*tan(pi a/2)/tan(pi b/2) ===")
print(f"{'a':>6} {'b':>6} {'a+b':>6} | {'analytic min':>13} verdict | {'ht_deriv min':>13} verdict")
for a, b in [(0.2, 0.5), (0.2, 0.7), (0.1, 0.6), (0.1, 0.85), (0.3, 0.6), (0.3, 0.75), (0.05, 0.99)]:
    wa = 1.0
    wb = wa * np.tan(np.pi * a / 2) / np.tan(np.pi * b / 2)
    atoms = np.array([-a, b])
    w = np.array([wa, wb])
    M1 = -wa * np.tan(np.pi * a / 2) + wb * np.tan(np.pi * b / 2)
    vA, xA = analytic_min(atoms, w)
    vH, xH = ht_min(atoms, w)
    vdA = "GAIN" if vA < -1e-6 else "PASSIVE"
    vdH = "GAIN" if vH < -1e-5 else "PASSIVE"
    print(f"{a:6.3f} {b:6.3f} {a+b:6.3f} | {vA:13.6f} {vdA:8s}| {vH:13.6f} {vdH:8s}  (M1 check={M1:.2e})")
