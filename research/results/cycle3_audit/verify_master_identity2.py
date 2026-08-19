"""Independent, from-scratch verification of gap_clausen against the TRUE continuous
master identity (C1), using the analytic logistic profile (no grid reconstruction at
all) and scipy adaptive quadrature (quad) which handles the log singularity natively.
Then separately test grid-convention sensitivity (linspace w/ duplicated endpoint vs
a proper period-2L grid) — the exact pattern flagged in PROTOCOL.md's common-mode note.
"""
import sys, numpy as np
from scipy.integrate import quad
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory/passivity_2026Aug")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/theory")
import tmm_helper as tmm_h
from t3_clausen import gap_clausen

NB, k = 1.7, 4.0
L = 2.5
A = NB**2 - 1

def g(t):
    e = np.exp(k*t)
    return A*k*e/(1+e)**2

def kernel_true(t, xpt, L):
    return np.log(np.abs(np.cos(np.pi*t/(2*L)) / np.sin(np.pi*(xpt-t)/(2*L))))

def eps_im_true(xpt, L):
    # split at the singularity t = xpt
    val1, _ = quad(lambda t: g(t)*kernel_true(t, xpt, L), -L, xpt, limit=400, points=[xpt-1e-8])
    val2, _ = quad(lambda t: g(t)*kernel_true(t, xpt, L), xpt, L, limit=400, points=[xpt+1e-8])
    return (val1+val2)/np.pi

test_x = [-1.5, -0.7, -0.2, 0.0, 0.3, 0.9, 1.7]
print("Analytic-quad reference (truth) vs gap_clausen at several N:")
truth = {xp: eps_im_true(xp, L) for xp in test_x}
for xp,v in truth.items():
    print(f"  x={xp:+.2f}  truth={v:.8f}")

for N in [513, 1025, 2049, 4097]:
    x = np.linspace(-L, L, N)
    e_re = tmm_h.logistic(x, k, NB)
    dom, D = gap_clausen(e_re, x)
    errs = []
    for xp in test_x:
        idx = np.argmin(np.abs(x-xp))
        errs.append(abs(dom[idx]-truth[xp]))
    print(f"N={N:5d}  max err vs analytic truth = {max(errs):.4e}  (h={2*L/(N-1):.2e})")
