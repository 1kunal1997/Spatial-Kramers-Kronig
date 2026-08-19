"""Independent brute-force check of the master identity (C1) and gap_clausen (C33),
using a fine direct quadrature route, NOT reusing t3_clausen's Cl2 or DiHiTI's FFT."""
import sys, numpy as np
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory/passivity_2026Aug")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/theory")
import tmm_helper as tmm_h
from t3_clausen import gap_clausen

def brute_eps_im(x, e_re, L, n_quad=200000):
    """Direct numerical evaluation of C1's master identity via fine Simpson-ish quad,
    piecewise-linear reconstruction of e_re(t) (NOT piecewise-constant drop),
    completely independent of the Clausen machinery."""
    tq = np.linspace(x[0], x[-1], n_quad, endpoint=False) + 0.5*(x[1]-x[0])*0  # cell centers approx
    tq = np.linspace(x[0]+1e-9, x[-1]-1e-9, n_quad)
    ereq = np.interp(tq, x, e_re)
    g = -np.gradient(ereq, tq)   # drop density
    out = np.zeros_like(x)
    dt = tq[1]-tq[0]
    for i, xi_ in enumerate(x):
        if i == 0 or i == len(x)-1:
            out[i] = 0.0
            continue
        # avoid singularity at t = xi_ by tiny exclusion
        diff = xi_ - tq
        mask = np.abs(diff) > 1e-6
        kernel = np.log(np.abs(np.cos(np.pi*tq[mask]/(2*L)) / np.sin(np.pi*diff[mask]/(2*L))))
        out[i] = np.sum(g[mask]*kernel)*dt/np.pi
    return out

np.random.seed(0)
L = 2.5
for N in [513, 1025]:
    x = np.linspace(-L, L, N)
    e_re = tmm_h.logistic(x, 4.0, 1.7)
    dom, D = gap_clausen(e_re, x)
    brute = brute_eps_im(x, e_re, L, n_quad=60000)
    # compare interior only
    interior = np.abs(x) < 0.9*L
    diff = np.abs(dom - brute)[interior]
    print(f"N={N}  max|clausen - brute|(interior) = {diff.max():.4e}  scale={np.abs(dom[interior]).max():.4e}")
