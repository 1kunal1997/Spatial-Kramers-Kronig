"""Shared machinery for the P0 seam-adjudication study.

Deliberately built to VARY THE CONSTRUCTION (PROTOCOL common-mode caution):
  * two grid conventions: linspace (endpoint DUPLICATED, the project default and
    the C22 bug grid) and periodic (endpoint EXCLUDED, spacing 2L/N)
  * an exact closed-form evaluator for atom drop measures (no FFT at all)
  * the production DiHiTI pipeline
  * an independent spectral DiHiTI that never calls scipy.signal.hilbert
"""
import sys, os
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'theory'))
sys.path.insert(0, os.path.join(_ROOT, 'theory', 'passivity_2026Aug'))
import numpy as np
import tmm_helper as tmm_h

# ----------------------------------------------------------------- grids
def grid_linspace(N, L=1.0):
    """np.linspace(-L, L, N): endpoint DUPLICATED under 2L-periodicity."""
    return np.linspace(-L, L, N)

def grid_periodic(N, L=1.0):
    """x_j = -L + 2L j/N: endpoint EXCLUDED, true period 2L."""
    return -L + 2.0 * L * np.arange(N) / N

# ------------------------------------------------- exact kernel (atoms)
def exact_kernel(x, atoms, L=1.0):
    """eps''(x) = (1/pi) sum_i w_i ln[cos(pi t_i/2L) / |sin(pi(x-t_i)/2L)|]."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for t, w in atoms:
        with np.errstate(divide='ignore'):
            out += w * (np.log(np.cos(np.pi * t / (2 * L)))
                        - np.log(np.abs(np.sin(np.pi * (x - t) / (2 * L)))))
    return out / np.pi

def M1_atoms(atoms, L=1.0):
    return float(sum(w * np.tan(np.pi * t / (2 * L)) for t, w in atoms))

def M2_atoms(atoms, L=1.0):
    return float(sum(w / np.cos(np.pi * t / (2 * L))**2 for t, w in atoms))

def M3_atoms(atoms, L=1.0):
    return float(sum(2 * w * np.tan(np.pi*t/(2*L)) / np.cos(np.pi*t/(2*L))**2 for t, w in atoms))

# ------------------------------------------------------- DiHiTI engines
def dihiti_production(x, e_re):
    """tmm_helper.ht_derivative verbatim (np.gradient -> scipy hilbert -> cumtrapz)."""
    return tmm_h.ht_derivative(x, e_re)

def dihiti_spectral(x, e_re, L=None):
    """INDEPENDENT route: periodic conjugate function by rfft, no scipy.signal.hilbert,
    no np.gradient, no cumulative_trapezoid.  Requires an endpoint-EXCLUDED grid.

    On [-L,L) with period 2L, eps' = sum_m c_m e^{i pi m x/L}; the DiHiTI partner is
    eps'' = sum_{m!=0} (-i sgn m) c_m e^{i pi m x/L}, then shifted so eps''(-L)=0.
    """
    N = len(e_re)
    C = np.fft.rfft(e_re)
    m = np.arange(len(C))
    Cim = -1j * C
    Cim[0] = 0.0
    y = np.fft.irfft(Cim, N)
    return y - y[0]

# ------------------------------------------------- edge / seam quantities
def edge_slopes(x, e_re, L=None, k=4):
    """g(+L) and g(-L) with g = -d eps'/dx, one-sided O(h^k) differences at the
    two ends of the (endpoint-INCLUSIVE view of the) sampled profile."""
    h = x[1] - x[0]
    # one-sided 4th-order first derivative stencils
    fwd = np.array([-25/12, 4, -3, 4/3, -1/4]) / h
    bwd = -fwd[::-1]
    dm = float(fwd @ e_re[:5])          # eps'_x at x[0]  (~ -L)
    dp = float(bwd @ e_re[-5:])         # eps'_x at x[-1] (~ +L)
    return -dp, -dm                     # g(+L), g(-L)

def M1_pv(x, e_re, L, cut_frac=None):
    """Symmetric principal value INT_{|t|<(1-eta)L} g(t) tan(pi t/2L) dt, g = -eps'_x.
    Returned for a ladder of cutoffs so convergence can be inspected."""
    g = -np.gradient(e_re, x)
    tn = np.tan(np.pi * x / (2 * L))
    out = []
    fr = cut_frac if cut_frac is not None else [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    for f in np.atleast_1d(fr):
        m = np.abs(x) <= f * L
        out.append(float(np.trapezoid(g[m] * tn[m], x[m])))
    return np.array(out), np.atleast_1d(fr)

def M2_pv(x, e_re, L, f=0.99):
    g = -np.gradient(e_re, x)
    sec2 = 1.0 / np.cos(np.pi * x / (2 * L))**2
    m = np.abs(x) <= f * L
    return float(np.trapezoid(g[m] * sec2[m], x[m]))

# ------------------------------------------------------- min conventions
def min_whole(x, y, L=1.0):
    i = int(np.argmin(y)); return float(y[i]), float(x[i])

def min_mask(x, y, L=1.0, frac=0.95):
    m = np.abs(x) <= frac * L
    i = int(np.argmin(y[m])); return float(y[m][i]), float(x[m][i])

def min_dropends(x, y, L=1.0, n=1):
    """Exclude ONLY the n structurally-zero nodes at each end."""
    yy = y[n:len(y)-n] if n else y
    xx = x[n:len(x)-n] if n else x
    i = int(np.argmin(yy)); return float(yy[i]), float(xx[i])
