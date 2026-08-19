"""Red-team cycle 3 / C25 hardening: shared machinery.

Two INDEPENDENT evaluators of the DiHiTI partner eps'' of a drop measure P
(P = -d eps'/dt, a positive measure on [-L,L]):

  (A) exact_kernel(x, atoms, L)  -- closed form
        eps''(x) = (1/pi) sum_i w_i ln[ cos(pi t_i/2L) / |sin(pi (x - t_i)/2L)| ]
      (this is the master identity C1 specialised to atoms; it is what the
       replicator's route 1 re-derived from scratch)

  (B) dihiti_of_atoms(...) -- build eps'(x) as an actual staircase of smoothed
      steps and push it through the PRODUCTION tmm_helper.ht_derivative.

Grid convention note (PROTOCOL common-mode caution): the periodic grid on
[-L, L) must NOT duplicate the endpoint.  We use
      x_j = -L + 2L*j/N,  j = 0..N-1      (endpoint EXCLUDED)
so scipy.signal.hilbert sees period exactly 2L.  N is deliberately taken both
as a power of two AND as a non-power-of-two in the convergence checks.
"""
import sys, os
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
import numpy as np
import tmm_helper as tmm_h


def periodic_grid(N, L=1.0):
    """Endpoint-EXCLUDED periodic grid, spacing 2L/N."""
    return -L + 2.0 * L * np.arange(N) / N


def exact_kernel(x, atoms, L=1.0):
    """atoms = list of (t_i, w_i).  Returns eps''(x) from the closed form."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for t, w in atoms:
        num = np.cos(np.pi * t / (2 * L))
        den = np.abs(np.sin(np.pi * (x - t) / (2 * L)))
        with np.errstate(divide='ignore'):
            out += w * (np.log(num) - np.log(den))
    return out / np.pi


def M1_atoms(atoms, L=1.0):
    return sum(w * np.tan(np.pi * t / (2 * L)) for t, w in atoms)


def smoothstep(x, t0, w):
    """C^inf-ish monotone step from 1 down to 0 centred at t0, width w (raised cosine)."""
    y = np.empty_like(x)
    lo = x <= t0 - w / 2
    hi = x >= t0 + w / 2
    mid = ~(lo | hi)
    y[lo] = 1.0
    y[hi] = 0.0
    s = (x[mid] - (t0 - w / 2)) / w
    y[mid] = 0.5 * (1 + np.cos(np.pi * s))
    return y


def eps_re_from_atoms(x, atoms, w, L=1.0):
    """eps'(x) = sum_i w_i * smoothstep(x; t_i, width w).  Drop density g >= 0 iff w_i >= 0."""
    e = np.zeros_like(x)
    for t, wt in atoms:
        e += wt * smoothstep(x, t, w)
    return e


def dihiti_of_atoms(atoms, N=1 << 16, w=None, L=1.0):
    """Push the smoothed staircase through PRODUCTION ht_derivative."""
    x = periodic_grid(N, L)
    if w is None:
        w = 40.0 * (2 * L / N)          # ~40 grid points across each step
    e_re = eps_re_from_atoms(x, atoms, w, L)
    e_im = tmm_h.ht_derivative(x, e_re)
    return x, e_im


def interior_min(x, y, L=1.0, frac=0.999, exclude=None, ex_rad=0.0):
    """min of y over |x| <= frac*L, optionally excluding radius ex_rad around points."""
    m = np.abs(x) <= frac * L
    if exclude is not None:
        for t in exclude:
            m &= np.abs(x - t) > ex_rad
    return y[m].min(), x[m][np.argmin(y[m])]
