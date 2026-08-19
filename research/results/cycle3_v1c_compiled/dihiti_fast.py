"""ctypes wrapper around libdihiti.so -- compiled O(N) legs of the DiHiTI
passivity pipeline. FFT stays in scipy (compiled already, not our target).

Public API:
    grad_fast(u, dx)                 -> np.ndarray, matches np.gradient(u,x) on uniform grid
    post_full(v, dx)                 -> np.ndarray, full de-tilted e_im (for correctness checks)
    post_min(v, dx, lo, hi)          -> float, interior min, NO full array materialized (fast path)
    verdict_compiled(x, e_re, frac)  -> (min_eim, passive_bool)   -- the actual candidate test
    verdict_baseline(x, e_re, frac)  -> (min_eim, passive_bool)   -- numpy/scipy baseline, same math
"""
import ctypes
import os
import numpy as np
from scipy.signal import hilbert

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_HERE, "libdihiti.so"))

_dbl_p = ctypes.POINTER(ctypes.c_double)

_lib.grad_fast.argtypes = [_dbl_p, ctypes.c_double, ctypes.c_long, _dbl_p]
_lib.grad_fast.restype = None

_lib.post_full.argtypes = [_dbl_p, ctypes.c_double, ctypes.c_long, _dbl_p]
_lib.post_full.restype = None

_lib.post_min.argtypes = [_dbl_p, ctypes.c_double, ctypes.c_long, ctypes.c_long,
                           ctypes.c_long, _dbl_p]
_lib.post_min.restype = ctypes.c_double


def _ptr(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(_dbl_p)


def grad_fast(u, dx):
    u, up = _ptr(u)
    N = len(u)
    out = np.empty(N, dtype=np.float64)
    _lib.grad_fast(up, ctypes.c_double(dx), ctypes.c_long(N), out.ctypes.data_as(_dbl_p))
    return out


def post_full(v, dx):
    v, vp = _ptr(v)
    N = len(v)
    out = np.empty(N, dtype=np.float64)
    _lib.post_full(vp, ctypes.c_double(dx), ctypes.c_long(N), out.ctypes.data_as(_dbl_p))
    return out


def post_min(v, dx, lo, hi):
    v, vp = _ptr(v)
    N = len(v)
    return _lib.post_min(vp, ctypes.c_double(dx), ctypes.c_long(N),
                          ctypes.c_long(lo), ctypes.c_long(hi), None)


def _interior_bounds(N, frac):
    """Interior index window matching the project convention |x| <= frac*L
    on a symmetric uniform grid x = linspace(-L, L, N)."""
    lo = int(np.ceil((1 - frac) / 2.0 * (N - 1)))
    hi = N - 1 - lo
    return lo, hi


def verdict_compiled(x, e_re, frac=0.95):
    """Candidate: fused C grad + FFT (scipy) + fused C post/min."""
    dx = x[1] - x[0]
    u = grad_fast(e_re, dx)
    v = np.imag(hilbert(u))
    lo, hi = _interior_bounds(len(x), frac)
    m = post_min(v, dx, lo, hi)
    scale = None  # caller applies the project's relative tolerance if needed
    return m, (m >= 0.0)


def verdict_baseline(x, e_re, frac=0.95):
    """Baseline: numpy gradient -> scipy hilbert -> numpy cumtrapz -> de-tilt -> mask -> min.
    Exactly the pipeline named in the task brief / tmm_helper.ht_derivative."""
    from scipy.integrate import cumulative_trapezoid
    u = np.gradient(e_re, x)
    v = np.imag(hilbert(u))
    e = cumulative_trapezoid(v, x, initial=0.0)
    e = e - np.linspace(e[0], e[-1], len(x))
    L = 0.5 * (x[-1] - x[0])
    mask = np.abs(x) <= frac * L
    m = e[mask].min()
    return m, (m >= 0.0)
