"""Shared evaluators for the solo NS-answer session (2026-08-19).

Conventions (P0-aware, stated in CHECKPOINT.md):
  L = 1. Cayley coordinate xi(t) = tan(pi*t/2). Drop measure P = sum_j w_j * delta_{t_j},
  all w_j > 0, atoms strictly inside (-1, 1). Total mass W = sum w_j (this is D).

  Gap function (unnormalized):  Delta(eta) = sum_j w_j ln|eta - xi_j| - W * 0.5*ln(1+eta^2)
  Master identity (C1 / L1):    pi * eps''(x) = -Delta(xi(x))
  PASSIVE  <=>  Delta(eta) <= 0 for all real eta
           <=>  all real critical values of Delta are <= 0   (node collapse, L3/C26,
                because Delta -> 0 at +-inf and -> -inf at the atoms).

Two independent evaluators:
  (1) exact_min_eps2: grid-free node test (polynomial roots of Q, closed-form Delta).
  (2) production_min_eps2: tmm_helper.ht_derivative on an erf-smoothed staircase,
      min over interior nodes (two structural endpoint zeros excluded).
All constructions used this session have M1 = 0 exactly and atoms away from the seam,
so the C39 seam-migration blindness is not in play; argmin is always reported anyway.
"""
import sys, os
import numpy as np
from numpy.polynomial import polynomial as npoly
from scipy.special import erf

_PROJECT_ROOT = "/Users/kunal/Documents/Spatial KK Project"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import tmm_helper as tmm_h  # read-only

L = 1.0

def xi_of_t(t):
    return np.tan(np.pi * t / 2.0)

def t_of_xi(xi):
    return 2.0 / np.pi * np.arctan(xi)

# ---------------------------------------------------------------- atomic measures
class Atomic:
    """Atomic drop measure: atoms t_j in (-1,1), weights w_j > 0."""
    def __init__(self, t, w):
        self.t = np.asarray(t, float)
        self.w = np.asarray(w, float)
        assert np.all(np.abs(self.t) < 1.0), "atoms must be strictly inside the window"
        self.xi = xi_of_t(self.t)
        self.W = self.w.sum()

    def M1(self):
        return np.sum(self.w * self.xi)

    def Delta(self, eta):
        eta = np.atleast_1d(np.asarray(eta, float))
        out = -self.W * 0.5 * np.log1p(eta**2)
        for xj, wj in zip(self.xi, self.w):
            out = out + wj * np.log(np.abs(eta - xj))
        return out

    def mix(self, other, s_self=1.0, s_other=1.0):
        return Atomic(np.concatenate([self.t, other.t]),
                      np.concatenate([s_self * self.w, s_other * other.w]))

def balanced_pair(a, b, mass=1.0):
    """Balanced pair: atoms at t=-a and t=+b (a,b > 0), weights chosen so M1 = 0.
    Cor 4.A: passive iff a + b <= L, active point eta* = tan(A-B), A=pi*b/2, B=pi*a/2."""
    A, B = np.pi * b / 2.0, np.pi * a / 2.0
    beta, gam = np.tan(A), np.tan(B)
    wp = mass * gam / (beta + gam)   # weight at +b (xi=beta)
    wm = mass * beta / (beta + gam)  # weight at -a (xi=-gam)
    return Atomic([b, -a], [wp, wm])

def pair_eta_star(a, b):
    A, B = np.pi * b / 2.0, np.pi * a / 2.0
    beta, gam = np.tan(A), np.tan(B)
    return (beta - gam) / (1.0 + beta * gam)

def pair_gap_star(a, b, mass=1.0):
    """Closed-form Delta(eta*) for the balanced pair (Cor 4.A eq. (*))."""
    A, B = np.pi * b / 2.0, np.pi * a / 2.0
    beta, gam = np.tan(A), np.tan(B)
    return mass * (gam * np.log(np.sin(B) / np.cos(A)) +
                   beta * np.log(np.sin(A) / np.cos(B))) / (beta + gam)

# ---------------------------------------------------------------- exact node test (L3)
def critical_etas(m: Atomic):
    """Real roots of Q(eta): the <= n candidate critical points of Delta (L3/C26).
    Q = (1+eta^2) * sum_j w_j prod_{k!=j}(eta-xi_k)  -  W*eta*prod_k(eta-xi_k)."""
    xi, w, W = m.xi, m.w, m.W
    n = len(xi)
    prod_all = npoly.polyfromroots(xi)          # prod (eta - xi_k), ascending coeffs
    Q = npoly.polymul([1.0, 0.0, 1.0], np.zeros(1))
    acc = np.zeros(1)
    for j in range(n):
        pj = npoly.polyfromroots(np.delete(xi, j))
        acc = npoly.polyadd(acc, w[j] * pj)
    Q = npoly.polysub(npoly.polymul([1.0, 0.0, 1.0], acc),
                      npoly.polymul([0.0, W], prod_all))
    # trim tiny leading coeffs (relative tolerance; K-lesson from h9 block F)
    c = Q
    scale = np.max(np.abs(c)) if np.max(np.abs(c)) > 0 else 1.0
    nz = np.nonzero(np.abs(c) > 1e-12 * scale)[0]
    c = c[: nz[-1] + 1] if len(nz) else c[:1]
    r = npoly.polyroots(c)
    real = r[np.abs(r.imag) < 1e-9 * (1.0 + np.abs(r.real))].real
    # exclude roots that coincide with atoms (Delta -> -inf there anyway)
    keep = []
    for x in real:
        if np.min(np.abs(x - xi)) > 1e-12 * (1.0 + np.abs(x)):
            keep.append(x)
    return np.asarray(keep)

def exact_sup_gap(m: Atomic):
    """sup over finite eta of Delta at critical points; tail limit is 0 exactly."""
    etas = critical_etas(m)
    if len(etas) == 0:
        return 0.0, np.nan
    vals = m.Delta(etas)
    k = int(np.argmax(vals))
    return float(vals[k]), float(etas[k])

def exact_min_eps2(m: Atomic):
    """min over x in (-L, L) of eps''(x) = -(1/pi) * sup_finite Delta  (<= 0 part only).
    Returns (min_eps2, x_argmin). If sup < 0 the profile is strictly passive in the
    interior; min interior eps'' is then positive but we return the *gap* diagnostic
    -(1/pi)*sup which is the quantity whose sign decides."""
    sup, eta = exact_sup_gap(m)
    x = t_of_xi(eta) if np.isfinite(eta) else np.nan
    return -sup / np.pi, x   # sign convention: negative value => gain

# ---------------------------------------------------------------- production evaluator
def staircase_eps1(x, m: Atomic, s=2e-3, base=0.0):
    """eps'(x) = base + sum_j w_j * 0.5*(1 - erf((x - t_j)/(sqrt(2) s)))  — a smoothed
    descending staircase whose drop density is sum_j w_j * (Gaussian at t_j, width s)."""
    e = np.full_like(x, base, dtype=float)
    for tj, wj in zip(m.t, m.w):
        e += wj * 0.5 * (1.0 - erf((x - tj) / (np.sqrt(2.0) * s)))
    return e

def production_min_eps2(m: Atomic, N=2**15, s=2e-3, grid="linspace"):
    """Push the smoothed staircase through PRODUCTION tmm_helper.ht_derivative.
    Verdict: min over interior nodes (structural endpoint zeros excluded)."""
    if grid == "linspace":
        x = np.linspace(-L, L, N)
    else:  # endpoint-excluded periodic grid
        x = -L + 2.0 * L * np.arange(N) / N
    e1 = staircase_eps1(x, m, s=s)
    e2 = tmm_h.ht_derivative(x, e1)
    interior = slice(1, -1)
    k = int(np.argmin(e2[interior])) + 1
    return float(e2[k]), float(x[k])

# ---------------------------------------------------------------- linear functionals
def apply_battery(phis, m: Atomic):
    """ell_i(g) = sum_j w_j * phi_i(t_j) — arbitrary linear functionals on atomic g."""
    return np.array([np.sum(m.w * phi(m.t)) for phi in phis])
