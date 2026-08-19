"""E1 — Theorem 1 (density of essential test points), numerical witness.

Claim: for every target eta0 and every window U = (eta0-r, eta0+r), the inflated
boundary pair (a,b) with eta* = eta0 and a+b = L + delta has Delta > 0 ONLY inside U
once delta is small enough — and the positivity set shrinks to {eta0} as delta -> 0.
Hence any sound-and-complete test set must meet every open interval: it must be dense.

Control: at delta = 0 the positivity set is empty and max Delta = 0 at eta0 exactly
(closed form, Cor 4.A). If the harness were wrong the delta->0 limit would not pin
to (eta0, 0).
"""
import numpy as np
from common import balanced_pair, pair_eta_star, exact_sup_gap

from scipy.optimize import brentq

def pair_for_eta(eta0, delta):
    """Boundary pair with active point ~eta0, inflated by delta on whichever side
    keeps both atoms strictly inside the window."""
    b = 0.5 + np.arctan(eta0) / np.pi
    a = 0.5 - np.arctan(eta0) / np.pi
    if b + delta < 1.0 - 1e-9:
        return balanced_pair(a, b + delta), a, b
    return balanced_pair(a + delta, b), a, b

for eta0 in (1.7, -0.4, 8.0):
    print(f"---- eta0 = {eta0}  (x0 = {2/np.pi*np.arctan(eta0):+.4f} L)")
    print("  control delta=0: ", end="")
    m, a, b = pair_for_eta(eta0, 0.0)
    sup, at = exact_sup_gap(m)
    print(f"sup Delta = {sup:+.2e} at eta = {at:.6f}  (expect 0 at {eta0})")
    for delta in (0.1, 0.03, 0.01, 0.003, 0.001):
        m, a, b = pair_for_eta(eta0, delta)
        sup, at = exact_sup_gap(m)
        # exact zero crossings of Delta around the max (bisection, not grid)
        f = lambda e: m.Delta(np.array([e]))[0]
        step = 1e-4
        eL = at
        while f(eL) > 0: eL -= step; step *= 1.5
        lo = brentq(f, eL, at, xtol=1e-12)
        step = 1e-4
        eR = at
        while f(eR) > 0: eR += step; step *= 1.5
        hi = brentq(f, at, eR, xtol=1e-12)
        # far-field check: Delta < 0 strictly outside [lo, hi] on a wide global grid
        wide = np.tan(np.pi * 0.5 * np.linspace(-0.9999995, 0.9999995, 400001))
        far = wide[(wide < lo - 1e-6) | (wide > hi + 1e-6)]
        worst_far = m.Delta(far).max()
        print(f"  delta={delta:6.3f}: sup={sup:+.3e} at {at:+.6f}  "
              f"positivity set = [{lo:+.6f}, {hi:+.6f}] (width {hi-lo:.6f})  "
              f"max Delta outside = {worst_far:+.2e}")
