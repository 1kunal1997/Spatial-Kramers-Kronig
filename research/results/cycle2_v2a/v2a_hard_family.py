"""V2a numerical backbone: the hard family for the Omega(N) query lower bound.

Claim to test: there is a base profile u0 with interior margin m0>0 and, for EVERY grid location
x_j in a set of size Theta(N), a perturbation p_j supported on O(1) GRID POINTS (width c*h, h=2L/N)
with ||p_j||_inf <= C*m0 (C an absolute constant, independent of N and of j) such that
u0 + p_j has interior min < 0.

If so, an adversary/pigeonhole argument gives Omega(N) queries -- and, crucially, the hard family
sits inside a FIXED small ball in both L_inf and total variation, so the bound is not an artifact of
allowing unbounded or non-smooth inputs.
"""
import numpy as np, json, os, sys
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
OUT = os.path.dirname(os.path.abspath(__file__))
L = 2.5; NB = 1.7; A = NB**2 - 1; W = 0.05; FRAC = 0.95

def dihiti(u, x):
    du = np.gradient(u, x); v = np.imag(hilbert(du))
    e = cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))
def imin(v, x):
    return v[np.abs(x) <= FRAC*L].min()

def bump(x, x0, sig):        # C^inf, compactly-supported-ish gaussian
    return np.exp(-((x-x0)/sig)**2)

def a_min(u0, x, x0, sig, m0):
    """smallest |a| (either sign) making the interior min negative; bisection."""
    best = None
    for sgn in (+1.0, -1.0):
        lo, hi = 0.0, 1e-3
        f = lambda a: imin(dihiti(u0 + sgn*a*bump(x, x0, sig), x), x)
        t = 0
        while f(hi) >= 0 and t < 40: hi *= 1.7; t += 1
        if f(hi) >= 0: continue
        for _ in range(50):
            mid = 0.5*(lo+hi)
            if f(mid) >= 0: lo = mid
            else: hi = mid
        if best is None or hi < best: best = hi
    return best

def run():
    res = []
    print(f"{'N':>7s} {'h':>10s} {'c(=sig/h)':>10s} {'m0/D':>10s} "
          f"{'a_min/D':>10s} {'a_min/m0':>9s} {'||p||_1/(D L)':>14s} {'TV(p)/D':>9s} {'x0*/L':>7s} {'#pos':>5s}")
    for N in (2**11, 2**12, 2**13, 2**14, 2**15):
        x = np.linspace(-L, L, N); h = x[1]-x[0]
        u0 = 1 + A/(1+np.exp(x/W)); D = u0[0]-u0[-1]
        m0 = imin(dihiti(u0, x), x)
        for c in (2.0, 4.0, 8.0):
            sig = c*h
            # scan positions across the interior; report the easiest and how many succeed
            xs = np.linspace(-0.97*L, 0.97*L, 25)
            vals = [(a_min(u0, x, x0, sig, m0), x0) for x0 in xs]
            vals = [(a, x0) for a, x0 in vals if a is not None]
            if not vals: print(f"{N:7d} {h:10.3e} {c:10.1f}  -- none flip --"); continue
            a, x0s = min(vals)
            p = a*bump(x, x0s, sig)
            l1 = np.trapezoid(np.abs(p), x); tv = 2*a
            print(f"{N:7d} {h:10.3e} {c:10.1f} {m0/D:10.3e} {a/D:10.3e} {a/m0:9.3f} "
                  f"{l1/(D*L):14.3e} {tv/D:9.3e} {x0s/L:+7.3f} {len(vals):5d}")
            res.append(dict(N=N, h=h, c=c, m0=m0/D, a_min=a/D, ratio=a/m0,
                            l1=l1/(D*L), tv=tv/D, x0=x0s/L, nsucc=len(vals), ntried=len(xs)))
    json.dump(res, open(os.path.join(OUT, "v2a_hard_family.json"), "w"), indent=1)

if __name__ == "__main__": run()
