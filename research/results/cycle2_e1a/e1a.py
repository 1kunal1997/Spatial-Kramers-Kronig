"""E1(a): move the cascade onto the DECIMATED grid.

Candidates
  B0 dihiti_full      : naive gradient->hilbert->cumtrapz, min over interior   [incumbent]
  B1 onehilb_full     : ramp-periodized single Hilbert, min over interior      [BEST baseline, 1.70x]
  S0 split_cascadeN   : cycle-1 split test  (cascade at N, decimated interior)
  S1 split_cascade_n  : E1(a)               (cascade at n, decimated interior)
  S2 decim_only       : decimation, no screen at all

Screen semantics (NEW this cycle -- cycle 1 computed the cascade but NEVER USED IT):
  if |g(-L)-g(+L)|/D > tau  -> DEFER to the full baseline (B1). Otherwise answer from decimation.
  A DEFER is not a wrong answer; it costs baseline time.
"""
import numpy as np, timeit, json, os, sys
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

OUT = os.path.dirname(os.path.abspath(__file__))
nb = 1.7; L = 1.0; A = nb**2 - 1
FRAC = 0.95           # interior window
NDEC = 512            # decimated grid size
TAU  = 1e-3           # edge-slope screen threshold (|g(-L)-g(+L)|/D)

# ---------------------------------------------------------------- transforms
def dihiti(u, x):
    du = np.gradient(u, x)
    v = np.imag(hilbert(du))
    e = cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))

def onehilb(u, x):
    L_ = 0.5*(x[-1]-x[0]); D = u[0]-u[-1]
    p = u + (D/(2*L_))*x
    v = np.imag(hilbert(p))
    return v - v[0]

# ---------------------------------------------------------------- cascade
def cascade(u, x):
    """necessary seam conditions. O(len(x)) transcendental evaluations."""
    L_ = 0.5*(x[-1]-x[0]); g = -np.gradient(u, x)
    slope = g[0]-g[-1]
    m = np.abs(x) < 0.98*L_
    M1 = np.trapezoid((g*np.tan(np.pi*x/(2*L_)))[m], x[m])
    M2 = np.trapezoid((g/np.cos(np.pi*x/(2*L_))**2)[m], x[m])
    sig = np.trapezoid(g*np.log(2*np.cos(np.pi*x/(2*L_))+1e-300), x)
    return slope, M1, M2, sig

def slope_only(u, x):
    """cheapest form of the screen: edge slopes only, no transcendentals, O(1) reads."""
    h0 = x[1]-x[0]; hN = x[-1]-x[-2]
    g0 = -(u[1]-u[0])/h0
    gN = -(u[-1]-u[-2])/hN
    return g0-gN

# ---------------------------------------------------------------- verdicts
def interior_min(v, x, frac=FRAC):
    Lx = 0.5*(x[-1]-x[0]); m = np.abs(x) <= frac*Lx
    return v[m].min()

def B0(u, x):  return interior_min(dihiti(u, x), x)
def B1(u, x):  return interior_min(onehilb(u, x), x)

def decim(u, x, n=NDEC):
    s = max(1, len(x)//n); return u[::s], x[::s]

def S2(u, x, n=NDEC):
    ud, xd = decim(u, x, n); return interior_min(onehilb(ud, xd), xd)

def S0(u, x, n=NDEC, tau=TAU):
    """cascade evaluated on the FULL grid (cycle-1 cost), screen wired in."""
    D = u[0]-u[-1]
    sl, M1, M2, sg = cascade(u, x)
    if abs(sl)/abs(D) > tau:
        return B1(u, x), True          # DEFER
    return S2(u, x, n), False

def S1(u, x, n=NDEC, tau=TAU):
    """E1(a): cascade evaluated on the DECIMATED grid."""
    D = u[0]-u[-1]
    ud, xd = decim(u, x, n)
    sl, M1, M2, sg = cascade(ud, xd)
    if abs(sl)/abs(D) > tau:
        return B1(u, x), True          # DEFER
    return interior_min(onehilb(ud, xd), xd), False

# ---------------------------------------------------------------- profiles
def build_cases(x, seed=7):
    A_ = A; cases = {}
    for w in [0.3, 0.15, 0.05, 0.02, 0.005]:
        cases[f"logistic w={w}"] = 1 + A_/(1+np.exp(x/w))
    for al in [0.0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.5]:
        cases[f"bumps a={al}"] = (1 + A_/(1+np.exp(x/0.05))
            + al*A_*(np.exp(-((x+0.45)/0.22)**2) - np.exp(-((x-0.45)/0.22)**2)))
    cases["triangle"] = 1 + 2.0*(np.sqrt(L**2+0.15**2) - np.sqrt(x**2+0.15**2))
    rng = np.random.default_rng(seed)
    for i in range(6):
        g = sum(rng.uniform(.1,1)*np.exp(-((x-rng.uniform(-.9,.9))/rng.uniform(.02,.3))**2)
                for _ in range(rng.integers(3,7)))
        G = cumulative_trapezoid(g, x, initial=0); u = 1+G[-1]-G
        cases[f"rand{i}"] = 1 + (u-1)*A_/(u[0]-u[-1])
    for m_ in [255, 256, 257, 512, 1024]:
        cases[f"cos m={m_}"] = 1 + A_/(1+np.exp(x/0.05)) + 0.02*A_*np.cos(m_*np.pi*x/L)
    # extra adversaries: dressings that are INVISIBLE on the decimated grid
    # cos(m pi x/L) with m a multiple of the decimation stride aliases to a constant
    for m_ in [2048, 4096]:
        cases[f"cos m={m_}"] = 1 + A_/(1+np.exp(x/0.05)) + 0.02*A_*np.cos(m_*np.pi*x/L)
    # near-threshold bumps
    for al in [0.6640, 0.6662, 0.6680]:
        cases[f"bumps* a={al}"] = (1 + A_/(1+np.exp(x/0.05))
            + al*A_*(np.exp(-((x+0.45)/0.22)**2) - np.exp(-((x-0.45)/0.22)**2)))
    return cases

# ---------------------------------------------------------------- accuracy
def accuracy(N=2**16):
    x = np.linspace(-L, L, N); cases = build_cases(x)
    rows = []
    print(f"{'case':16s} {'truth/D':>11s} {'S1/D':>11s} {'S1':>6s} {'S0':>6s} {'S2':>6s} {'|sl_N|/D':>9s} {'|sl_n|/D':>9s}")
    n_s1 = n_s0 = n_s2 = 0; n_def1 = n_def0 = 0
    for k, u in cases.items():
        D = u[0]-u[-1]
        t = B0(u, x)
        v1, d1 = S1(u, x); v0, d0 = S0(u, x); v2 = S2(u, x)
        ok1 = (t >= 0) == (v1 >= 0); ok0 = (t >= 0) == (v0 >= 0); ok2 = (t >= 0) == (v2 >= 0)
        n_s1 += ok1; n_s0 += ok0; n_s2 += ok2; n_def1 += d1; n_def0 += d0
        slN = abs(cascade(u, x)[0])/abs(D)
        ud, xd = decim(u, x); sln = abs(cascade(ud, xd)[0])/abs(D)
        tag = lambda ok, d: ('DEFER' if d else ('ok' if ok else 'MISS'))
        print(f"{k:16s} {t/D:+11.3e} {v1/D:+11.3e} {tag(ok1,d1):>6s} {tag(ok0,d0):>6s} "
              f"{'ok' if ok2 else 'MISS':>6s} {slN:9.2e} {sln:9.2e}")
        rows.append(dict(case=k, truth=t/D, s1=v1/D, ok1=bool(ok1), def1=bool(d1),
                         ok0=bool(ok0), def0=bool(d0), ok2=bool(ok2), slopeN=slN, slopen=sln))
    n = len(cases)
    print(f"\nN={N}  S1 (E1a) {n_s1}/{n} correct, {n_def1} defers | "
          f"S0 (cycle1) {n_s0}/{n}, {n_def0} defers | S2 (no screen) {n_s2}/{n}")
    return rows

# ---------------------------------------------------------------- timing
def timing(Ns=(2**12, 2**14, 2**16, 2**18), reps=7, number=10):
    print(f"\n{'N':>8s} {'B0 dihiti':>11s} {'B1 onehilb':>11s} {'S0':>11s} {'S1':>11s} "
          f"{'S1/B0':>7s} {'S1/B1':>7s}")
    out = []
    for N in Ns:
        x = np.linspace(-L, L, N); u = 1 + A/(1+np.exp(x/0.05))
        f = lambda fn: min(timeit.repeat(lambda: fn(u, x), number=number, repeat=reps))/number
        t0 = f(B0); t1 = f(B1); ts0 = f(S0); ts1 = f(S1)
        print(f"{N:8d} {t0*1e6:10.1f}u {t1*1e6:10.1f}u {ts0*1e6:10.1f}u {ts1*1e6:10.1f}u "
              f"{t0/ts1:7.2f} {t1/ts1:7.2f}")
        out.append(dict(N=N, B0=t0, B1=t1, S0=ts0, S1=ts1))
    return out

if __name__ == "__main__":
    rows = accuracy()
    tm = timing()
    json.dump(dict(acc=rows, timing=tm, params=dict(NDEC=NDEC, TAU=TAU, FRAC=FRAC)),
              open(os.path.join(OUT, "e1a_results.json"), "w"), indent=1)
