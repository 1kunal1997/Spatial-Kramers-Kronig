"""E1(a) v2 -- honest, screened, defer-weighted.

Ground truth: dihiti at the FULL problem size N, interior |x|<=0.95L.
Engine on the decimated grid: dihiti (NOT onehilb -- see F2, onehilb's grid bias
at n=512 exceeds the PASS margin).

Variants
  B0  dihiti(N)                       incumbent
  B1  onehilb(N)                      fastest exact-in-the-limit baseline
  Sf  screen at N  + dihiti(n)        cycle-1 split test, screen ACTUALLY WIRED
  Sd  screen at n  + dihiti(n)        E1(a)

Screen: DEFER iff |g(-L)-g(+L)|/|D| > tau.  A DEFER costs a full B1 run.
Reported: false ANSWERS (sign wrong while not deferring), defer rate, and
defer-weighted expected wall clock.
"""
import numpy as np, timeit, json, os
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

OUT = os.path.dirname(os.path.abspath(__file__))
nb = 1.7; L = 1.0; A = nb**2 - 1; FRAC = 0.95

def dihiti(u, x):
    du = np.gradient(u, x); v = np.imag(hilbert(du))
    e = cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))

def onehilb(u, x):
    L_ = 0.5*(x[-1]-x[0]); D = u[0]-u[-1]
    v = np.imag(hilbert(u + (D/(2*L_))*x)); return v - v[0]

def imin(v, x, frac=FRAC):
    Lx = 0.5*(x[-1]-x[0]); return v[np.abs(x) <= frac*Lx].min()

def screen(u, x):
    """|g(-L)-g(+L)| via np.gradient's 2nd-order one-sided end formulas."""
    g = -np.gradient(u, x); return abs(g[0]-g[-1])

def build_cases(x, nrand=200, seed=11):
    A_ = A; c = {}; kind = {}
    for w in [0.3, 0.15, 0.05, 0.02, 0.005]:
        c[f"logistic w={w}"] = 1 + A_/(1+np.exp(x/w)); kind[f"logistic w={w}"] = 'phys'
    for al in [0.0,0.3,0.6,0.6640,0.6662,0.6680,0.8,0.9,1.0,1.5]:
        k = f"bumps a={al}"
        c[k] = (1 + A_/(1+np.exp(x/0.05))
                + al*A_*(np.exp(-((x+0.45)/0.22)**2) - np.exp(-((x-0.45)/0.22)**2)))
        kind[k] = 'phys'
    rng = np.random.default_rng(seed)
    for i in range(nrand):
        g = sum(rng.uniform(.1,1)*np.exp(-((x-rng.uniform(-.9,.9))/rng.uniform(.02,.3))**2)
                for _ in range(rng.integers(3,7)))
        G = cumulative_trapezoid(g, x, initial=0); u = 1+G[-1]-G
        c[f"rand{i}"] = 1 + (u-1)*A_/(u[0]-u[-1]); kind[f"rand{i}"] = 'phys'
    for m_ in [63,64,65,127,128,129,255,256,257,511,512,513,1024,2048,4096]:
        k = f"cos m={m_}"
        c[k] = 1 + A_/(1+np.exp(x/0.05)) + 0.02*A_*np.cos(m_*np.pi*x/L); kind[k] = 'adv'
    return c, kind

def run(N=2**16, ns=(128,256,512,1024,2048), taus=(1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e9)):
    x = np.linspace(-L, L, N); cases, kind = build_cases(x)
    keys = list(cases)
    truth = {}; sN = {}; sn = {}; dec = {}
    for k in keys:
        u = cases[k]; D = abs(u[0]-u[-1])
        if D < 1e-12:  continue
        truth[k] = imin(dihiti(u, x), x)
        sN[k] = screen(u, x)/D
        sn[k] = {}; dec[k] = {}
        for n in ns:
            s = max(1, N//n); ud, xd = u[::s], x[::s]
            sn[k][n] = screen(ud, xd)/D
            dec[k][n] = imin(dihiti(ud, xd), xd)
    keys = [k for k in keys if k in truth]

    # --- timing primitives (logistic w=0.05 as the representative cost)
    u0 = cases["logistic w=0.05"]
    T = lambda f: min(timeit.repeat(f, number=10, repeat=7))/10
    t_B0 = T(lambda: imin(dihiti(u0, x), x))
    t_B1 = T(lambda: imin(onehilb(u0, x), x))
    t_scrN = T(lambda: screen(u0, x))
    t_dec = {}; t_scrn = {}
    for n in ns:
        s = N//n; ud, xd = u0[::s], x[::s]
        t_dec[n] = T(lambda ud=ud, xd=xd: imin(dihiti(ud, xd), xd))
        t_scrn[n] = T(lambda ud=ud, xd=xd: screen(ud, xd))
    print(f"N={N}  timing(us): B0 {t_B0*1e6:.1f}  B1 {t_B1*1e6:.1f}  screen@N {t_scrN*1e6:.1f}")
    for n in ns:
        print(f"   n={n:5d}: decim-dihiti {t_dec[n]*1e6:7.1f}  screen@n {t_scrn[n]*1e6:6.1f}")

    rows = []
    print(f"\n{'variant':7s} {'n':>5s} {'tau':>8s} {'falseANS':>9s} {'(phys)':>7s} {'(adv)':>6s} "
          f"{'defer%':>7s} {'E[t]us':>8s} {'vsB0':>7s} {'vsB1':>7s}")
    for variant in ('Sf', 'Sd'):
        for n in ns:
            for tau in taus:
                fa = fa_p = fa_a = nd = 0
                for k in keys:
                    sc = sN[k] if variant == 'Sf' else sn[k][n]
                    if sc > tau: nd += 1; continue
                    if (truth[k] >= 0) != (dec[k][n] >= 0):
                        fa += 1; fa_p += kind[k] == 'phys'; fa_a += kind[k] == 'adv'
                p = nd/len(keys)
                ts = t_scrN if variant == 'Sf' else t_scrn[n]
                Et = ts + (1-p)*t_dec[n] + p*t_B1
                print(f"{variant:7s} {n:5d} {tau:8.0e} {fa:9d} {fa_p:7d} {fa_a:6d} "
                      f"{100*p:6.1f}% {Et*1e6:8.1f} {t_B0/Et:7.2f} {t_B1/Et:7.2f}")
                rows.append(dict(variant=variant, n=n, tau=tau, false_ans=fa, fa_phys=fa_p,
                                 fa_adv=fa_a, defer=p, Et=Et, vsB0=t_B0/Et, vsB1=t_B1/Et))
    json.dump(dict(N=N, rows=rows, t_B0=t_B0, t_B1=t_B1, t_scrN=t_scrN,
                   t_dec={str(k): v for k, v in t_dec.items()},
                   t_scrn={str(k): v for k, v in t_scrn.items()},
                   nkeys=len(keys)), open(os.path.join(OUT, f"e1a2_N{N}.json"), "w"), indent=1)
    return rows

if __name__ == "__main__":
    run()
