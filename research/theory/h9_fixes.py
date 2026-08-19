"""Fixes for the three quadrature defects in h9_domination_check.py (blocks B, C, E, F)."""
import sys, os
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))

def dihiti(e, x):
    v = np.imag(hilbert(np.gradient(e, x)))
    c = cumulative_trapezoid(v, x, initial=0.0)
    return c - np.linspace(c[0], c[-1], len(x))

def logln(eta, xg, dxi):
    """ln|eta-xi| with the singular cell replaced by its exact cell average
       (1/h)int_{-h/2}^{h/2} ln|s| ds = ln(h/2) - 1."""
    d = np.abs(eta - xg)
    out = np.empty_like(d)
    bad = d < 0.5*dxi
    out[~bad] = np.log(d[~bad])
    out[bad] = np.log(np.maximum(dxi[bad], 1e-300)/2.0) - 1.0
    return out

def gap(g, t, eta, L):
    """Delta(eta) = int ln|eta-xi(t)| g dt - (D/2)ln(1+eta^2), singular cell corrected."""
    xg = np.tan(np.pi*t/(2*L))
    dxi = np.gradient(xg)                      # local xi-spacing
    D = np.trapezoid(g, t)
    eta = np.atleast_1d(np.asarray(eta, float))
    out = np.array([np.trapezoid(logln(e, xg, dxi)*g, t) for e in eta])
    return out - 0.5*D*np.log(1+eta**2)

L = 1.0; nb = 1.7
def logistic(x, w): return 1.0 + (nb**2-1)/(1.0+np.exp(x/w))

print("== B (fixed). gap identity vs DiHiTI ==")
for w, N in [(0.08, 4001), (0.05, 8001), (0.20, 4001)]:
    x = np.linspace(-L, L, N); ep = logistic(x, w); g = -np.gradient(ep, x)
    m = np.abs(x) < 0.97*L
    lhs = -gap(g, x, np.tan(np.pi*x[m]/(2*L)), L)
    ei = dihiti(ep, x)
    rel = np.max(np.abs(lhs - np.pi*ei[m]))/np.max(np.pi*ei)
    print(f"  w={w}, N={N}: max rel err {rel:.2e}   [{'PASS' if rel<1e-2 else 'FAIL'}]")

print("== C (fixed). verdict agreement on labelled corpus ==")
import profile_generator as pg
rng = np.random.default_rng(7)
agree=0; tot=0; dis=[]; margins=[]
for smp in pg.sample(240, rng=rng):
    xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
    mm = np.abs(xs) < 0.97*Ls
    dl = gap(gs, xs, np.tan(np.pi*xs[mm]/(2*Ls)), Ls)
    scale = max(smp['e_im'].max(), 1e-12)
    dom_min = -dl.max()/np.pi
    v_dom = dom_min >= -1e-3*scale
    tot += 1
    margins.append((dom_min, smp['min_eim']))
    if v_dom == smp['passive']: agree += 1
    else: dis.append((smp['name'], round(smp['min_eim'],5), round(dom_min,5)))
margins = np.array(margins)
ok = margins[:,1] > -1e9
print(f"  agreement {agree}/{tot}")
print(f"  max |domination min - DiHiTI min| = {np.max(np.abs(margins[:,0]-margins[:,1])):.2e}")
print(f"  disagreements: {dis[:8]}")

print("== E (fixed). necessity for a+b>L, near-atomic balanced pairs ==")
Ne = 2**14; xe = np.linspace(-L, L, Ne); tanx = np.tan(np.pi*xe/(2*L))
def eps_of(gv):
    G = cumulative_trapezoid(gv, xe, initial=0.0); return 1.0 + G[-1] - G
for (a, b) in [(0.5,0.5),(0.25,0.75),(0.55,0.55),(0.3,0.85),(0.2,0.95),(0.45,0.60)]:
    bt, gm = np.tan(np.pi*b/(2*L)), np.tan(np.pi*a/(2*L))
    s = 0.004
    gv = gm*np.exp(-0.5*((xe-b)/s)**2) + bt*np.exp(-0.5*((xe+a)/s)**2)
    D = np.trapezoid(gv, xe)
    mn = dihiti(eps_of(gv), xe).min()/D
    A, B = np.pi*b/(2*L), np.pi*a/(2*L)
    cf = -(np.tan(A)*np.log(np.sin(A)/np.cos(B)) + np.tan(B)*np.log(np.sin(B)/np.cos(A)))/((bt+gm)*np.pi)
    print(f"  a={a} b={b} (a+b={a+b:.2f}): DiHiTI min/D={mn:+.5f}  closed form={min(cf,0.0):+.5f}"
          f"   [{'passive' if a+b<=1 else 'GAIN'} predicted]")

print("== F (fixed). node count <= n ==")
def nodes(ts, ws, L=L):
    xs = np.tan(np.pi*np.asarray(ts,float)/(2*L)); ws = np.asarray(ws,float); D = ws.sum(); n = len(xs)
    Q = np.zeros(n+2)
    for j in range(n):
        term = np.polymul(np.poly(np.delete(xs,j)), [1.0,0.0,1.0]); Q[-len(term):] += ws[j]*term
    t2 = np.polymul(np.poly(xs), [D,0.0]); Q[-len(t2):] -= t2
    tol = 1e-10*np.max(np.abs(Q)) if np.max(np.abs(Q))>0 else 0
    nz = np.nonzero(np.abs(Q) > tol)[0]
    if len(nz)==0: return np.array([])
    Q = Q[nz[0]:]
    if len(Q)<=1: return np.array([])
    r = np.roots(Q); return np.real(r[np.abs(r.imag) < 1e-8*(1+np.abs(r.real))]), len(Q)-1
rng = np.random.default_rng(3); worst = -99
for _ in range(400):
    n = int(rng.integers(1,10)); ts = np.sort(rng.uniform(-0.95,0.95,n)); ws = rng.uniform(0.05,1,n)
    r, deg = nodes(ts, ws); worst = max(worst, deg - n)
print(f"  max(deg Q - n) over 400 random atomic profiles = {worst}   [{'PASS' if worst<=0 else 'FAIL'}]")
