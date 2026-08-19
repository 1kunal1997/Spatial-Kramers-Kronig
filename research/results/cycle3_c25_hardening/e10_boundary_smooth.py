"""E10 -- the boundary a+b = L, done HONESTLY.

E9 found production DiHiTI returns min eps'' < 0 at a+b = L.  Diagnosis to test:
smoothing an atom AT the corner b spreads half its mass to t > b, so the smoothed
profile's true support is [-a-w/2, b+w/2] and a+b -> L + w: it VIOLATES H3.
Test 1: fix N, vary the smoothing width w -> if the negativity tracks w (not h), it is
        a hypothesis violation, not quadrature.
Test 2: build a genuinely SMOOTH g >= 0 whose support is contained in [-a,b] with
        a+b = L (bumps that die before the corners) and check the sign.
"""
import numpy as np
from common import *
L = 1.0

def corner_pair(a,b):
    wm, wp = np.tan(np.pi*b/2), np.tan(np.pi*a/2); s=wm+wp
    return [(-a,wm/s),(b,wp/s)]

print("TEST 1 -- fix N = 2^18, vary smoothing width w (a=b=0.5, a+b=L):")
N = 1<<18; h = 2*L/N
for k in (10, 20, 40, 80, 160, 320):
    w = k*h
    x = periodic_grid(N,L)
    e_re = eps_re_from_atoms(x, corner_pair(0.5,0.5), w, L)
    import tmm_helper as tmm_h
    e = tmm_h.ht_derivative(x, e_re)
    m,xm = interior_min(x,e,L,0.99999,[-0.5,0.5], 1.5*w)
    print(f"  w = {k:4d}h = {w:.3e}:  min eps'' = {m:+.4e} at x={xm:+.5f}   (w^2 = {w**2:.2e})")

print("\nTEST 2 -- smooth g >= 0 with support STRICTLY inside [-a,b], a+b = L exactly.")
print("  g = two raised-cosine bumps that vanish at the corners; M1 = 0 enforced exactly.")
def bump(t, c, half):
    y = np.zeros_like(t); m = np.abs(t-c) < half
    y[m] = 0.5*(1+np.cos(np.pi*(t[m]-c)/half))
    return y
for (a,b) in [(0.5,0.5),(0.25,0.75),(0.1,0.9)]:
  for N in (1<<15, 1<<17, 1<<19):
    x = periodic_grid(N,L)
    cm, hm = -a/2, a/2          # bump filling [-a, 0]
    cp, hp =  b/2, b/2          # bump filling [0, b]
    gm, gp = bump(x,cm,hm), bump(x,cp,hp)
    tn = np.tan(np.pi*x/2)
    Im = np.trapz(gm*tn, x); Ip = np.trapz(gp*tn, x)
    g = gm*(-Ip) + gp*(Im)      # M1 = 0 by construction (coeffs > 0 since Im<0<Ip)
    g = g/np.trapz(g,x)
    # eps'(x) = -int_{-L}^{x} g
    from scipy.integrate import cumulative_trapezoid
    e_re = -cumulative_trapezoid(g, x, initial=0)
    import tmm_helper as tmm_h
    e = tmm_h.ht_derivative(x, e_re)
    m,xm = interior_min(x,e,L,0.99999)
    mi,_ = interior_min(x,e,L,0.95)
    print(f"  a={a} b={b} N={N:7d}: M1={np.trapz(g*tn,x):+.2e}  min(0.99999L)={m:+.4e} at x={xm:+.4f}"
          f"   min(0.95L)={mi:+.4e}  {'PASS' if m>=-1e-14 else '** NEG **'}")
