"""E5/E7/E9 -- are the four hypotheses load-bearing, and how brittle are they?

H1 g >= 0        (eps' monotone non-increasing)
H2 M1 = 0        (balance)
H3 supp g in [-a,b] with a+b <= L
H4 the DiHiTI gauge (seam-normalised kernel; eps''(+-L)=0)
"""
import numpy as np
from common import *
L = 1.0
xg = periodic_grid(1 << 18, L)

def corner_pair(a, b):
    wm, wp = np.tan(np.pi*b/2), np.tan(np.pi*a/2); s = wm+wp
    return [(-a, wm/s), (b, wp/s)]

def imin(atoms, ex=2e-3, frac=0.99999):
    y = exact_kernel(xg, atoms, L)
    return interior_min(xg, y, L, frac, [t for t,_ in atoms], ex)

print("=== H2: how brittle is M1 = 0?  (box a+b = 0.5L, DEEP inside the safe region) ===")
base = corner_pair(0.25, 0.25)          # a+b = 0.5 L, comfortably passive
M2 = sum(w/np.cos(np.pi*t/2)**2 for t,w in base)
print(f"  base: M1={M1_atoms(base):+.1e}  M2={M2:.4f}  min eps''={imin(base)[0]:+.4e}")
print(f"  {'M1':>10} {'min eps(interior 0.99999L)':>28} {'argmin':>10} {'-M1^2/(2 pi M2)':>18} {'min over |x|<0.95L':>20}")
for d in (1e-1,1e-2,1e-3,1e-4,1e-5):
    at = [(-0.25, base[0][1]+d/np.tan(np.pi*0.25/2)*0 ), (0.25, base[1][1])]
    # perturb by adding a small extra atom at t=0.25 to create M1 = d
    at = base + [(0.25, d/np.tan(np.pi*0.25/2))]
    m, xm = imin(at)
    m95, _ = interior_min(exact_kernel(xg,at,L)*0+xg, xg, L)  # placeholder
    y = exact_kernel(xg, at, L)
    mm = interior_min(xg, y, L, 0.95, [t for t,_ in at], 2e-3)[0]
    print(f"  {M1_atoms(at):10.2e} {m:28.4e} {xm:10.5f} {-M1_atoms(at)**2/(2*np.pi*sum(w/np.cos(np.pi*t/2)**2 for t,w in at)):18.3e} {mm:20.4e}")
print("  -> gain induced by M1 != 0 lives in a SHRINKING neighbourhood of the seam;")
print("     a fixed interior window (e.g. 0.95L) reports PASS for every one of these.")

print("\n=== H1: drop g >= 0 (allow eps' to be non-monotone) ===")
print("  signed measure inside the MIDDLE HALF box a+b = 0.5L, M1 = 0:")
found = 0
rng = np.random.default_rng(7)
worst = 0
for _ in range(400):
    t = rng.uniform(-0.25, 0.25, size=3)
    w = rng.normal(size=3)
    # enforce M1 = 0 by projecting out the tan direction
    v = np.tan(np.pi*t/2); w = w - v*(w@v)/(v@v)
    at = list(zip(t, w/np.abs(w).sum()))
    m,_ = imin(at)
    if m < worst: worst = m; best = at
    if m < -1e-9: found += 1
print(f"  {found}/400 signed measures in [-0.25,0.25] with M1=0 are NON-passive; worst min eps''={worst:+.3e}")
print("  -> H1 (g>=0) is LOAD-BEARING: without it the box condition gives nothing.")

print("\n=== H4: the gauge.  Same measure, gauge factor cos(pi t/2L) removed ===")
at = corner_pair(0.5, 0.5)
y_g  = exact_kernel(xg, at, L)
y_ng = np.array([sum(-w*np.log(np.abs(np.sin(np.pi*(xg-t)/2)))/np.pi for t,w in at)][0])
print(f"  a+b = L exactly.  with gauge : min = {interior_min(xg,y_g,L,0.99999,[t for t,_ in at],2e-3)[0]:+.4e}")
print(f"                    no gauge   : min = {interior_min(xg,y_ng,L,0.99999,[t for t,_ in at],2e-3)[0]:+.4e}"
      f"  (and eps''(+-L) = {y_ng[0]:+.4f} != 0)")
print("  -> H4 load-bearing: the threshold a+b<=L is a statement about THIS gauge only.")

print("\n=== E9: boundary case a+b = L through the PRODUCTION pipeline (smooth g) ===")
for (a,b) in [(0.5,0.5),(0.25,0.75),(0.1,0.9)]:
    for N in (1<<15, 1<<17, 1<<19):
        x,e = dihiti_of_atoms(corner_pair(a,b), N=N, L=L)
        w = 40.0*(2*L/N)
        m,xm = interior_min(x,e,L,0.99999,[-a,b], 3*w)
        print(f"  a={a} b={b} N={N:7d}: min eps'' = {m:+.4e} at x={xm:+.5f}"
              f"   {'PASS' if m>=0 else '** NEGATIVE **'}")
