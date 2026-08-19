"""E11 -- degenerate / extreme corners of the BOX form of C25.
   (i) a = 0  (support one-sided)
   (ii) b -> L  (Cayley coordinate blows up)
   (iii) how far can a+b be pushed and still admit a passive corner-touching measure?
        (i.e. how big is the counterexample regime found in E1)
"""
import numpy as np
from common import *
L=1.0
xg = periodic_grid(1<<18, L)
def cp(a,b):
    wm,wp = np.tan(np.pi*b/2), np.tan(np.pi*a/2); s=wm+wp; return [(-a,wm/s),(b,wp/s)]
def imin(at, ex=2e-3, frac=0.99999):
    y = exact_kernel(xg,at,L); return interior_min(xg,y,L,frac,[t for t,_ in at],ex)

print("(i) a -> 0: with g>=0 supported in [0,b], M1=0 forces all mass at t=0.")
at=[(0.0,1.0)]
print(f"    single atom at 0: M1={M1_atoms(at):.1e}  min eps''={imin(at)[0]:+.4e}  "
      f"(kernel = -ln|sin(pi x/2)|/pi >= 0 everywhere, exactly)  -> trivially PASSIVE for ANY b<=L")

print("\n(ii) b -> L with a = L-b (a+b = L exactly). Cayley coord tan(pi b/2) -> inf:")
for e in (1e-1,1e-2,1e-3,1e-4,1e-6,1e-8):
    a,b = e, 1-e
    at = cp(a,b); m,xm = imin(at)
    print(f"    a={a:.0e} b={1-e:.10f}: weights={at[0][1]:.3e},{at[1][1]:.3e} "
          f"min eps''={m:+.3e} at x={xm:+.6f}   {'PASS' if m>=-1e-13 else 'NEG'}")

print("\n(iii) HOW BIG is the counterexample regime?  measure = lam*(corner pair on [-a,b])")
print("      + (1-lam)*(balanced pair at +-0.1).  Largest lam still PASSIVE:")
def bp(tm,tp):
    wm,wp=np.tan(np.pi*tp/2),np.tan(np.pi*tm/2); s=wm+wp; return [(-tm,wm/s),(tp,wp/s)]
mid = bp(0.1,0.1)
for (a,b) in [(0.4,0.9),(0.5,0.9),(0.9,0.9),(0.5,1.4),(0.95,0.99),(0.99,0.99)]:
    lo,hi = 0.0,1.0
    for _ in range(50):
        lam=(lo+hi)/2
        P=[(t,lam*w) for t,w in cp(a,b)]+[(t,(1-lam)*w) for t,w in mid]
        m,_=imin(P, 2e-3)
        if m>=0: lo=lam
        else: hi=lam
    print(f"    box [-{a},{b}] a+b={a+b:.2f}L : passive for all lam <= {lo:.4f} "
          f"(i.e. up to {100*lo:.1f}% of the mass sitting on the two corners)")
