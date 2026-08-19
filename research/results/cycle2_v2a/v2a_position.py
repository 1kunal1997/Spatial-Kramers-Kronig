"""Position-resolved flip amplitude: is a_min(x0) ~ pi * eps''_0(x0)?
If so the hard family's uniform constant is set by max eps''_0 = O(D), N-independent."""
import numpy as np, json, os
from v2a_hard_family import dihiti, imin, L, NB, A, W, FRAC
from v2a_compact import cbump, a_min
OUT=os.path.dirname(os.path.abspath(__file__))
N=2**13; x=np.linspace(-L,L,N); h=x[1]-x[0]
u0=1+A/(1+np.exp(x/W)); D=u0[0]-u0[-1]; e0=dihiti(u0,x); m0=imin(e0,x)
k=8; half=k*h
print(f"N={N} m0/D={m0/D:.4e}  half-width={half:.3e} = {2*k} grid points")
print(f"{'x0/L':>8s} {'eps0(x0)/D':>12s} {'a_min/D':>11s} {'a_min/(pi eps0)':>16s}")
rows=[]
for x0 in np.linspace(-0.95*L,0.95*L,21):
    a=a_min(u0,x,x0,half); e=np.interp(x0,x,e0)
    print(f"{x0/L:+8.3f} {e/D:12.4e} {a/D:11.4e} {a/(np.pi*max(e,1e-30)):16.3f}")
    rows.append(dict(x0=x0/L,e0=e/D,a=a/D,ratio=float(a/(np.pi*max(e,1e-30)))))
json.dump(rows,open(os.path.join(OUT,"v2a_position.json"),"w"),indent=1)
