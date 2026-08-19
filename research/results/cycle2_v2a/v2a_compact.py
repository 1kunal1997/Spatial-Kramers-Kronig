"""V2a, airtight version: COMPACTLY supported C^inf bumps on exactly 2k grid points.
Needs a UNIFORM constant: report min AND max of a_min over many disjoint positions."""
import numpy as np, json, os
from v2a_hard_family import dihiti, imin, L, NB, A, W, FRAC
OUT=os.path.dirname(os.path.abspath(__file__))

def cbump(x, x0, half):
    t=(x-x0)/half; y=np.zeros_like(x); m=np.abs(t)<1
    y[m]=np.exp(-1.0/(1.0-t[m]**2))*np.e     # normalised to peak 1
    return y

def a_min(u0,x,x0,half):
    best=None
    for sgn in (+1.,-1.):
        f=lambda a: imin(dihiti(u0+sgn*a*cbump(x,x0,half),x),x)
        lo,hi=0.,1e-3; t=0
        while f(hi)>=0 and t<40: hi*=1.7; t+=1
        if f(hi)>=0: continue
        for _ in range(45):
            mid=.5*(lo+hi)
            if f(mid)>=0: lo=mid
            else: hi=mid
        if best is None or hi<best: best=hi
    return best

res=[]
print(f"{'N':>7s} {'k(pts/2)':>9s} {'m0/D':>10s} {'min a/D':>10s} {'max a/D':>10s} "
      f"{'max/m0':>8s} {'#flip':>6s} {'#disjoint':>10s} {'maxTV/D':>9s}")
for N in (2**12, 2**13, 2**14):
    x=np.linspace(-L,L,N); h=x[1]-x[0]
    u0=1+A/(1+np.exp(x/W)); D=u0[0]-u0[-1]; m0=imin(dihiti(u0,x),x)
    for k in (4, 8, 16):
        half=k*h
        xs=np.linspace(-0.95*L,0.95*L,41)
        a=[a_min(u0,x,x0,half) for x0 in xs]
        ok=[v for v in a if v is not None]
        if len(ok)<len(a): print(f"{N:7d} {k:9d}  only {len(ok)}/{len(a)} positions flip"); 
        if not ok: continue
        ndis=int(2*FRAC*L/(2*half))
        print(f"{N:7d} {k:9d} {m0/D:10.3e} {min(ok)/D:10.3e} {max(ok)/D:10.3e} {max(ok)/m0:8.2f} "
              f"{len(ok):6d} {ndis:10d} {2*max(ok)/D:9.3e}")
        res.append(dict(N=N,k=k,m0=m0/D,amin=min(ok)/D,amax=max(ok)/D,ratio=max(ok)/m0,
                        nflip=len(ok),ntried=len(a),ndisjoint=ndis))
json.dump(res,open(os.path.join(OUT,"v2a_compact.json"),"w"),indent=1)
