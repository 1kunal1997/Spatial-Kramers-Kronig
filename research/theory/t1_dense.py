"""Dense vectorised scan of the balanced overlapping 2-arc extreme points."""
import numpy as np
from t1_fast import cl2, m1_arc, PI
TH = np.linspace(-PI, PI, 401)

def gains(p1,q1,p2,q2):
    """vectorised; returns gain (nan where infeasible)."""
    w1,w2=(q1-p1)/2,(q2-p2)/2; c1,c2=(p1+q1)/2,(p2+q2)/2
    ok=(w1>1e-9)&(w2>1e-9)&(np.minimum(q1,q2)>=np.maximum(p1,p2))
    ok&=(p1>-PI+1e-9)&(p2>-PI+1e-9)&(q1<PI-1e-9)&(q2<PI-1e-9)
    with np.errstate(all='ignore'):
        a=np.where(ok,m1_arc(np.where(ok,c1,0),np.where(ok,w1,1)),0.0)
        b=np.where(ok,m1_arc(np.where(ok,c2,0),np.where(ok,w2,1)),0.0)
        den=a-b; ok=ok&(np.abs(den)>1e-13)
        lam=np.where(ok,-b/np.where(ok,den,1.0),np.nan)
    ok=ok&(lam>=0)&(lam<=1)
    if not ok.any(): return np.full(p1.shape,np.nan), lam
    T=TH[None,:]
    def PA(th,c,w): return (cl2(th-c-w)-cl2(th-c+w))/(2*w)
    P = lam[:,None]*PA(T,c1[:,None],w1[:,None])+(1-lam)[:,None]*PA(T,c2[:,None],w2[:,None])
    Pi= lam*PA(np.array(PI),c1,w1)+(1-lam)*PA(np.array(PI),c2,w2)
    g=P.max(axis=1)-Pi
    return np.where(ok,g,np.nan), lam

rng=np.random.default_rng(0)
print("A) random dense scan, 2 general overlapping arcs inside (-pi,pi)")
best=(-9e9,None)
for it in range(12):
    K=20000
    E=rng.uniform(1e-6,0.3)           # varying seam clearance
    lo,hi=-PI+E,PI-E
    p1=rng.uniform(lo,hi,K); q1=rng.uniform(lo,hi,K)
    p2=rng.uniform(lo,hi,K); q2=rng.uniform(lo,hi,K)
    p1,q1=np.minimum(p1,q1),np.maximum(p1,q1); p2,q2=np.minimum(p2,q2),np.maximum(p2,q2)
    g,lam=gains(p1,q1,p2,q2)
    if np.all(np.isnan(g)): continue
    i=int(np.nanargmax(g))
    if g[i]>best[0]: best=(g[i],(p1[i]/PI,q1[i]/PI,p2[i]/PI,q2[i]/PI,lam[i]))
print(f"   max gain over 2.4e6 draws = {best[0]:.3e}  at (p1,q1|p2,q2)/pi={best[1]}")

print()
print("B) atom + wide arc:  lam*delta_c + (1-lam)*U[p,q],  p<=c<=q")
best=(-9e9,None)
for it in range(12):
    K=20000
    E=rng.uniform(1e-6,0.3); lo,hi=-PI+E,PI-E
    p=rng.uniform(lo,hi,K); q=rng.uniform(lo,hi,K); p,q=np.minimum(p,q),np.maximum(p,q)
    c=p+rng.uniform(0,1,K)*(q-p)
    e=1e-7
    g,lam=gains(c-e,c+e,p,q)
    if np.all(np.isnan(g)): continue
    i=int(np.nanargmax(g))
    if g[i]>best[0]: best=(g[i],(c[i]/PI,p[i]/PI,q[i]/PI,lam[i]))
print(f"   max gain over 2.4e6 draws = {best[0]:.3e}  at (c|p,q)/pi={best[1]}")

print()
print("C) NESTED arcs (the classic unimodal staircase): U[p1,q1] inside U[p2,q2]")
best=(-9e9,None)
for it in range(12):
    K=20000
    E=rng.uniform(1e-6,0.3); lo,hi=-PI+E,PI-E
    p2=rng.uniform(lo,hi,K); q2=rng.uniform(lo,hi,K); p2,q2=np.minimum(p2,q2),np.maximum(p2,q2)
    u=rng.uniform(0,1,(K,2)); u.sort(axis=1)
    p1=p2+u[:,0]*(q2-p2); q1=p2+u[:,1]*(q2-p2)
    g,lam=gains(p1,q1,p2,q2)
    if np.all(np.isnan(g)): continue
    i=int(np.nanargmax(g))
    if g[i]>best[0]: best=(g[i],(p1[i]/PI,q1[i]/PI,p2[i]/PI,q2[i]/PI,lam[i]))
print(f"   max gain over 2.4e6 draws = {best[0]:.3e}  at (p1,q1|p2,q2)/pi={best[1]}")
