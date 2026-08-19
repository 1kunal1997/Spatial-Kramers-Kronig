import numpy as np, time
from t1_fast import gain_batch, PI
TH = np.linspace(-PI, PI, 1201)
rng = np.random.default_rng(0)
EPS = 1e-5
print(f"{'phi_m':>7} {'maxgain':>11}  worst (p1,q1,p2,q2)  lam")
res={}
for phim in [0.0,0.2,0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.2,2.5,2.8,3.0]:
    best=(-9e9,None,None)
    for _ in range(12):
        K=20000
        p1=rng.uniform(-PI+EPS,phim,K); q1=rng.uniform(phim,PI-EPS,K)
        p2=rng.uniform(-PI+EPS,phim,K); q2=rng.uniform(phim,PI-EPS,K)
        g,lam=gain_batch(p1,q1,p2,q2,TH)
        i=int(np.nanargmax(g))
        if g[i]>best[0]: best=(g[i],(p1[i],q1[i],p2[i],q2[i]),lam[i])
    # local refine
    x0=np.array(best[1]); s=0.3
    for _ in range(60):
        K=4000
        cand=x0+rng.normal(0,s,(K,4))
        cand[:,0]=np.clip(cand[:,0],-PI+EPS,phim); cand[:,2]=np.clip(cand[:,2],-PI+EPS,phim)
        cand[:,1]=np.clip(cand[:,1],phim,PI-EPS);  cand[:,3]=np.clip(cand[:,3],phim,PI-EPS)
        g,lam=gain_batch(cand[:,0],cand[:,1],cand[:,2],cand[:,3],TH)
        i=int(np.nanargmax(g))
        if g[i]>best[0]: best=(g[i],tuple(cand[i]),lam[i]); x0=cand[i]
        s*=0.94
    b=best[1]
    res[phim]=best
    print(f"{phim:7.2f} {best[0]:11.6f}  ({b[0]:+.4f},{b[1]:+.4f},{b[2]:+.4f},{b[3]:+.4f}) lam={best[2]:.5f}")
