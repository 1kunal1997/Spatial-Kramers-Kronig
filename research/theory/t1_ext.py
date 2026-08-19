"""Is the extremal unimodal balanced measure on [P,Q] always 'atom at an end + U[P,Q]'?"""
import numpy as np
from t1_fast import cl2, m1_arc, PI
TH=np.linspace(-PI,PI,601)

def _A(th,p,q): return (cl2(th-q)-cl2(th-p))/(q-p)

def gains(p1,q1,p2,q2):
    ok=(q1-p1>1e-9)&(q2-p2>1e-9)&(np.minimum(q1,q2)>=np.maximum(p1,p2))
    with np.errstate(all='ignore'):
        a=m1_arc((p1+q1)/2,np.where(ok,(q1-p1)/2,1)); b=m1_arc((p2+q2)/2,np.where(ok,(q2-p2)/2,1))
        den=a-b; ok=ok&np.isfinite(den)&(np.abs(den)>1e-13)
        lam=np.where(ok,-b/np.where(ok,den,1.0),np.nan)
    ok=ok&(lam>=0)&(lam<=1)
    if not ok.any(): return np.full(np.shape(p1),np.nan)
    T=TH[None,:]
    P=lam[:,None]*_A(T,p1[:,None],q1[:,None])+(1-lam)[:,None]*_A(T,p2[:,None],q2[:,None])
    Pi=lam*_A(np.array(PI),p1,q1)+(1-lam)*_A(np.array(PI),p2,q2)
    return np.where(ok,P.max(axis=1)-Pi,np.nan)

def full_search(P,Q,rng,rounds=5,K=8000):
    best=-9e9; arg=None
    for _ in range(rounds):
        u=rng.uniform(0,1,(K,4))
        p1=P+u[:,0]*(Q-P); q1=p1+u[:,1]*(Q-p1)
        p2=P+u[:,2]*(Q-P); q2=p2+u[:,3]*(Q-p2)
        g=gains(p1,q1,p2,q2)
        if np.all(np.isnan(g)): continue
        i=int(np.nanargmax(g))
        if g[i]>best: best=g[i]; arg=(p1[i],q1[i],p2[i],q2[i])
    # polish
    if arg is not None:
        x0=np.array(arg); s=(Q-P)*0.2
        for _ in range(30):
            c=np.clip(x0+rng.normal(0,s,(2000,4)),P,Q)
            c[:,1]=np.maximum(c[:,1],c[:,0]); c[:,3]=np.maximum(c[:,3],c[:,2])
            g=gains(c[:,0],c[:,1],c[:,2],c[:,3])
            if not np.all(np.isnan(g)):
                i=int(np.nanargmax(g))
                if g[i]>best: best=g[i]; x0=c[i]
            s*=0.9
    return best

def end_atom(P,Q,e=1e-6):
    out=[]
    for c in (P,Q):
        g=gains(np.array([c-e]),np.array([c+e]),np.array([P]),np.array([Q]))[0]
        out.append(-9.0 if np.isnan(g) else g)
    return max(out)

rng=np.random.default_rng(5)
print(f"{'P/pi':>7} {'Q/pi':>7} {'full 4-param':>13} {'atom-at-end':>12}   verdict")
for Pf,Qf in [(-0.5,0.999),(-0.5,0.97),(-0.8,0.99),(-0.25,0.995),(-0.9,0.95),(-0.6,0.9),
              (-0.95,0.99),(-0.3,0.999),(-0.7,0.999),(-0.99,0.99),(-0.5,0.9),(-0.4,0.8)]:
    P,Q=Pf*PI,Qf*PI
    fs=full_search(P,Q,rng); ea=end_atom(P,Q)
    print(f"{Pf:7.3f} {Qf:7.3f} {fs:13.6f} {ea:12.6f}   {'end-atom is extremal' if ea>=fs-1e-4 else 'FULL SEARCH BEATS IT'}")
