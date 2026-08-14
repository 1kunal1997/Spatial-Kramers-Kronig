"""The decisive test: targets where DiHiTI PRODUCES GAIN.
Can the constructor repair them into passive profiles, and at what cost in R?"""
import sys; sys.path.insert(0,"/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
import tmm_helper as tmm_h
nb,L,N=1.7,2.5,2048; xx=np.linspace(-L,L,N); A_=nb**2-1.0
def H(f): return np.imag(hilbert(f))
def dihiti(e):
    n=len(e); c=cumulative_trapezoid(H(np.gradient(e,xx)),xx,initial=0)
    return c-np.linspace(c[0],c[-1],n)
K=60; h=2.2*(2*L)/(K+1); ctr=np.linspace(-L+h,L-h,K)
Phi=np.zeros((N,K))
for k,xc in enumerate(ctr):
    m=np.abs(xx-xc)<h; Phi[m,k]=0.5*(1+np.cos(np.pi*(xx[m]-xc)/h))
M=np.column_stack([-H(Phi[:,k]) for k in range(K)])
AA=np.column_stack([M,np.ones(N),xx])
lo=np.array([0.0]*K+[-np.inf,-np.inf]); hi=np.full(K+2,np.inf)
def construct(t):
    s=lsq_linear(AA,t,bounds=(lo,hi),max_iter=800)
    return Phi@s.x[:K], AA@s.x
def stack(er,ei,nl=400):
    i=np.linspace(0,N-1,nl).astype(int)
    return ([nb]+list(np.sqrt(er[i]+1j*ei[i]))+[1.0],
            [np.inf]+list(np.full(nl,2*L/nl))+[np.inf])
def g(x0,s): return np.exp(-((xx-x0)/s)**2)
logi=tmm_h.logistic(xx,4.0,nb)
targets=[('logistic (control)', logi),
         ('logistic+bump a=0.5', logi+0.5*A_*g(0,.6)),
         ('logistic+bump a=1.0', logi+1.0*A_*g(0,.6)),
         ('logistic+bump a=2.0', logi+2.0*A_*g(0,.6)),
         ('opp bumps a=1.0',     logi+1.0*(A_*g(-.9,.45)-A_*g(.9,.45))),
         ('triangle s=2.0',      1.0+2.0*(np.sqrt(L**2+.15**2)-np.sqrt(xx**2+.15**2)))]
print("="*100)
print(f"{'target':<22}{'min eps DiHiTI':>17}{'min eps constr':>17}"
      f"{'rms |dev|':>11}{'R DiHiTI':>12}{'R constr':>12}{'R bare':>11}")
print("="*100)
lam=1.0*2*L
nb_,db_=[nb,1.0],[np.inf,np.inf]
_,Rb,_=tmm_h.TRA(nb_,db_,lamb=lam,angle=0,pol='s')
for lb,t in targets:
    ed=dihiti(t); ec,rc=construct(t)
    nd,dd=stack(t,ed); nc,dc=stack(rc,ec)
    _,Rd,_=tmm_h.TRA(nd,dd,lamb=lam,angle=0,pol='s')
    _,Rc,_=tmm_h.TRA(nc,dc,lamb=lam,angle=0,pol='s')
    print(f"{lb:<22}{ed.min():>17.6f}{ec.min():>17.6f}"
          f"{float(np.sqrt(((rc-t)**2).mean())):>11.4f}{Rd:>12.3e}{Rc:>12.3e}{Rb:>11.3e}")
print("="*100)
print("(lambda/L = 1.0, s-pol, normal incidence.  'min eps' < 0 means GAIN.)")
