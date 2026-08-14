"""Constructor v2: eps' = -H[eps''] + c0 + c1*x , with eps'' >= 0 by basis.
The ramp c1*x is the non-periodic part that carries the index step; it
contributes NOTHING to eps'' (H[linear]=0), so passivity is decided entirely
by the periodic remainder. This is exactly the decomposition DiHiTI performs."""
import sys; sys.path.insert(0,"/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
import tmm_helper as tmm_h
nb,eps_b,L,N = 1.7,1.7**2,2.5,2048
xx=np.linspace(-L,L,N)
def H(f): return np.imag(hilbert(f))
def dihiti(e):
    n=len(e); c=cumulative_trapezoid(H(np.gradient(e,xx)),xx,initial=0)
    return c-np.linspace(c[0],c[-1],n)
target = tmm_h.logistic(xx,4.0,nb)

K=60; h=2.2*(2*L)/(K+1); ctr=np.linspace(-L+h,L-h,K)
Phi=np.zeros((N,K))
for k,xc in enumerate(ctr):
    m=np.abs(xx-xc)<h; Phi[m,k]=0.5*(1+np.cos(np.pi*(xx[m]-xc)/h))
M=np.column_stack([-H(Phi[:,k]) for k in range(K)])
A=np.column_stack([M,np.ones(N),xx])                 # + const + RAMP
lo=np.array([0.0]*K+[-np.inf,-np.inf]); hi=np.full(K+2,np.inf)
s=lsq_linear(A,target,bounds=(lo,hi),max_iter=800)
eim=Phi@s.x[:K]; ere=A@s.x
eid=dihiti(target)
fc,_,_=tmm_h.skk_spectral_fom(xx,ere,eim); fd,_,_=tmm_h.skk_spectral_fom(xx,target,eid)
print("="*88); print("CONSTRUCTOR v2  (ramp included)"); print("="*88)
print(f"  {'':<32}{'CONSTRUCTED':>16}{'DiHiTI (paper)':>18}")
for lb,a,b in [("min eps'' (gain if <0)",eim.min(),eid.min()),
               ("peak eps''",eim.max(),eid.max()),
               ("total loss INT eps''",np.trapezoid(eim,xx),np.trapezoid(eid,xx)),
               ("spectral FoM %",fc,fd),
               ("eps' swing achieved",ere[0]-ere[-1],target[0]-target[-1]),
               ("max |eps'-target|",np.abs(ere-target).max(),0.0),
               ("rms |eps'-target|",float(np.sqrt(((ere-target)**2).mean())),0.0)]:
    print(f"  {lb:<32}{a:>16.6f}{b:>18.6f}")
print(f"  {'fitted ramp slope c1':<32}{s.x[K+1]:>16.6f}")
print(f"  {'implied endpoint jump':<32}{abs(2*L*s.x[K+1]):>16.6f}")
print()
print("="*88); print("TMM  n_b | coating | air   (s-pol, normal incidence)"); print("="*88)
def stack(er,ei,nl=400):
    i=np.linspace(0,N-1,nl).astype(int)
    return ([nb]+list(np.sqrt(er[i]+1j*ei[i]))+[1.0],
            [np.inf]+list(np.full(nl,2*L/nl))+[np.inf])
nb_,db_=[nb,1.0],[np.inf,np.inf]
nd,dd=stack(target,eid); nc,dc=stack(ere,eim)
print(f"{'lambda/L':>10}{'R bare':>12}{'R DiHiTI':>13}{'R constructed':>16}{'A DiHiTI':>11}{'A constr':>11}")
for lo_ in (0.2,0.5,1.0,2.0,5.0,10.0):
    lam=lo_*2*L
    _,Rb,_=tmm_h.TRA(nb_,db_,lamb=lam,angle=0,pol='s')
    _,Rd,Ad=tmm_h.TRA(nd,dd,lamb=lam,angle=0,pol='s')
    _,Rc,Ac=tmm_h.TRA(nc,dc,lamb=lam,angle=0,pol='s')
    print(f"{lo_:>10.2f}{Rb:>12.3e}{Rd:>13.3e}{Rc:>16.3e}{Ad:>11.3f}{Ac:>11.3f}")
