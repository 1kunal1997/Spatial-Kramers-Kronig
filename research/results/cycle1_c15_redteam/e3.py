import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
nb=1.7; L=1.0; A=nb**2-1
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0)
    return e-np.linspace(e[0],e[-1],len(x))
n=512; xs=np.linspace(-L,L,n)
M=np.zeros((n,n))
for j in range(n):
    e=np.zeros(n); e[j]=1.0; M[:,j]=dihiti(e,xs)
zr=np.where(np.abs(M).max(axis=1)<1e-14)[0]
print("identically-zero rows of DiHiTI matrix (indices):", zr, " -> eps''(+-L)=0 is an OPERATOR IDENTITY, not a facet")
print("rank:", np.linalg.matrix_rank(M), "of", n)
# interior margin of the logistic, normalized
for w in [0.05,0.15,0.3]:
    u=1+A/(1+np.exp(xs/w)); v=dihiti(u,xs); D=u[0]-u[-1]
    m=np.abs(xs)<0.999*L
    print(f" logistic w={w}: full min={v.min():.2e} interior min/D={v[m].min()/D:.4e} at x/L={xs[m][v[m].argmin()]/L:.4f}; max/D={v.max()/D:.3f}")
# convex-cone check on interior margins
u1=1+A/(1+np.exp(xs/0.05)); u2=1+A/(1+np.exp(xs/0.3))
m=np.abs(xs)<0.999*L
for t in [0,.25,.5,.75,1]:
    u=t*u1+(1-t)*u2; v=dihiti(u,xs)
    print(f"  t={t}: interior min={v[m].min():.3e}")
