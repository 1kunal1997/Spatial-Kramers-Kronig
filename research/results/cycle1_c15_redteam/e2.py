import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
nb=1.7; L=1.0; A=nb**2-1
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0)
    return e-np.linspace(e[0],e[-1],len(x))
N=2**13; x=np.linspace(-L,L,N)

# 1. LINEARITY: is DiHiTI a linear operator on eps'? (=> passive set is a convex cone)
u1=1+A/(1+np.exp(x/0.05)); u2=1+A/(1+np.exp(x/0.2))
lhs=dihiti(2.3*u1-1.7*u2,x); rhs=2.3*dihiti(u1,x)-1.7*dihiti(u2,x)
print("linearity max err:", np.abs(lhs-rhs).max(), " scale:", np.abs(rhs).max())

# 2. convex cone check: passive + passive, and positive scaling
for t in [0.0,0.25,0.5,0.75,1.0]:
    u=t*u1+(1-t)*u2
    print(f"  t={t}: min eps''={dihiti(u,x).min():.3e}")

# 3. Build the matrix A explicitly for small N and inspect: is it dense? rank? facets
n=256; xs=np.linspace(-L,L,n)
M=np.zeros((n,n))
for j in range(n):
    e=np.zeros(n); e[j]=1.0
    M[:,j]=dihiti(e,xs)
print("matrix rank:", np.linalg.matrix_rank(M), "of", n)
print("row 0 (seam) max abs:", np.abs(M[0]).max(), " row n-1:", np.abs(M[-1]).max())
print("density (frac |A_ij|>1e-12):", (np.abs(M)>1e-12).mean())
# how many DISTINCT active constraints define the cone? i.e. are rows redundant?
# check: is any row a nonneg combination of others (=> redundant facet)?  cheap proxy: normalized row correlations
R=M/np.linalg.norm(M,axis=1,keepdims=True)
C=R@R.T
np.fill_diagonal(C,0)
print("max off-diag row correlation:", C.max())
