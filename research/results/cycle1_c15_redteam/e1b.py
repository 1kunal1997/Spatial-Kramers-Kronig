import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
nb=1.7; L=1.0
def logistic(x,w): return 1.0+(nb**2-1)/(1.0+np.exp(x/w))
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0)
    return e-np.linspace(e[0],e[-1],len(x))
def onehilb(u,x):
    """v = H[eps' + (D/2L) x] + const, const fixed by v(-L)=0."""
    N=len(x); L=0.5*(x[-1]-x[0]); D=u[0]-u[-1]
    p = u + (D/(2*L))*x            # periodic-ized
    v = np.imag(hilbert(p))
    return v - v[0]
N=2**14; x=np.linspace(-L,L,N)
for w in [0.05,0.02,0.15]:
    u=logistic(x,w); a=dihiti(u,x); b=onehilb(u,x)
    print(f"w={w}: a.max={a.max():.5f} b.max={b.max():.5f} a.min={a.min():.2e} b.min={b.min():.2e} rel rms={np.sqrt(np.mean((a-b)**2))/a.max():.2e}")
    # interior comparison excluding 1% edges
    m=np.abs(x)<0.99*L
    print(f"   interior rel rms={np.sqrt(np.mean((a-b)[m]**2))/a.max():.2e}  a.min_int={a[m].min():.3e} b.min_int={b[m].min():.3e}")
