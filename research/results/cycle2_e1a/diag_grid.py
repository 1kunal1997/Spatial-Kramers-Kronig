"""Does C22's 1.70x survive once the periodization is correct?
Grid A (mission convention): x = linspace(-L, L, N)   -> endpoint duplicated, period must be N-1.
Grid B (proper periodic):    x = -L + 2L*k/N, k=0..N-1 -> N samples = one period, power of two."""
import numpy as np, timeit
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
L=1.0; nb=1.7; A=nb**2-1; FRAC=0.95
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0); return e-np.linspace(e[0],e[-1],len(x))
def imin(v,x):
    Lx=0.5*(x[-1]-x[0]); return v[np.abs(x)<=FRAC*Lx].min()
T=lambda f: min(timeit.repeat(f,number=10,repeat=7))/10
for N in (2**12,2**14,2**16):
    # Grid A
    xa=np.linspace(-L,L,N); ua=1+A/(1+np.exp(xa/0.05)); Da=ua[0]-ua[-1]
    def ohA(u=ua,x=xa,D=Da):
        p=(u+(D/(2*L))*x)[:-1]; v=np.imag(hilbert(p)); return (v-v[0]).min()
    def ohA_dup(u=ua,x=xa,D=Da):
        v=np.imag(hilbert(u+(D/(2*L))*x)); return (v-v[0]).min()
    # Grid B
    xb=-L+2*L*np.arange(N)/N; ub=1+A/(1+np.exp(xb/0.05)); Db=ub[0]-ub[-1]-(ub[0]-ub[-1])*0
    Db=ub[0]-(ub[-1]+(ub[-1]-ub[-2]))   # extrapolate the missing endpoint
    def ohB(u=ub,x=xb,D=Db):
        p=u+(D/(2*L))*x; v=np.imag(hilbert(p)); return (v-v[0]).min()
    tA=T(lambda: imin(dihiti(ua,xa),xa)); tAo=T(ohA); tAd=T(ohA_dup); tB=T(ohB)
    tBd=T(lambda: imin(dihiti(ub,xb),xb))
    print(f"N={N:6d}  gridA dihiti {tA*1e6:8.1f}us | onehilb(N-1, correct) {tAo*1e6:8.1f}us "
          f"({tA/tAo:5.2f}x) | onehilb(N, duplicated/buggy) {tAd*1e6:7.1f}us ({tA/tAd:5.2f}x)")
    print(f"          gridB dihiti {tBd*1e6:8.1f}us | onehilb(N, correct)   {tB*1e6:8.1f}us ({tBd/tB:5.2f}x)")
    print(f"          N-1 = {N-1} factors: ", end="")
    m=N-1; f=[]
    d=2
    while d*d<=m:
        while m%d==0: f.append(d); m//=d
        d+=1
    if m>1: f.append(m)
    print(f)
