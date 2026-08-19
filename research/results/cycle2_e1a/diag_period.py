"""Is the onehilb decimation error the endpoint-DUPLICATION periodization artifact?
np.linspace(-L,L,N) has p[0]==p[-1] for the ramp-periodized p, so feeding all N samples to
scipy.signal.hilbert declares the period to be 2L*N/(N-1), not 2L.  Drop the duplicate."""
import numpy as np
from scipy.signal import hilbert
from e1a3 import build_cases, L, A, FRAC, ramp

def oh_dup(u,x):            # current convention: N samples, endpoint duplicated
    v=np.imag(hilbert(ramp(u,x))); return v-v[0]
def oh_per(u,x):            # correct: one period = N-1 samples
    p=ramp(u,x)[:-1]; v=np.imag(hilbert(p)); v=v-v[0]
    return np.append(v, v[0])
def imin(v,x,frac=FRAC):
    Lx=0.5*(x[-1]-x[0]); return v[np.abs(x)<=frac*Lx].min()

N=2**16; x=np.linspace(-L,L,N); cases,_=build_cases(x,nrand=0)
for key in ['logistic w=0.05','logistic w=0.3','bumps a=1.0','cos m=257']:
    u=cases[key]; D=abs(u[0]-u[-1])
    print(f"--- {key}")
    td=imin(oh_dup(u,x),x)/D; tp=imin(oh_per(u,x),x)/D
    print(f"   full N: dup {td:+.6e}   per {tp:+.6e}")
    for n in (256,512,1024,2048,4096):
        s=N//n; ud,xd=u[::s],x[::s]
        print(f"   n={n:5d}: dup {imin(oh_dup(ud,xd),xd)/D:+.4e} (err {abs(imin(oh_dup(ud,xd),xd)/D-td):.2e})"
              f"   per {imin(oh_per(ud,xd),xd)/D:+.4e} (err {abs(imin(oh_per(ud,xd),xd)/D-tp):.2e})")
