import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
nb=1.7; L=1.0; A=nb**2-1
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0); return e-np.linspace(e[0],e[-1],len(x))
print(" w    N     min/D@0.95L   min/D@0.99L   min/D@(1-2/N)L")
for w in [0.05,0.3]:
  for N in [2**10,2**12,2**14,2**16]:
    x=np.linspace(-L,L,N); u=1+A/(1+np.exp(x/w)); v=dihiti(u,x); D=u[0]-u[-1]
    r=[]
    for f in [0.95,0.99]:
        m=np.abs(x)<=f*L; r.append(v[m].min()/D)
    r.append(v[2:-2].min()/D)
    print(f" {w}  {N:6d}  {r[0]:.4e}   {r[1]:.4e}   {r[2]:.4e}")
