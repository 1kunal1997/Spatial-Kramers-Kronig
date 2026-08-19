"""ADJ3: (i) small-k behaviour of the compact-bump law (how few grid points before it dies);
        (ii) FIXED-x0 a_min vs Gaussian width sigma -- does it really rise at small sigma (M4)?
        (iii) M4's easiest-position migration."""
import numpy as np, os
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
L=2.5; NB=1.7; A=NB**2-1; W=0.05; FRAC=0.95
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0); return e-np.linspace(e[0],e[-1],len(x))
def cbump(x,x0,half):
    t=(x-x0)/half; y=np.zeros_like(x); m=np.abs(t)<1
    y[m]=np.exp(-1.0/(1.0-t[m]**2))*np.e; return y
def gb(x,x0,s): return np.exp(-((x-x0)/s)**2)
def imin(v,x): return v[np.abs(x)<=FRAC*L].min()

N=2**15; x=np.linspace(-L,L,N); h=x[1]-x[0]
u0=1+A/(1+np.exp(x/W)); D=u0[0]-u0[-1]; e0=dihiti(u0,x)
print("(i) compact bump, small k, min_x Delta/a  (asymptote -0.79750)")
for k in (1,2,3,4,6,8,12,16,24,32):
    d=dihiti(cbump(x,0.0,k*h),x); print(f"   k={k:4d} ({2*k:5d} pts)  minD/a = {imin(d,x):+.5f}")

print("\n(ii) FIXED x0, a_min vs Gaussian sigma (M4's quantity, but at fixed position)")
print(f"{'sig/L':>9s} {'sig/h':>8s} {'minD/a':>10s} {'a_min/D (x0=0)':>15s} {'a_min/D (x0=-0.9L)':>19s}")
for sl in (0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001):
    s=sl*L
    out=[]
    for x0 in (0.0,-0.9*L):
        d=dihiti(gb(x,x0,s),x); md=imin(d,x); e=np.interp(x0,x,e0)
        out.append((e/max(-md,1e-30))/D)
    d0=imin(dihiti(gb(x,0.0,s),x),x)
    print(f"{sl:9.4f} {s/h:8.1f} {d0:10.5f} {out[0]:15.5e} {out[1]:19.5e}")

print("\n(iii) M4 easiest-position scan reproduced (25 positions in [-0.97L,0.97L]), N=2**14")
N2=2**14; x2=np.linspace(-L,L,N2); u02=1+A/(1+np.exp(x2/W)); D2=u02[0]-u02[-1]; e02=dihiti(u02,x2)
for sl in (0.5,0.1,0.02,0.01,0.005):
    s=sl*L; best=None
    for x0 in np.linspace(-0.97*L,0.97*L,25):
        d=dihiti(gb(x2,x0,s),x2); md=imin(d,x2)
        if md>=0: continue
        e=np.interp(x0,x2,e02); a=e/(-md)
        if best is None or a<best[0]: best=(a,x0)
    print(f"   sigma/L={sl:6.3f}  a_min/D={best[0]/D2:.5e}  x0*={best[1]:+.4f} (={best[1]/L:+.3f}L)")
