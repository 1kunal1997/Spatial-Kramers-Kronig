"""ADJ2: (i) is min_x Delta/a independent of x0 (the quantity Lemma 1 needs)?
        (ii) does the endpoint=True grid bug (C22) change anything?
        (iii) reproduce a_min(x0) with min-over-interior and check kappa spread."""
import numpy as np, json, os
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
OUT=os.path.dirname(os.path.abspath(__file__))
L=2.5; NB=1.7; A=NB**2-1; W=0.05; FRAC=0.95
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0)
    return e-np.linspace(e[0],e[-1],len(x))
def cbump(x,x0,half):
    t=(x-x0)/half; y=np.zeros_like(x); m=np.abs(t)<1
    y[m]=np.exp(-1.0/(1.0-t[m]**2))*np.e
    return y
def imin(v,x): return v[np.abs(x)<=FRAC*L].min()

for name,xx in (("endpoint=True (cycle2)",np.linspace(-L,L,2**15)),
                ("endpoint=False        ",np.linspace(-L,L,2**15,endpoint=False))):
    x=xx; h=x[1]-x[0]; u0=1+A/(1+np.exp(x/W)); D=u0[0]-u0[-1]
    e0=dihiti(u0,x); m0=imin(e0,x)
    print(f"\n### {name}  N={len(x)} m0/D={m0/D:.4e}")
    print(f"{'x0/L':>8s} {'minDelta/a':>12s} {'e0(x0)/D':>11s} {'a_min/D':>10s} {'kappa=a/e0':>11s} {'k/pi':>7s}")
    rows=[]
    for k in (64,):
        half=k*h
        for x0 in np.linspace(-0.9*L,0.9*L,13):
            d=dihiti(cbump(x,x0,half),x); md=imin(d,x)
            e=np.interp(x0,x,e0)
            # exact flip amplitude by linearity: need min(e0 + a*d) = 0
            aa=np.linspace(0,50*e/max(-md,1e-30),4001)[1:]
            f=[imin(e0+a*d,x) for a in aa]
            f=np.array(f); idx=np.argmax(f<0)
            amin=aa[idx] if f[idx]<0 else np.nan
            kap=amin/e
            rows.append(dict(x0=x0/L,md=float(md),e0=float(e/D),amin=float(amin/D),kap=float(kap)))
            print(f"{x0/L:+8.3f} {md:12.4e} {e/D:11.4e} {amin/D:10.4e} {kap:11.4f} {kap/np.pi:7.4f}")
    kk=np.array([r['kap'] for r in rows]); mm=np.array([r['md'] for r in rows])
    print(f"  kappa: min={kk.min():.4f} max={kk.max():.4f} ratio={kk.max()/kk.min():.3f} rms/mean={kk.std()/kk.mean()*100:.2f}%")
    print(f"  minDelta/a: min={mm.min():.5f} max={mm.max():.5f} ratio={mm.max()/mm.min():.4f}")
