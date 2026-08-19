"""ADJ1: what does Lemma 1 actually predict, and does it hold?
Distinguish (a) Delta(x0)  [replicator's literal reading]  from
            (b) min_x Delta [what Lemma 1 states and what cycle2 bisected on].
"""
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

N=2**16
x=np.linspace(-L,L,N); h=x[1]-x[0]
z=np.zeros_like(x)
x0=0.0
print(f"N={N} h={h:.3e}")
print(f"{'k(pts/2)':>9s} {'half/L':>10s} {'D(x0)/a':>12s} {'minD/a':>12s} {'argmin-x0 /half':>16s}")
rows=[]
for k in (4,8,16,32,64,128,256,512,1024,2048,4096):
    half=k*h
    if half>0.3*L: break
    p=cbump(x,x0,half)
    d=dihiti(p,x)                      # linear operator: Delta for a=1
    i0=np.argmin(np.abs(x-x0))
    interior=np.abs(x)<=FRAC*L
    j=np.argmin(np.where(interior,d,np.inf))
    rows.append(dict(k=k,half=half/L,d_x0=float(d[i0]),dmin=float(d[j]),arg=float((x[j]-x0)/half)))
    print(f"{k:9d} {half/L:10.3e} {d[i0]:12.4e} {d[j]:12.4e} {(x[j]-x0)/half:16.4f}")
json.dump(rows,open(os.path.join(OUT,"adj1.json"),"w"),indent=1)

# direct: max(-H[phi]) for this bump shape, computed on a fine standalone grid
M=2**18; s=np.linspace(-40,40,M)
phi=np.zeros_like(s); m=np.abs(s)<1; phi[m]=np.exp(-1.0/(1.0-s[m]**2))*np.e
Hphi=np.imag(hilbert(phi))   # scipy convention
print("\nmax(-H[phi]) =",(-Hphi).max()," at s=",s[np.argmin(Hphi)])
print("max(+H[phi]) =",(Hphi).max())
print("=> 1/(pi C_phi) should equal max(-H[phi]) ; implied C_phi =",1/(np.pi*(-Hphi).max()))
