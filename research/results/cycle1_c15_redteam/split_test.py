"""Red-team V1 candidate: SPLIT passivity test.
   seam  -> analytic necessary cascade (O(N), no transform)
   interior (|x|<=0.95L) -> decimated single-Hilbert (O(n log n))
   Compared against full-N DiHiTI ground truth at N=2^16."""
import numpy as np, timeit
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
nb=1.7; L=1.0; A=nb**2-1
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0); return e-np.linspace(e[0],e[-1],len(x))
def cascade(u,x):
    """necessary seam conditions, O(N), no transform"""
    L=0.5*(x[-1]-x[0]); g=-np.gradient(u,x)
    slope=g[0]-g[-1]
    m=np.abs(x)<0.98*L
    M1=np.trapezoid((g*np.tan(np.pi*x/(2*L)))[m],x[m])
    M2=np.trapezoid((g/np.cos(np.pi*x/(2*L))**2)[m],x[m])
    sig=np.trapezoid(g*np.log(2*np.cos(np.pi*x/(2*L))+1e-300),x)
    return slope,M1,M2,sig
def split_verdict(u,x,n=512,frac=0.95):
    s=len(x)//n; xd=x[::s]; ud=u[::s]
    vd=dihiti(ud,xd); md=np.abs(xd)<=frac*L
    return vd[md].min()
def truth(u,x,frac=0.95):
    v=dihiti(u,x); m=np.abs(x)<=frac*L; return v[m].min(), v[1:-1].min()

N=2**16; x=np.linspace(-L,L,N)
cases={}
for w in [0.3,0.15,0.05,0.02,0.005]: cases[f"logistic w={w}"]=1+A/(1+np.exp(x/w))
for al in [0.0,0.3,0.6,0.8,0.9,1.0,1.5]:
    cases[f"bumps a={al}"]=1+A/(1+np.exp(x/0.05))+al*A*(np.exp(-((x+0.45)/0.22)**2)-np.exp(-((x-0.45)/0.22)**2))
cases["triangle"]=1+2.0*(np.sqrt(L**2+0.15**2)-np.sqrt(x**2+0.15**2))
rng=np.random.default_rng(7)
for i in range(6):
    g=sum(rng.uniform(.1,1)*np.exp(-((x-rng.uniform(-.9,.9))/rng.uniform(.02,.3))**2) for _ in range(rng.integers(3,7)))
    G=cumulative_trapezoid(g,x,initial=0); u=1+G[-1]-G; cases[f"rand{i}"]=1+(u-1)*A/(u[0]-u[-1])
# adversarial cosine dressings at/near decimation Nyquist for n=512
for m_ in [255,256,257,512,1024]:
    cases[f"cos m={m_}"]=1+A/(1+np.exp(x/0.05))+0.02*A*np.cos(m_*np.pi*x/L)

print(f"{'case':16s} {'truth95/D':>11s} {'split/D':>11s} {'sign ok':>8s}  {'|slope|/D':>9s} {'M1/D':>9s} {'M2/D':>8s} {'sig/D':>7s}")
nfail=0
for k,u in cases.items():
    D=u[0]-u[-1]
    t95,tfull=truth(u,x); sp=split_verdict(u,x)
    ok = (t95>=0)==(sp>=0)
    sl,M1,M2,sg=cascade(u,x)
    if not ok: nfail+=1
    print(f"{k:16s} {t95/D:+11.3e} {sp/D:+11.3e} {'ok' if ok else 'MISS':>8s}  {abs(sl)/D:9.2e} {M1/D:+9.2e} {M2/D:+8.2e} {sg/D:+7.3f}")
print(f"\nsign agreement (interior, 0.95L): {len(cases)-nfail}/{len(cases)}")

Nb=2**14; xb=np.linspace(-L,L,Nb); ub=1+A/(1+np.exp(xb/0.05))
t0=min(timeit.repeat(lambda: dihiti(ub,xb).min(),number=20,repeat=5))/20
t1=min(timeit.repeat(lambda: (cascade(ub,xb), split_verdict(ub,xb,n=512)),number=20,repeat=5))/20
print(f"\nN=2^14 wall clock: baseline dihiti+min {t0*1e6:.1f}us ; split test (cascade + decim n=512) {t1*1e6:.1f}us ; speedup {t0/t1:.2f}x")
