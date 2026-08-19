import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
nb=1.7; L=1.0; A=nb**2-1
def dihiti(u,x):
    du=np.gradient(u,x); v=np.imag(hilbert(du))
    e=cumulative_trapezoid(v,x,initial=0.0); return e-np.linspace(e[0],e[-1],len(x))

def periodize(u,x):
    L=0.5*(x[-1]-x[0]); D=u[0]-u[-1]; return u+(D/(2*L))*x, D

def tail_bound(u,x,n,k=3):
    """O(N), no FFT: certified bound on sum_{|m|>n/2}|p_hat_m| via ||p^(k)||_2 (finite differences)."""
    p,D=periodize(u,x); N=len(x); h=x[1]-x[0]
    q=p.copy()
    for _ in range(k):                      # periodic k-th finite difference, O(kN)
        q=(np.roll(q,-1)-np.roll(q,1))/(2*h)
    # ||p^(k)||_{L2}^2 = (1/2L) int |p^(k)|^2 = sum |(i m pi/L)^k p_hat_m|^2
    S=np.sqrt(np.mean(q**2))
    ms=np.arange(n//2+1,N//2)
    if len(ms)==0: return 0.0
    W=np.sqrt(2*np.sum((L/(np.pi*ms))**(2*k)))
    return S*W

def decim_verdict(u,x,n):
    N=len(x); s=N//n
    xd=x[::s]; ud=u[::s]
    return dihiti(ud,xd), xd

print(" profile        n     interior-min/D (decimated)   true(N=2^16)/D   |err|/D    certbound/D  usable?")
def report(name,u,x,D):
    vT=dihiti(u,x); m=np.abs(x)<=0.95*L; tru=vT[m].min()/D
    for n in [64,128,256,512,1024]:
        vd,xd=decim_verdict(u,x,n); md=np.abs(xd)<=0.95*L
        est=vd[md].min()/D; tb=tail_bound(u,x,n)/D
        ok = "YES" if tb<abs(tru) else "defer"
        print(f" {name:14s} {n:5d}   {est:+.4e}            {tru:+.4e}   {abs(est-tru):.2e}   {tb:.2e}    {ok}")
N=2**16; x=np.linspace(-L,L,N)
u=1+A/(1+np.exp(x/0.05)); report("logistic .05",u,x,u[0]-u[-1])
u=1+A/(1+np.exp(x/0.005)); report("logistic .005",u,x,u[0]-u[-1])
for al in [0.6,0.9,1.0,1.5]:
    u=1+A/(1+np.exp(x/0.05))+al*A*(np.exp(-((x+0.45)/0.22)**2)-np.exp(-((x-0.45)/0.22)**2))
    report(f"bumps a={al}",u,x,u[0]-u[-1])
