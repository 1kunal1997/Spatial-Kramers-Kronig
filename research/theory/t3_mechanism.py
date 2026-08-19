"""T3 mechanism tests: is C28's stated diagnosis right?"""
import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
from t3_clausen import gap_clausen

# ---- the OLD (cycle-1) evaluator, verbatim from h9_fixes.py -------------
def logln(eta,xg,dxi):
    d=np.abs(eta-xg); out=np.empty_like(d); bad=d<0.5*dxi
    out[~bad]=np.log(d[~bad]); out[bad]=np.log(np.maximum(dxi[bad],1e-300)/2.0)-1.0
    return out
def gap_old(g,t,eta,L):
    xg=np.tan(np.pi*t/(2*L)); dxi=np.gradient(xg); D=np.trapezoid(g,t)
    eta=np.atleast_1d(np.asarray(eta,float))
    out=np.array([np.trapezoid(logln(e,xg,dxi)*g,t) for e in eta])
    return out-0.5*D*np.log(1+eta**2)

rng=np.random.default_rng(7); S=list(pg.sample(240,rng=rng))
IDX=[41,46,72,89,162,170,199]

print("="*78)
print("TEST 1  — does the OLD disagreement vanish as the truncation frac -> 1 ?")
print("          (C28 says the 0.97L cut removes the near-seam minimum)")
for frac in [0.90,0.95,0.97,0.99,0.999,0.9999]:
    ag=0
    for s in S:
        x=s['x']; L=x[-1]; g=-np.gradient(s['e_re'],x); mm=np.abs(x)<frac*L
        dl=gap_old(g,x,np.tan(np.pi*x[mm]/(2*L)),L)
        sc=max(s['e_im'].max(),1e-12)
        if (-dl.max()/np.pi >= -1e-3*sc)==s['passive']: ag+=1
    print(f"   frac={frac:<8} agreement {ag}/240")

print()
print("="*78)
print("TEST 2  — where does the DiHiTI minimum actually live? (C28 says near the seam)")
for i in IDX:
    s=S[i]; x=s['x']; L=x[-1]; ei=s['e_im']
    print(f"   {i:4d} {s['name']:15s} argmin at |x|/L = {abs(x[np.argmin(ei)])/L:.4f}")

print()
print("="*78)
print("TEST 3  — is the OLD error an eta-independent CONSTANT offset?")
for i in IDX:
    s=S[i]; x=s['x']; L=x[-1]; g=-np.gradient(s['e_re'],x); mm=np.abs(x)<0.97*L
    old=-gap_old(g,x,np.tan(np.pi*x[mm]/(2*L)),L)/np.pi
    e=old-s['e_im'][mm]
    print(f"   {i:4d} {s['name']:15s} err: mean {e.mean():+.6f}  std {e.std():.2e}  std/|mean| {e.std()/abs(e.mean()):.1e}")

print()
print("="*78)
print("TEST 4  — is the offset EXACTLY the quadrature error of sigma = int g ln cos(pi t/2L) dt?")
print("          prediction: err_const = -(sigma_trapz - sigma_exact)/pi")
from scipy.special import polygamma
def sigma_exact(e_re,x):
    """int g ln cos(pi t/2L) dt with the log kernel integrated exactly per cell (Clausen)."""
    from t3_clausen import _clausen_grid
    N=len(x); M=N-1; L=0.5*(x[-1]-x[0]); h=(x[-1]-x[0])/M
    cl=_clausen_grid(M); kk=np.arange(0,M+1)
    W=-cl[np.mod(kk,M)]; A=(L/np.pi)*(W-(2*np.pi*kk/M)*np.log(2.0))
    C=A[1:]-A[:-1]                       # C[j] = int_{cell j} ln cos(pi t/2L) dt
    w=e_re[:-1]-e_re[1:]
    return np.dot(w/h, C)
for i in IDX:
    s=S[i]; x=s['x']; L=x[-1]; g=-np.gradient(s['e_re'],x); mm=np.abs(x)<0.97*L
    old=-gap_old(g,x,np.tan(np.pi*x[mm]/(2*L)),L)/np.pi
    err=(old-s['e_im'][mm]).mean()
    lc=np.log(np.cos(np.pi*x/(2*L)))     # -inf at endpoints -> clip as np would see it
    with np.errstate(divide='ignore'):
        lc=np.log(np.maximum(np.cos(np.pi*x/(2*L)),1e-300))
    sig_tr=np.trapezoid(g*lc,x); sig_ex=sigma_exact(s['e_re'],x)
    pred=-(sig_tr-sig_ex)/np.pi
    print(f"   {i:4d} {s['name']:15s} err_obs {err:+.6f}   pred {pred:+.6f}   ratio {err/pred if pred!=0 else np.nan:.3f}")

print()
print("="*78)
print("TEST 5  — does the OLD error vanish like h*ln(1/h) under grid refinement?")
for i in [46,41]:
    s0=S[i]
    print(f"   profile {i} {s0['name']} {s0['params']}")
    prev=None
    for N in [1024,2048,4096,8192,16384]:
        s=pg.make(s0['name'],N=N,**s0['params'])
        x=s['x']; L=x[-1]; g=-np.gradient(s['e_re'],x); mm=np.abs(x)<0.97*L
        old=-gap_old(g,x,np.tan(np.pi*x[mm]/(2*L)),L)/np.pi
        e=(old-s['e_im'][mm]).mean()
        newd,_=gap_clausen(s['e_re'],x)
        en=np.abs(newd-s['e_im']).max()
        h=2*L/(N-1); hlogh=h*np.log(1/h)
        print(f"      N={N:6d}  old_offset {e:+.3e}  old/(h ln 1/h) {e/hlogh:+.4f}   new max|err| {en:.2e}")

print()
print("="*78)
print("TEST 6  — is the offset proportional to the EDGE VALUES g(+-L)?  (C28 says it is")
print("          about edge-slope MISMATCH g(+L)!=g(-L); the sigma mechanism says it is")
print("          about the SUM g(+L)+g(-L), i.e. mismatch is irrelevant.)")
for i in IDX:
    s=S[i]; x=s['x']; L=x[-1]; g=-np.gradient(s['e_re'],x); mm=np.abs(x)<0.97*L
    old=-gap_old(g,x,np.tan(np.pi*x[mm]/(2*L)),L)/np.pi
    e=(old-s['e_im'][mm]).mean()
    print(f"   {i:4d} {s['name']:15s} g(-L)={g[0]:+.4f} g(+L)={g[-1]:+.4f}  sum={g[0]+g[-1]:+.4f}"
          f"  diff={g[-1]-g[0]:+.4f}  offset={e:+.6f}  offset/sum={e/(g[0]+g[-1]):+.5f}")
