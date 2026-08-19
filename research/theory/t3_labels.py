"""Coordinator's request: for each of the 7, compare INTERIOR min (DiHiTI) vs corpus label
vs (L1) verdict, under BOTH the old and the fixed evaluator."""
import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0,_ROOT); sys.path.insert(0,os.path.join(_ROOT,'theory'))
sys.path.insert(0,os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
from t3_clausen import gap_clausen

def logln(eta,xg,dxi):
    d=np.abs(eta-xg); out=np.empty_like(d); bad=d<0.5*dxi
    out[~bad]=np.log(d[~bad]); out[bad]=np.log(np.maximum(dxi[bad],1e-300)/2.0)-1.0
    return out
def gap_old(g,t,eta,L):
    xg=np.tan(np.pi*t/(2*L)); dxi=np.gradient(xg); D=np.trapezoid(g,t)
    eta=np.atleast_1d(np.asarray(eta,float))
    return np.array([np.trapezoid(logln(e,xg,dxi)*g,t) for e in eta])-0.5*D*np.log(1+eta**2)

rng=np.random.default_rng(7); S=list(pg.sample(240,rng=rng))
IDX=[41,46,72,89,162,170,199]
print(f"{'i':>4} {'family':16} {'label':6} {'wholemin':>11} {'intmin(DiHiTI)':>15} {'intmin(L1 fixed)':>17} {'L1old':>7} {'L1new':>7}")
for i in IDX:
    s=S[i]; x=s['x']; L=x[-1]; ei=s['e_im']; sc=max(ei.max(),1e-12)
    m=np.abs(x)<0.97*L
    g=-np.gradient(s['e_re'],x)
    old=-gap_old(g,x,np.tan(np.pi*x[m]/(2*L)),L)/np.pi
    new,_=gap_clausen(s['e_re'],x)
    vold = old.min()>=-1e-3*sc
    vnew = new.min()>=-1e-3*sc
    print(f"{i:4d} {s['name']:16} {str(s['passive']):6} {ei.min():11.3e} {ei[m].min():15.3e} {new[m].min():17.3e} "
          f"{str(vold):>7} {str(vnew):>7}")
print()
print("interior-min-based labels (frac=0.97) vs corpus whole-domain labels, all 240:")
dis=0; flips=0
for s in S:
    x=s['x']; L=x[-1]; ei=s['e_im']; sc=max(ei.max(),1e-12); m=np.abs(x)<0.97*L
    lab_whole = ei.min()>=-1e-3*sc
    lab_int   = ei[m].min()>=-1e-3*sc
    new,_=gap_clausen(s['e_re'],x)
    if lab_whole!=lab_int: flips+=1
    if (new[m].min()>=-1e-3*sc)!=lab_int: dis+=1
print(f"  labels that change when using interior min instead of whole-domain min: {flips}/240")
print(f"  (L1 fixed, interior) vs (DiHiTI interior) disagreements: {dis}/240")
