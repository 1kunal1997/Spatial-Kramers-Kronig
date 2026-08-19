"""T3 diagnosis: reproduce block C (233/240) and identify the true mechanism."""
import sys, os
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg

def logln(eta, xg, dxi):
    d = np.abs(eta - xg); out = np.empty_like(d); bad = d < 0.5*dxi
    out[~bad] = np.log(d[~bad]); out[bad] = np.log(np.maximum(dxi[bad],1e-300)/2.0) - 1.0
    return out

def gap(g, t, eta, L):
    xg = np.tan(np.pi*t/(2*L)); dxi = np.gradient(xg); D = np.trapezoid(g, t)
    eta = np.atleast_1d(np.asarray(eta,float))
    out = np.array([np.trapezoid(logln(e,xg,dxi)*g, t) for e in eta])
    return out - 0.5*D*np.log(1+eta**2)

rng = np.random.default_rng(7)
samples = list(pg.sample(240, rng=rng))
np.save('/tmp/t3_nsamp.npy', np.array([len(samples)]))

for frac in [0.97]:
    agree=0; dis=[]
    for i,smp in enumerate(samples):
        xs=smp['x']; Ls=xs[-1]; gs=-np.gradient(smp['e_re'],xs)
        mm=np.abs(xs)<frac*Ls
        dl=gap(gs,xs,np.tan(np.pi*xs[mm]/(2*Ls)),Ls)
        scale=max(smp['e_im'].max(),1e-12)
        dom_min=-dl.max()/np.pi
        v=dom_min>=-1e-3*scale
        if v==smp['passive']: agree+=1
        else: dis.append((i,smp['name'],smp['params'],round(smp['min_eim'],6),round(dom_min,6),round(scale,5),smp['passive']))
    print(f"frac={frac}: agreement {agree}/240")
    for d in dis: print("   ",d)
