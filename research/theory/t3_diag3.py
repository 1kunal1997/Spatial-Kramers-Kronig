import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
rng = np.random.default_rng(7); samples=list(pg.sample(240,rng=rng))

def logln(eta,xg,dxi):
    d=np.abs(eta-xg); out=np.empty_like(d); bad=d<0.5*dxi
    out[~bad]=np.log(d[~bad]); out[bad]=np.log(np.maximum(dxi[bad],1e-300)/2.0)-1.0
    return out
def gap(g,t,eta,L):
    xg=np.tan(np.pi*t/(2*L)); dxi=np.gradient(xg); D=np.trapezoid(g,t)
    eta=np.atleast_1d(np.asarray(eta,float))
    out=np.array([np.trapezoid(logln(e,xg,dxi)*g,t) for e in eta])
    return out-0.5*D*np.log(1+eta**2)

for i in [46,72,170,41]:
    s=samples[i]; x=s['x']; L=x[-1]; ei=s['e_im']; g=-np.gradient(s['e_re'],x)
    mm=np.abs(x)<0.97*L
    dl=gap(g,x,np.tan(np.pi*x[mm]/(2*L)),L)
    eps_dom = -dl/np.pi
    err = eps_dom - ei[mm]
    print(f"{i} {s['name']}: max|err|={np.abs(err).max():.4e} scale={ei.max():.3f} rel={np.abs(err).max()/ei.max():.2e}")
    print(f"    err at argmin region: ei_min={ei.min():+.5f}  dom_min={eps_dom.min():+.5f}")
    # err profile
    xm=x[mm]
    for q in [0.0,0.5,0.8,0.89,0.95]:
        k=np.argmin(np.abs(xm/L-q)); print(f"      x/L={xm[k]/L:+.3f}: dihiti={ei[mm][k]:+.5f} dom={eps_dom[k]:+.5f} err={err[k]:+.5f}")
    # edge slopes
    print(f"    g(-L)={g[0]:.4f} g(+L)={g[-1]:.4f}  D={np.trapezoid(g,x):.4f}")
