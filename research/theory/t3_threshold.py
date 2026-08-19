import sys, os
import numpy as np
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT,'theory'))
sys.path.insert(0, os.path.join(_ROOT,'theory','passivity_2026Aug'))
import profile_generator as pg
from t3_clausen import gap_clausen

def interior_min(v, x, frac=0.999):
    m=np.abs(x)<frac*x[-1]; return v[m].min()

print("Near-threshold: opposite_bumps alpha crossover (C18 says alpha_c = 0.66623)")
def f(alpha, N=8192, which='dihiti'):
    s=pg.make('opposite_bumps', N=N, alpha=alpha, sep=0.9, w=0.45)
    x=s['x']
    v = s['e_im'] if which=='dihiti' else gap_clausen(s['e_re'],x)[0]
    return interior_min(v,x)
for which in ['dihiti','clausen']:
    lo,hi=0.3,1.2
    for _ in range(45):
        mid=0.5*(lo+hi)
        if f(mid,which=which)>=0: lo=mid
        else: hi=mid
    print(f"  {which:8s} alpha_c = {0.5*(lo+hi):.6f}")

print()
print("Near-threshold: balanced two-atom law, bisect S* = t+ + t- (Cor 4.A says S* = L)")
print("(skipped here - covered in cycle 1 block D)")

print()
print("Self-convergence: is the residual the new evaluator or DiHiTI?")
print("  reference = gap_clausen at N=131073 (odd -> M=2^17), restricted")
s0=pg.make('tanh_slopes', N=131073, mL=3.652473832785226, mR=5.091793139905501)
ref_x=s0['x']; ref_dom,_=gap_clausen(s0['e_re'],ref_x); ref_di=s0['e_im']
print(f"  at N=131073: max|clausen - dihiti| = {np.abs(ref_dom-ref_di).max():.3e}")
for N in [2049,4097,8193,16385,32769]:
    s=pg.make('tanh_slopes', N=N, mL=3.652473832785226, mR=5.091793139905501)
    x=s['x']; dom,_=gap_clausen(s['e_re'],x)
    idx=np.searchsorted(ref_x, x); idx=np.clip(idx,0,len(ref_x)-1)
    e_cl=np.abs(dom-ref_dom[idx]).max()          # clausen vs fine clausen
    e_di=np.abs(s['e_im']-ref_di[idx]).max()     # dihiti vs fine dihiti
    print(f"  N={N:7d}  |clausen-ref_cl| {e_cl:.3e}   |dihiti-ref_di| {e_di:.3e}   |cl-di| {np.abs(dom-s['e_im']).max():.3e}")
