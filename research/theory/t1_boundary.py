"""T1: the boundary of the passive set for unimodal balanced measures.
Extremal configuration (found by dense search): a point mass at ONE END of a uniform arc.
   mu = lam*delta_p + (1-lam)*U[p,q]      (mode at p; unimodal, since density is
                                           an atom at the left edge + a plateau)
Balance fixes lam.  Scan the (p,q) plane for the zero-gain boundary."""
import numpy as np
from t1_fast import cl2, m1_arc, PI
TH = np.linspace(-PI, PI, 4001)

def gain_atom_end(p, q, side='left'):
    if q-p < 1e-9: return -9.0
    c = p if side=='left' else q
    T = np.tan(c/2)
    m = m1_arc((p+q)/2, (q-p)/2)
    if abs(T-m) < 1e-14: return -9.0
    lam = -m/(T-m)
    if not (0 <= lam <= 1): return np.nan
    Phi = lam*np.log(np.abs(2*np.sin((TH-c)/2))+1e-300) \
        + (1-lam)*(cl2(TH-q)-cl2(TH-p))/(q-p)
    Pi  = lam*np.log(np.abs(2*np.sin((PI-c)/2))) \
        + (1-lam)*(cl2(PI-q)-cl2(PI-p))/(q-p)
    return float(Phi.max()-Pi)

print("mu = lam*delta_p + (1-lam)U[p,q]  (atom at LEFT end).  Boundary q*(p):")
print(f"{'p/pi':>8} {'q*/pi':>9} {'span/pi':>9} {'lam*':>8}")
rows=[]
for pf in np.arange(-0.95, 0.0, 0.05):
    p = pf*PI
    lo, hi = p+1e-6, PI-1e-9
    if gain_atom_end(p, hi) <= 0 or np.isnan(gain_atom_end(p,hi)):
        print(f"{pf:8.3f}       ---  (no failure even at q=pi)"); continue
    for _ in range(60):
        mid=0.5*(lo+hi); g=gain_atom_end(p,mid)
        if np.isnan(g) or g<=0: lo=mid
        else: hi=mid
    q=0.5*(lo+hi)
    T=np.tan(p/2); m=m1_arc((p+q)/2,(q-p)/2); lam=-m/(T-m)
    rows.append((p,q,lam))
    print(f"{pf:8.3f} {q/PI:9.4f} {(q-p)/PI:9.4f} {lam:8.4f}")
np.save('/tmp/t1_boundary.npy', np.array(rows))
