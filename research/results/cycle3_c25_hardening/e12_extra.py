"""E12 -- two free contributions found while attacking C25.

(A) The condition depends only on the EFFECTIVE reach, and non-interval supports give nothing
    new: by the same Bauer/<=2-atom reduction, for ANY support set S the family
    {g>=0, M1=0, supp g in S} is uniformly passive iff  a+b <= L  with
        a = sup{-t : t in S, t<0},   b = sup{t : t in S, t>0}.
    i.e. only the interval hull of S matters.  (Stated; the reduction is the same as C25's.)

(B) Threshold depth law for the extremal corner pair: how does min eps'' grow just past L?
"""
import numpy as np
from common import *
L=1.0; xg = periodic_grid(1<<19, L)
def cp(a,b):
    wm,wp=np.tan(np.pi*b/2),np.tan(np.pi*a/2); s=wm+wp; return [(-a,wm/s),(b,wp/s)]
print("  base (a0,b0) with a0+b0=L; then b = b0 + d.   min eps'' / d  ->  ?")
print(f"{'a0':>6} {'b0':>6} | " + " ".join(f"{d:>11.0e}" for d in (1e-2,1e-3,1e-4,1e-5)))
for (a0,b0) in [(0.5,0.5),(0.25,0.75),(0.1,0.9),(0.4,0.6),(0.75,0.25),(0.05,0.95)]:
    row=[]
    for d in (1e-2,1e-3,1e-4,1e-5):
        at = cp(a0, b0+d)
        y = exact_kernel(xg, at, L)
        m,_ = interior_min(xg,y,L,0.99999,[t for t,_ in at],2e-3)
        row.append(m/d)
    print(f"{a0:6.2f} {b0:6.2f} | " + " ".join(f"{v:11.5f}" for v in row))
print("\n  -> min eps'' ~ -C(a0,b0) * (a+b-L).  Compare the theoretical slope from Cor 4.A:")
print("     Delta(eta*) = gamma ln(sinB/cosA) + beta ln(sinA/cosB), min eps'' = -Delta(eta*)/(pi(beta+gamma))")
for (a0,b0) in [(0.5,0.5),(0.25,0.75),(0.1,0.9),(0.05,0.95)]:
    A0,B0 = np.pi*b0/2, np.pi*a0/2
    # d/db of Delta at A+B=pi/2, normalised
    dd=1e-7; A=np.pi*(b0+dd)/2
    be,ga = np.tan(A), np.tan(B0)
    D = ga*np.log(np.sin(B0)/np.cos(A)) + be*np.log(np.sin(A)/np.cos(B0))
    print(f"     a0={a0:.2f} b0={b0:.2f}: analytic slope = {-D/(np.pi*(be+ga))/dd:10.5f}")
