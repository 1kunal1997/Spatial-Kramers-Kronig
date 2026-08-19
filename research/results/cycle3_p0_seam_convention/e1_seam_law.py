"""E1 -- reproduce C39 and test the EDGE EXPANSION closed forms (T1).

Prediction (derived, not fitted):
    eps''(-L+delta) = (1/pi)[ -M1*e + (M2/2)e^2 + (M3/6)e^3 ... ],  e = pi delta/(2L)
  => for M1 > 0 the gain sits near x = -L at
        delta*/L = (2/pi) * M1/M2      (LINEAR in M1)
        depth     = -M1^2/(2 pi M2)    (QUADRATIC in M1)
Everything below uses the EXACT closed-form kernel: no FFT, no grid, no quadrature.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import minimize_scalar
from seamlib import exact_kernel, M1_atoms, M2_atoms, M3_atoms

L = 1.0
def corner_pair(a, b):
    wm, wp = np.tan(np.pi*b/2), np.tan(np.pi*a/2); s = wm + wp
    return [(-a, wm/s), (b, wp/s)]

base = corner_pair(0.25, 0.25)          # a+b = 0.5L, deep inside the safe region

print("="*94)
print("E1a  SEAM MIGRATION: red-team's C39 family, vs the closed-form edge expansion")
print("="*94)
print(f"{'M1':>9} {'x* measured':>13} {'x* predicted':>13} {'rel err':>9} "
      f"{'depth meas':>13} {'depth pred':>13} {'rel err':>9} {'min|x|<0.95L':>14}")
rows = []
for d in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    at = base + [(0.25, d/np.tan(np.pi*0.25/2))]
    M1, M2 = M1_atoms(at), M2_atoms(at)
    # measured: continuous minimisation of the exact kernel near x=-L (no grid at all)
    d_pred = (2/np.pi) * M1 / M2
    f = lambda xx: exact_kernel(np.array([xx]), at, L)[0]
    r = minimize_scalar(f, bracket=(-1+0.02*d_pred, -1+d_pred, -1+8*d_pred), tol=1e-14)
    xstar, depth = r.x, r.fun
    xp, dp = -1.0 + d_pred, -M1**2/(2*np.pi*M2)
    # 0.95L mask on a fine grid
    xg = np.linspace(-0.95, 0.95, 200001)
    m95 = exact_kernel(xg, at, L)
    m95 = float(np.min(m95[np.min(np.abs(xg[:,None]-np.array([t for t,_ in at])[None,:]),axis=1) > 2e-3]))
    print(f"{M1:9.1e} {xstar:13.6f} {xp:13.6f} {abs(xstar-xp)/abs(1+xp):9.2e} "
          f"{depth:13.4e} {dp:13.4e} {abs(depth-dp)/abs(dp):9.2e} {m95:14.4e}")
    rows.append(dict(M1=M1, M2=M2, xstar=xstar, xpred=xp, depth=depth, dpred=dp, mask95=m95))

print("\nPOWER LAWS (log-log slopes, consecutive decades):")
M1s = np.array([r['M1'] for r in rows]); ds = np.array([1+r['xstar'] for r in rows])
dep = np.array([-r['depth'] for r in rows])
print("  distance-to-seam (1+x*) vs M1 :", np.round(np.diff(np.log(ds))/np.diff(np.log(M1s)), 5))
print("  gain depth        vs M1       :", np.round(np.diff(np.log(dep))/np.diff(np.log(M1s)), 5))
print("  gain depth        vs distance :", np.round(np.diff(np.log(dep))/np.diff(np.log(ds)), 5))
print(f"\n  depth / (pi*M2/8 * dist^2) = "
      f"{np.round(dep/((np.pi*np.array([r['M2'] for r in rows])/8)*ds**2), 6)}")

print("\n" + "="*94)
print("E1b  IS THE TRAP BENIGN?  max gain a mask at fraction rho can hide")
print("="*94)
M2b = M2_atoms(base)
print(f"  base M2 = {M2b:.4f};  interior margin of the balanced base profile = "
      f"{exact_kernel(np.linspace(-0.9,0.9,20001), base, L).min():+.4e}")
print(f"  {'rho':>8} {'max hidden gain = (pi M2/8)(1-rho)^2':>40}")
for rho in (0.90, 0.95, 0.99, 0.999, 0.9999):
    print(f"  {rho:8.4f} {np.pi*M2b/8*(1-rho)**2:40.3e}")
