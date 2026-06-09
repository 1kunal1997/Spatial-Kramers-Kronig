"""FU2: is the high-thickness TMM failure 'too much absorption', or T underflow /
matrix overflow? A saturates at 1, but T ~ exp(-c*d/lambda) keeps shrinking.
Throwaway."""
import sys, os, warnings
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import tmm_helper as tmm_h
warnings.filterwarnings('error')  # turn numpy warnings into catchable errors

nb = 1.7; T = 5.0; k = 40.0 / T
dx = 1/(100*k); xx = np.linspace(-20/k, 20/k, 1 + int(np.floor((40/k)/dx)))
e_re = tmm_h.logistic(xx, k, nb)
ee = e_re + 1j * tmm_h.ht_derivative(xx, e_re)
nc, dc = tmm_h.discretize_profile(xx, ee, delta=0.01)
n_t = [nb] + list(nc) + [1.0]; d_t = [np.inf] + list(dc) + [np.inf]

print(f"{'d/lam':>7} {'T':>12} {'R':>11} {'A':>9} {'-ln(T)':>9}  note")
for r in (10, 50, 100, 200, 350, 500, 750, 1000):
    lam = T / r                      # equivalent to coating thickness 5*r um at lam=5
    try:
        Tt, Rt, At = tmm_h.TRA(n_t, d_t, lamb=lam, angle=np.radians(45)/nb, pol='p')
        note = ''
        if np.isnan(Tt) or np.isnan(Rt): note = 'NaN!'
        lnT = -np.log(Tt) if Tt > 0 else np.inf
        print(f"{r:>7d} {Tt:>12.3e} {Rt:>11.3e} {At:>9.4f} {lnT:>9.1f}  {note}")
    except Exception as exc:
        print(f"{r:>7d} {'ERROR':>12} {'':>11} {'':>9} {'':>9}  {type(exc).__name__}: {str(exc)[:50]}")
