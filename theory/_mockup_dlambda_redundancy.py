"""Numeric demo: in a non-dispersive stack, R depends only on d/lambda.
A wavelength sweep (fix d, vary lambda) and a thickness sweep (fix lambda,
scale coating) give identical R at matched d/lambda. Throwaway."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import tmm_helper as tmm_h

nb = 1.7; delta = 0.01; pol = 's'
angle_air = 80.0
ang_nb = np.arcsin(np.sin(np.radians(angle_air)) / nb)


def build_skk(D):
    """Geometrically-scaled sKK coating of total thickness D (um), shape fixed."""
    xx = np.linspace(-D/2, D/2, 5001)
    k = 20.0 / D                      # 4 at D=5 -> preserves shape under scaling
    e_re = (nb**2 - 1) / (1 + np.exp(k * xx)) + 1
    e_im = tmm_h.ht_derivative(xx, e_re)
    return tmm_h.discretize_profile(xx, e_re + 1j * e_im, delta=delta)


def R_of(D, lam):
    nc, dc = build_skk(D)
    n_t = [nb] + list(nc) + [1.0]
    d_t = [np.inf] + list(dc) + [np.inf]
    _, R, _ = tmm_h.TRA(n_t, d_t, lamb=lam, angle=ang_nb, pol=pol)
    return R


print(f"sKK, {pol}-pol, {angle_air:.0f} deg in air\n")
print(f"{'d/lam':>6} | {'wavelength sweep (d=5um)':>30} | {'thickness sweep (lam=5um)':>30}")
print(f"{'':>6} | {'(d,  lam)        R':>30} | {'(d,  lam)        R':>30}")
print("-" * 75)
for ratio in (0.5, 1.0, 2.0):
    lam_A = 5.0 / ratio          # fix d=5, choose lam to hit ratio
    R_A = R_of(5.0, lam_A)
    D_B = 5.0 * ratio            # fix lam=5, choose d to hit ratio
    R_B = R_of(D_B, 5.0)
    print(f"{ratio:>6.2f} | (5.0, {lam_A:5.2f})   {R_A:.3e} | ({D_B:4.1f}, 5.00)   {R_B:.3e}")
