"""How does the three-regime ratio(d/lambda) change when discretization is
converged? Compare paper delta=0.01 vs fine delta=0.00125 at normal incidence.
Throwaway."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
import tmm_helper as tmm_h

nb = 1.7; T = 5.0; k = 40.0 / T
dx = 1 / (100 * k); xmin = -20 / k; xmax = -xmin
nx = 1 + int(np.floor((xmax - xmin) / dx))
xx = np.linspace(xmin, xmax, nx)
e_re = tmm_h.logistic(xx, k, nb)
e_im = tmm_h.ht_derivative(xx, e_re)
ee = e_re + 1j * e_im

dlam = np.logspace(np.log10(0.1), np.log10(100), 80)
fig, (axR, axRatio) = plt.subplots(1, 2, figsize=(13, 5))

for delta, style in [(0.01, '--'), (0.00125, '-')]:
    nc_s, dc_s = tmm_h.discretize_profile(xx, ee, delta=delta)
    nc_g, dc_g = tmm_h.discretize_profile(xx, e_re + 0j, delta=delta)
    nts = [nb] + list(nc_s) + [1.0]; dts = [np.inf] + list(dc_s) + [np.inf]
    ntg = [nb] + list(nc_g) + [1.0]; dtg = [np.inf] + list(dc_g) + [np.inf]
    Rsk, Rgr = [], []
    for r in dlam:
        lam = T / r
        _, Rs, _ = tmm_h.TRA(nts, dts, lamb=lam, angle=0.0, pol='s')
        _, Rg, _ = tmm_h.TRA(ntg, dtg, lamb=lam, angle=0.0, pol='s')
        Rsk.append(Rs); Rgr.append(Rg)
    Rsk, Rgr = np.array(Rsk), np.array(Rgr)
    lab = f'delta={delta} ({len(nc_s)} lay)'
    axR.loglog(dlam, Rsk, style, color='C2', label=f'sKK {lab}')
    axR.loglog(dlam, Rgr, style, color='C1', label=f'GRIN {lab}')
    axRatio.loglog(dlam, Rgr / np.clip(Rsk, 1e-18, None), style, color='C0', label=lab)

axR.set_xlabel(r'$d/\lambda$'); axR.set_ylabel('R (normal incidence, s)')
axR.legend(fontsize=8); axR.set_title('Absolute R: sKK (green) & GRIN (orange)')
axRatio.set_xlabel(r'$d/\lambda$'); axRatio.set_ylabel(r'$R_{GRIN}/R_{sKK}$')
axRatio.legend(fontsize=9); axRatio.set_title('Ratio')
out = os.path.join(_PROJECT_ROOT, 'theory', '_mockup_converged_ratio.png')
fig.tight_layout(); fig.savefig(out, dpi=150); print('saved', out)
