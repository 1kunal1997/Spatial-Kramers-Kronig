"""R vs angle for the Lorentzian sKK terminator (reflectionless orientation s=-1),
compared against the PML baseline."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = HERE

skk = np.load(os.path.join(HERE, 'skk_scan_v1.npz'))
pml = np.load(os.path.join(HERE, 'pml_baseline_sweep.npz'))

fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.semilogy(pml['theta'], pml['R'], 'k-', lw=2, label='PML (PMLfactor=1)')

colors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
for c, A in zip(colors, [1, 2, 5, 10]):
    d = skk[f's-1_A{A}']
    ax.semilogy(d[0], d[1], '-', color=c, lw=1.6, label=rf'sKK Lorentzian, $A$={A}')

ax.set_xlabel(r'$\theta$ (deg)')
ax.set_ylabel(r'Reflectance  $|S_{11}|^2$')
ax.set_xlim(0, 89); ax.set_ylim(1e-12, 2)
ax.grid(True, which='major', alpha=0.3)
ax.legend(fontsize=8, loc='lower right')
ax.set_title(r'Lorentzian sKK terminator vs PML   ($L=2\lambda$, $\gamma=L/20$)', fontsize=10)
fig.tight_layout()
p = os.path.join(OUT, 'R_vs_angle_lorentzian_sKK.png')
fig.savefig(p, dpi=200)
print('saved', p)
for A in [1, 2, 5, 10]:
    d = skk[f's-1_A{A}']
    print(f'  A={A:<3g} R(0)={d[1][0]:.3e}  min={d[1].min():.3e} @ {d[0][int(np.argmin(d[1]))]:.0f} deg  R(89)={d[1][-1]:.3e}')
print(f'  PML     R(0)={pml["R"][0]:.3e}  min={pml["R"].min():.3e} @ {pml["theta"][int(np.argmin(pml["R"]))]:.0f} deg  R(89)={pml["R"][-1]:.3e}')
