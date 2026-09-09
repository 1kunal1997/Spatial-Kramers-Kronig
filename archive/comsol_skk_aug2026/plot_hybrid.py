import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
h = np.load(os.path.join(HERE, 'skk_hybrid_v1.npz'))

fig, ax = plt.subplots(figsize=(6.8, 5.0))
style = [('PML_pf1_D0',      'k-',  2.2, 'PML alone (control, $D$=0)'),
         ('noPML_pf1_D3.6',  '-',   1.8, 'sKK alone ($D$=3.6)'),
         ('PML_pf1_D1',      '-',   1.5, 'hybrid, PMLfactor=1, $D$=1'),
         ('PML_pf1_D3.6',    '-',   1.5, 'hybrid, PMLfactor=1, $D$=3.6'),
         ('PML_pf1_D8',      '-',   1.5, 'hybrid, PMLfactor=1, $D$=8'),
         ('PML_pf4_D3.6',    '-',   1.8, 'hybrid, PMLfactor=4, $D$=3.6')]
cols = ['k', '#d62728', '#9ecae1', '#4292c6', '#08519c', '#2ca02c']
for (k, ls, lw, lab), c in zip(style, cols):
    d = h[k]
    ax.semilogy(d[0], d[1], ls if ls != '-' else '-', color=c, lw=lw, label=lab)
ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel(r'Reflectance $|S_{11}|^2$')
ax.set_xlim(0, 89); ax.set_ylim(1e-12, 2); ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower right')
ax.set_title('Hybrid PML + sKK material: the two mechanisms are incompatible', fontsize=10)
fig.tight_layout()
p = os.path.join(HERE, 'R_vs_angle_hybrid.png'); fig.savefig(p, dpi=200)
print('saved', p, '\n')
print(f'{"config":22s} {"R(0)":>11} {"R(60)":>11} {"R(89)":>9}   vs PML alone')
base = h['PML_pf1_D0']
for k, *_ in style:
    d = h[k]
    i60 = int(np.argmin(abs(d[0]-60)))
    print(f'{k:22s} {d[1][0]:>11.3e} {d[1][i60]:>11.3e} {d[1][-1]:>9.3f}   '
          f'{d[1][0]/base[1][0]:>10.1f}x worse at 0 deg')
