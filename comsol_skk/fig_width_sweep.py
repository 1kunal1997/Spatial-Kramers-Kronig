"""Figures for exp_width_sweep: R vs each shape parameter at fixed L = 10 lam, N = 400.
One figure per swept parameter; colour = profile, linestyle = angle."""
import numpy as np, skk_lib as S
from figstyle import newax, save, C, LBL

res, meta = S.load_result('width_sweep')
AS = {0: '-', 80: '--', 89: ':'}
XL = {'w': r'feature width $w/\lambda$', 'ell': r'taper length $\ell/\lambda$',
      'A': 'Lorentzian amplitude $A$', 'amp': r"turned-logistic $\epsilon''$ plateau"}
for sweep, xl in XL.items():
    rows = [g for g in meta['grid'] if g['sweep'] == sweep]
    kinds = sorted({g['kind'] for g in rows}, key=list(LBL).index)
    fig, ax = newax(xl=xl, yl='R', fs=(7.4, 5.2))
    for kind in kinds:
        xs = [g['val'] for g in rows if g['kind'] == kind]
        for ai, th in enumerate(meta['angles']):
            ys = [res[g['key']][1][ai] for g in rows if g['kind'] == kind]
            ax.loglog(xs, ys, AS[th], color=C[kind], lw=2.2, marker='o', ms=5,
                      label=f'{LBL[kind]}, ' + rf'$\theta={th}^\circ$')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title(f'Shape sweep: {xl}', fontsize=12.5, weight='bold')
    fom = '; '.join(f"{g['kind']} {g['val']:g}: {g['one_sidedness']:.2f}%" for g in rows)
    save(fig, f'width_sweep_{sweep}.png',
         'L = 10 lam, N = 400 (resolved), air background. One-sidedness FoM of each profile: ' + fom)
