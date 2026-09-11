"""Figures for exp_mesh_grading: R vs centre/tail element ratio r, one figure per
(profile, feature width, angle). Colour = N; solid = centre-dense, dashed = entry-dense;
the r = 1 point is the uniform mesh. Vertical line = predicted r* = 0.4 lam/(n w)."""
import numpy as np, skk_lib as S
from figstyle import newax, save, LBL

res, meta = S.load_result('mesh_grading')
NC = {100: '#1f77b4', 200: '#ff7f0e', 400: '#2ca02c'}
RS = [1] + meta['ratios']; ANG = meta['angles']

for kind in meta['kinds']:
    for wtag, wl in meta['widths'].items():
        for ai, th in enumerate(ANG):
            fig, ax = newax(xl='element ratio  $r$ = largest / smallest', yl='R', fs=(7.4, 5.2))
            for N in meta['N']:
                for grading, ls in (('centre', '-'), ('entry', '--')):
                    ys = [res[S.key(kind=kind, w=wtag, N=N, grading=('uniform' if r == 1 else grading), r=r)][1][ai]
                          for r in RS]
                    ax.semilogy(RS, ys, ls, color=NC[N], lw=2.2, marker='o', ms=4,
                                label=f'{grading}-dense, N={N}')
            rs = meta['r_star_prediction'][wtag]
            ax.axvline(rs, color='crimson', ls=':', lw=1.5)
            ax.text(rs*1.05, 0.04, 'predicted $r^*$', color='crimson', fontsize=9, transform=ax.get_xaxis_transform())
            ax.set_xscale('log'); ax.set_xlim(0.9, 12)
            ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax.set_title(rf'Mesh grading, {LBL[kind]}, $w=\lambda/{1/wl:g}$, $\theta={th:g}^\circ$',
                         fontsize=12, weight='bold')
            save(fig, f'mesh_grading_{kind}_{wtag}_theta{th:g}.png',
                 f'L = 10 lam, N elements redistributed with a geometric distribution of ratio r; r = 1 is '
                 f'uniform. Solid = fine in the centre (symmetric+reverse), dashed = fine at the entry face '
                 f'(reverse). Red line = r* = 0.4 lam/(n w) from PLAN.md, the ratio at which the profile '
                 f'and wave resolution constraints are balanced. Air background.')
