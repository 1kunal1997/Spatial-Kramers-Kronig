"""Figure for exp_stacked: R vs sKK share of a fixed 10 lam footprint, one figure per angle."""
import numpy as np, skk_lib as S
from figstyle import newax, save

res, meta = S.load_result('stacked')
SP = meta['splits_L_skk']; LT = meta['L_total']
for ai, th in enumerate(meta['angles']):
    fig, ax = newax(xl=r'sKK share of the footprint  $L_{sKK}/L_{tot}$', yl='R', fs=(7.6, 5.2))
    for case, pf, col, ls, lbl in [('lossy+PML', 1, '#1f77b4', '-', 'sKK | lossy slab + PML pf=1 | PEC'),
                                   ('lossy+PML', 4, '#1f77b4', '--', 'sKK | lossy slab + PML pf=4 | PEC'),
                                   ('lossy', 0, '#8B0000', '-', 'sKK | lossy slab, no PML | PEC'),
                                   ('air+PML', 4, '#ff7f0e', '--', 'sKK | AIR + PML pf=4 | PEC (eps\'\' jump)')]:
        xs = [Ls/LT for Ls in SP]; ys = [res[S.key(case=case, Lskk=Ls, pf=pf)][1][ai] for Ls in SP]
        ax.semilogy(xs, ys, ls, color=col, lw=2.2, marker='o', ms=6, label=lbl)
    ax.semilogy([1.0], [res[S.key(case='skk_alone', Lskk=LT)][1][ai]], 'o', color='#2ca02c', ms=9,
                label='sKK alone, 10 lam')
    for pf, mk in ((1, 's'), (4, 'D')):
        ax.semilogy([0.0], [res[S.key(case='pml_alone', pf=pf)][1][ai]], mk, color='k', ms=8,
                    label=f'PML alone pf={pf}, 10 lam')
    ax.set_xlim(-0.05, 1.05); ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title(rf'sKK in front of a PML, 10$\lambda$ footprint, $\theta={th:g}^\circ$', fontsize=12.5, weight='bold')
    save(fig, f'stacked_theta{th:g}.png',
         'Total footprint 10 lam and element size lam/40 on every point, so all points cost the same. '
         'Turned logistic (w = lam/5, eps\'\' plateau 1) in air. "Lossy slab" = eps 1 + 1i, i.e. the medium '
         'the turned logistic ends in, so a PML on it stretches the SAME medium. x = 0: PML alone; '
         'x = 1: sKK alone.')
