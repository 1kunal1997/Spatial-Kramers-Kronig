"""Figures for exp_discrete_hilbert: continuous pair sampled at the nodes vs DHT of the
node sequence, R vs N, one figure per angle."""
import numpy as np, skk_lib as S
from figstyle import newax, save, C, LBL

res, meta = S.load_result('discrete_hilbert')
MS = {'continuous': ('-', 'o'), 'dht': ('--', 's')}
for ai, th in enumerate(meta['angles']):
    fig, ax = newax(xl='elements across the terminator (table on the nodes)', yl='R', fs=(7.4, 5.2))
    for kind in meta['kinds']:
        for mode, (ls, mk) in MS.items():
            ys = [res[S.key(kind=kind, N=N, mode=mode)][1][ai] for N in meta['N']]
            ax.loglog(meta['N'], ys, ls, color=C[kind], lw=2.2, marker=mk, ms=6,
                      label=f'{LBL[kind]}, {mode} HT')
    ax.legend(fontsize=9, loc='best')
    ax.set_title(rf'Continuous vs discrete Hilbert partner, $\theta={th:g}^\circ$', fontsize=12.5, weight='bold')
    fom = '; '.join(f"{g['kind']} N={g['N']} {g['mode']}: {g['one_sidedness']:.3f}%" for g in meta['grid'])
    save(fig, f'discrete_hilbert_theta{th:g}.png',
         'L = 10 lam, w = lam/5. The profile table sits ON the N+1 mesh nodes in both modes, so COMSOL sees '
         'the same piecewise-linear interpolant class; only the Hilbert partner differs. Logistic in n_b = 1.7, '
         'turned logistic in air. One-sidedness of the node sequence: ' + fom)
