"""Figure for exp_lateral_mesh: does the lateral element count matter at grazing?"""
import numpy as np, skk_lib as S
from figstyle import newax, save, C, LBL

res, meta = S.load_result('lateral_mesh')
ST = {5: '-', 10: '--', 20: ':'}
fig, ax = newax(fs=(7.6, 5.2))
rows = []
for kind in ['turned_logistic', 'tapered', 'PML']:
    col = C.get(kind, 'k'); lbl = LBL.get(kind, 'PML pf=4')
    base = res[S.key(kind=kind, Nx=5)][1]
    for nx in meta['Nx']:
        d = res[S.key(kind=kind, Nx=nx)]
        ax.semilogy(d[0], d[1], ST[nx], color=col, lw=2.2, marker='o', ms=5, label=f'{lbl}, $N_x$={nx}')
        ok = base > 1e-13                      # ignore points at the solver floor
        rows.append((lbl, nx, np.nanmax(np.abs(d[1][ok]/base[ok] - 1))*100))
ax.set_ylim(1e-13, 1.5); ax.set_xlim(-2, 91)
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.set_title(r'Lateral elements across $L_x=\lambda/4$', fontsize=13, weight='bold')
note = 'Max |change| relative to N_x = 5 (points with R > 1e-13 only): ' + '; '.join(f'{l} N_x={n}: {c:.2f}%' for l, n, c in rows if n > 5)
save(fig, 'lateral_mesh.png', 'L = 10 lam, 400 elements along y, air background. ' + note)
print(note)
