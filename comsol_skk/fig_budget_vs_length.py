"""Figures for exp_budget_vs_length: R vs L/lam at fixed N, one figure per (angle, width
rule, profile). Colour = element budget N; solid = sKK profile, dashed = PML pf=4,
dotted = PML pf=1 -- all at the SAME N and the SAME thickness."""
import numpy as np, skk_lib as S
from figstyle import newax, save, LBL

res, meta = S.load_result('budget_vs_length')
NC = {100: '#1f77b4', 200: '#ff7f0e', 400: '#2ca02c'}
LS = meta['L_over_lam']; NS = meta['N']; ANG = meta['angles']
RULE = meta['width_rules']

def curve(kind, N, ai, rule=None, pf=None):
    out = []
    for L in LS:
        k = S.key(kind=kind, rule=rule, L=L, N=N) if pf is None else S.key(kind='PML', pf=pf, L=L, N=N)
        out.append(res[k][1][ai] if k in res else np.nan)
    return np.array(out)

for ai, th in enumerate(ANG):
    for rule in RULE:
        for kind in meta['kinds']:
            fig, ax = newax(xl=r'terminator thickness $L/\lambda$', yl='R', fs=(7.6, 5.2))
            for N in NS:
                ax.loglog(LS, curve(kind, N, ai, rule), '-', color=NC[N], lw=2.3, marker='o', ms=4,
                          label=f'{LBL[kind]}, N={N}')
            for N in NS:
                ax.loglog(LS, curve('PML', N, ai, pf=4), '--', color=NC[N], lw=1.6, label=f'PML pf=4, N={N}')
                ax.loglog(LS, curve('PML', N, ai, pf=1), ':', color=NC[N], lw=1.6, label=f'PML pf=1, N={N}')
            ax.axvspan(0.2, 2, color='0.9', zorder=0)
            ax.text(0.63, 1e-13*3, "paper's $d/\\lambda$ window", fontsize=8, color='0.4', ha='center')
            ax.set_ylim(1e-13, 1.5); ax.set_xlim(0.15, 70)
            ax.legend(fontsize=7.5, loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax.set_title(rf'Fixed budget, {LBL[kind]}, {RULE[rule]}, $\theta={th:g}^\circ$',
                         fontsize=12, weight='bold')
            save(fig, f'budget_vs_length_{rule}_{kind}_theta{th:g}.png',
                 f'Element count N held fixed along each curve while the terminator thickness L varies, '
                 f'so elements per lam fall as 1/L. {RULE[rule]}. PML curves are at the SAME N and the '
                 f'SAME L (swap model: one domain). Air background for the sKK profile and the PML. '
                 f'Grey band = the paper\'s optimal d/lam window, which was for front-face R with a '
                 f'substrate behind, not a PEC. NaN/gaps = solve did not converge.')
