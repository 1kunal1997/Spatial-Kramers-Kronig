"""Figures for exp_angle_vs_terminator: R vs angle, and the profiles that produced it.

Reads data/angle_vs_terminator.npz and the profile files it names -- no COMSOL, no licence.
"""
import os
import numpy as np
import skk_lib as S
from figstyle import newax, save, C, LBL, STY, PC

res, meta = S.load_result('angle_vs_terminator')
KINDS = [k for k in ['logistic', 'lorentzian', 'tapered', 'turned_logistic']
         if f'R_{k}' in res]
NOTE = (f"Identical geometry and mesh on every curve: terminator L = {meta['L_over_lam']:g} lam "
        f"with {meta['N_term']} elements, background n_b = {meta['n_bg']}, PEC backing, s-pol. "
        "The PML occupies the SAME domain as the sKK profiles and therefore the same thickness "
        "-- equal footprint is structural here, not an argument. Feature widths are in the "
        "figure legend and in the data file's meta block.")


def fig_R():
    fig, ax = newax(fs=(7.6, 5.2))
    for k in KINDS:
        d = res[f'R_{k}']
        ax.semilogy(d[0], d[1], STY[k], color=C[k], lw=2.2, label=LBL[k])
    for pf in meta.get('pml_factors', []):
        d = res[f'R_PML_pf{pf}']
        ax.semilogy(d[0], d[1], '--' if pf != 1 else '-', color=PC[pf], lw=2.0,
                    label=f'PML pf={pf}')
    ax.set_ylim(1e-14, 1.5); ax.set_xlim(0, 89)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title(rf"All terminators at $L={meta['L_over_lam']:g}\lambda$, "
                 rf"{meta['N_term']} elements", fontsize=13, weight='bold')
    save(fig, 'angle_vs_terminator.png', NOTE)


def fig_profiles():
    """Plotted straight from the files COMSOL was fed."""
    fig, ax = newax(xl=r'$y/\lambda$  (0 = entry face)', yl=r"$\epsilon\,'$", fs=(7.6, 5.2))
    ax2 = ax.twinx(); ax2.set_ylabel(r"$\epsilon\,''$"); ax2.grid(False)
    for k in KINDS:
        f = os.path.join(S.HERE, meta['profiles'][k]['file'])
        y, er, ei, _ = S.read_profile(f)
        Lp = y[-1]/S.LAM
        ax.plot(Lp - y/S.LAM, er, STY[k], color=C[k], lw=2.0, label=LBL[k])
        ax2.plot(Lp - y/S.LAM, ei, ':', color=C[k], lw=1.6)
    ax.set_xlim(0, meta['L_over_lam']); ax.legend(fontsize=9, loc='upper right')
    ax.set_title('Terminator profiles as imported', fontsize=13, weight='bold')
    save(fig, 'angle_vs_terminator_profiles.png',
         "Solid = eps' (left axis), dotted = eps'' (right axis), read back from the same "
         "profiles/*.txt files that COMSOL imported, so the figure cannot disagree with the "
         "model. Wave enters from the LEFT.")


if __name__ == '__main__':
    print('writing figures:')
    fig_R(); fig_profiles()
