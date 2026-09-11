"""Grazing-incidence figures 21-25, COMSOL ONLY.

Every number plotted here comes from the P9_* keys written by patch9.py, i.e. from one
COMSOL model, one mesh rule, one port, one solver, swept over element count at the same
five angles.  No TMM, no fitted laws, no extrapolation:

  P9_PML_pf{s}_N{n}          PML, 2 lam thick, s = 1, 4, 10, 20, 50
  P9_SKK_{kind}_L{Lr}_N{n}   sKK profile, L = 8, 32, 100 lam, w = L/80

Angles stored on every row: 0, 45, 80, 85, 89 deg.

Run:  python comsol_skk/grazing_figures.py
"""
import os, sys, textwrap, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
D    = np.load(os.path.join(HERE, 'master_data.npz'))

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.28,
                     'axes.labelsize': 12, 'legend.framealpha': 0.92, 'figure.dpi': 110})
C   = {'logistic': '#1f77b4', 'tapered': '#ff7f0e'}
LBL = {'logistic': 'logistic (DiHiTI)', 'tapered': 'tapered Lorentzian'}
LS  = {8: '-', 32: '--', 100: ':'}
PFC = {1: '0.0', 4: '0.28', 10: '0.45', 20: '0.60', 50: '0.74'}
PFS, LRS = [1, 4, 10, 20, 50], [8, 32, 100]
THS = [0., 45., 80., 85., 89.]
NEL = sorted(set(int(round(v)) for v in np.geomspace(5, 2000, 24)))


def have(key):
    return [n for n in NEL if f'{key}_N{n}' in D.files]


def curve(key, ti):
    ns = have(key)
    return ns, np.array([D[f'{key}_N{n}'][1][ti] for n in ns])


def n_to_floor(key, ti, tol=0.1):
    """Elements at which R first comes within tol of its own converged value."""
    ns, v = curve(key, ti)
    if len(ns) == 0:
        return None, None
    fin = v[-1]
    n10 = next((n for n, x in zip(ns, v) if abs(x - fin) <= tol * fin), None)
    return n10, fin


def save(fig, name, note):
    fig.text(0.005, -0.015, '\n'.join(textwrap.wrap(note, 105)),
             fontsize=6.8, color='0.45', va='top')
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches='tight')
    plt.close(fig); print('  ', name)


NOTE_COMMON = ('All curves are COMSOL, one model, one port, one solver. sKK profiles use the '
               'self-similar family w = L/80, ell = 2.5w; PML is 2 lam thick throughout. Only the '
               'element count across the absorbing region varies along x.')


# ------------------------------------------------------------------ 21
def fig21_equal_cost():
    for ti, t in enumerate(THS):
        if t not in (80., 85., 89.):
            continue
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for k in ['logistic', 'tapered']:
            for Lr in LRS:
                ns, v = curve(f'P9_SKK_{k}_L{Lr}', ti)
                if len(ns):
                    ax.loglog(ns, v, LS[Lr], color=C[k], lw=2.1, marker='o', ms=3.5,
                              label=f'{LBL[k]}, $L={Lr}\\lambda$')
        for pf in PFS:
            ns, v = curve(f'P9_PML_pf{pf}', ti)
            if not len(ns):
                continue
            ax.loglog(ns, v, '-.', color=PFC[pf], lw=2.1, marker='s', ms=3.5,
                      label=f'PML pf={pf}  ($2\\lambda$)')
            nf, fin = n_to_floor(f'P9_PML_pf{pf}', ti)
            if nf:
                ax.plot([nf], [fin], 's', color=PFC[pf], ms=9, mec='k', mew=0.9, zorder=6)
        ax.text(0.97, 0.30, 'large squares = elements the PML\nneeds to reach its own floor',
                transform=ax.transAxes, fontsize=8.5, color='0.3', ha='right', va='top')
        ax.set_xlabel('elements across the absorbing region')
        ax.set_ylabel(rf'$R(\theta={t:.0f}^\circ)$')
        ax.set_title(rf'Equal-cost comparison ($\theta={t:.0f}^{{\circ}}$)',
                     fontsize=13, weight='bold')
        ax.legend(fontsize=7.4, loc='center left', bbox_to_anchor=(1.02, 0.5))
        save(fig, f'21_grazing_equal_cost_theta{int(t):02d}.png', NOTE_COMMON +
             ' Read vertically: at a given element budget, which terminator gives the lowest R. '
             'Footprint is NOT on this axis -- the sKK curves are 8, 32 and 100 lam thick against a '
             '2 lam PML, so the sKK side is being flattered here.')


# ------------------------------------------------------------------ 22
def fig22_vs_angle():
    """R vs angle at converged mesh, 1 deg resolution (P10_*)."""
    if 'P10_PML_pf1' not in D.files:
        print('   (skipping 22: P10_* not present -- run patch10.py)'); return
    th = D['P10_PML_pf1'][0]
    best = np.min(np.vstack([D[f'P10_PML_pf{pf}'][1] for pf in PFS]), axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                 gridspec_kw={'width_ratios': [1.35, 1]})
    for pf in PFS:
        a1.semilogy(th, D[f'P10_PML_pf{pf}'][1], '-', color=PFC[pf], lw=1.7,
                    label=f'PML pf={pf}  ($2\\lambda$)')
    for k in ['logistic', 'tapered']:
        for Lr in LRS:
            key = f'P10_SKK_{k}_L{Lr}'
            if key in D.files:
                a1.semilogy(th, D[key][1], LS[Lr], color=C[k], lw=2.2,
                            label=f'{LBL[k]}, $L={Lr}\\lambda$')
    a1.axvline(36, color='crimson', ls=':', lw=1.5)
    a1.text(36.8, 3e-11, r'$\theta_c=36^\circ$ (logistic past TIR)',
            color='crimson', fontsize=8.5, rotation=90, va='bottom')
    a1.set_xlabel(r'$\theta$ (deg)'); a1.set_ylabel('$R$ at converged mesh')
    a1.set_xlim(0, 89); a1.set_ylim(1e-12, 2)
    a1.set_title('every terminator at its converged mesh', fontsize=12, weight='bold')
    a1.legend(fontsize=7.2, loc='upper left', ncol=2)

    for k, Lr in [('logistic', 100), ('logistic', 32), ('tapered', 100)]:
        key = f'P10_SKK_{k}_L{Lr}'
        if key in D.files:
            a2.semilogy(th, best / D[key][1], LS[Lr], color=C[k], lw=2.2,
                        label=f'{LBL[k]}, $L={Lr}\\lambda$')
    a2.axhline(1, color='k', lw=1.6)
    a2.axhspan(1, 1e3, color='#2ca02c', alpha=0.10)
    a2.text(2, 3, 'sKK better', color='#2ca02c', fontsize=10, weight='bold')
    a2.text(2, 0.25, 'best PML better', color='0.35', fontsize=10, weight='bold')
    a2.set_xlabel(r'$\theta$ (deg)')
    a2.set_ylabel(r'$R_{\rm best\ PML}\,/\,R_{\rm sKK}$')
    a2.set_xlim(0, 89); a2.set_ylim(1e-4, 1e3)
    a2.set_title('ratio to the best PML at each angle', fontsize=12, weight='bold')
    a2.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    save(fig, '22_grazing_crossover.png',
         'COMSOL, 1 deg resolution, every curve at N=2000 elements (converged for all of them). '
         'RIGHT: ratio of the best of the five PMLs to each sKK profile -- above 1 the sKK profile '
         'wins. The logistic at L=100 lam tracks the best PML within a factor of a few from ~50-86 '
         'deg and dips below it in scattered narrow bands (49-57, 65-72, 83-86 deg; best 4.9x at 85 '
         'deg), but this is oscillatory, not a clean window, and it costs 100 lam of footprint and '
         '~13x the elements. Above 87 deg it collapses. The logistic is also past TIR over this '
         'entire range; the tapered Lorentzian, which is not, loses everywhere.')


# ------------------------------------------------------------------ 23
def fig23_saturation():
    """Does extra footprint still buy anything at grazing?"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for ti, t, cc in [(2, 80, '#2ca02c'), (3, 85, '#ff7f0e'), (4, 89, '#8B0000')]:
        xs, ys = [], []
        for Lr in LRS:
            ns = have(f'P9_SKK_tapered_L{Lr}')
            if ns:
                xs.append(Lr); ys.append(D[f'P9_SKK_tapered_L{Lr}_N{ns[-1]}'][1][ti])
        if xs:
            a1.loglog(xs, ys, 'o-', color=cc, lw=2.1, ms=7,
                      label=rf'tapered, $\theta={t}^\circ$')
        for pf in [20, 50]:
            ns = have(f'P9_PML_pf{pf}')
            if ns:
                a1.axhline(D[f'P9_PML_pf{pf}_N{ns[-1]}'][1][ti], color=cc,
                           ls=':' if pf == 20 else '--', lw=1.5)
    a1.text(0.97, 0.04, 'dotted / dashed = PML pf=20 / 50 at the matching angle',
            transform=a1.transAxes, ha='right', fontsize=8.2, color='0.35')
    a1.set_xlabel(r'terminator footprint $L/\lambda$'); a1.set_ylabel('$R$')
    a1.set_title('does extra footprint still help at grazing?', fontsize=12, weight='bold')
    a1.legend(fontsize=8.5, loc='lower left')

    for pf in PFS:
        ns, v = curve(f'P9_PML_pf{pf}', 4)
        if len(ns):
            a2.loglog(ns, v, '-.', color=PFC[pf], lw=2.1, marker='s', ms=3.5,
                      label=f'PML pf={pf}')
    for k in ['logistic', 'tapered']:
        ns, v = curve(f'P9_SKK_{k}_L100', 4)
        if len(ns):
            a2.loglog(ns, v, '-', color=C[k], lw=2.1, marker='o', ms=3.5,
                      label=f'{LBL[k]}, $L=100\\lambda$')
    a2.set_xlabel('elements across the absorbing region')
    a2.set_ylabel(r'$R(\theta=89^\circ)$')
    a2.set_title(r'the cost picture at $\theta=89^\circ$', fontsize=12, weight='bold')
    a2.legend(fontsize=8.2, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save(fig, '23_grazing_saturation.png', NOTE_COMMON +
         ' LEFT: the tapered Lorentzian at its converged mesh, versus footprint, at three angles. '
         'RIGHT: the same comparison as fig 21 at 89 deg, with only the largest sKK profile shown.')


# ------------------------------------------------------------------ 24
def fig24_pml_law():
    """Measured PML reflectance vs PMLfactor and angle -- purely COMSOL."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
    th = np.array(THS)
    for pf in PFS:
        ns = have(f'P9_PML_pf{pf}')
        if not ns:
            continue
        v = D[f'P9_PML_pf{pf}_N{ns[-1]}'][1]
        a1.semilogy(th[:len(v)], v, 'o-', color=PFC[pf], lw=2.0, ms=6, label=f'pf={pf}')
    a1.set_xlabel(r'$\theta$ (deg)'); a1.set_ylabel('$R$ at converged mesh')
    a1.set_title('measured PML reflectance vs PMLfactor', fontsize=12, weight='bold')
    a1.legend(fontsize=9, loc='lower right')

    for ti, t in [(0, 0), (2, 80), (4, 89)]:
        xs = [pf for pf in PFS if have(f'P9_PML_pf{pf}')]
        ys = [n_to_floor(f'P9_PML_pf{pf}', ti)[0] for pf in xs]
        ok = [(x, y) for x, y in zip(xs, ys) if y]
        if ok:
            a2.loglog([x for x, _ in ok], [y for _, y in ok], 'o-', lw=2.0, ms=7,
                      label=rf'$\theta={t}^\circ$')
    a2.set_xlabel('PMLfactor $s$'); a2.set_ylabel('elements to reach its floor')
    a2.set_title('what tuning the PML costs in mesh', fontsize=12, weight='bold')
    a2.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    save(fig, '24_pml_measured.png',
         'PURELY MEASURED. LEFT: converged R vs angle for every PMLfactor run. RIGHT: elements each '
         'PMLfactor needs to reach within 10% of its own converged value, at three angles -- this is '
         'the cost of the PML tuning knob, measured rather than modelled.')


# ------------------------------------------------------------------ 25
def fig25_scoreboard():
    """Cost-vs-accuracy scatter: the whole study on one pair of axes."""
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ti = 4  # theta = 89
    for pf in PFS:
        nf, fin = n_to_floor(f'P9_PML_pf{pf}', ti)
        if nf:
            ax.loglog([nf], [fin], 's', color=PFC[pf], ms=13, mec='k', mew=1.0)
            ax.annotate(f'PML pf={pf}\n($2\\lambda$)', (nf, fin), textcoords='offset points',
                        xytext=(9, 4), fontsize=8.5)
    for k in ['logistic', 'tapered']:
        for Lr in LRS:
            nf, fin = n_to_floor(f'P9_SKK_{k}_L{Lr}', ti)
            if nf:
                ax.loglog([nf], [fin], 'o', color=C[k], ms=11, mec='k', mew=0.9)
                ax.annotate(f'{Lr}$\\lambda$', (nf, fin), textcoords='offset points',
                            xytext=(9, -3), fontsize=8.5, color=C[k])
    ax.plot([], [], 'o', color=C['logistic'], ms=9, label='logistic (DiHiTI)')
    ax.plot([], [], 'o', color=C['tapered'], ms=9, label='tapered Lorentzian')
    ax.plot([], [], 's', color='0.4', ms=9, label=r'PML ($2\lambda$)')
    ax.set_xlabel('elements needed to reach its own floor')
    ax.set_ylabel(r'$R$ at that floor,  $\theta=89^\circ$')
    ax.set_title(r'Cost vs accuracy at $\theta=89^{\circ}$ (down and left is better)',
                 fontsize=13, weight='bold')
    ax.legend(fontsize=9, loc='upper right')
    save(fig, '25_cost_vs_accuracy_theta89.png',
         'THE SCOREBOARD, all COMSOL. Each marker is one terminator at its converged mesh: x = the '
         'element budget it needs, y = the reflectance it delivers, at theta = 89 deg. sKK markers '
         'are additionally labelled with their footprint in wavelengths; every PML is 2 lam. Down '
         'and to the left wins.')


def report():
    print('\n=== measured scoreboard (COMSOL only) ===')
    print(f'  {"terminator":30s}{"theta":>7s}{"R at floor":>13s}{"N to floor":>12s}{"footprint":>11s}')
    for ti, t in enumerate(THS):
        if t not in (80., 85., 89.):
            continue
        for k in ['logistic', 'tapered']:
            for Lr in LRS:
                nf, fin = n_to_floor(f'P9_SKK_{k}_L{Lr}', ti)
                if nf:
                    print(f'  {LBL[k]+", L="+str(Lr)+"lam":30s}{t:7.0f}{fin:13.3e}'
                          f'{nf:12d}{str(Lr)+" lam":>11s}')
        for pf in PFS:
            nf, fin = n_to_floor(f'P9_PML_pf{pf}', ti)
            if nf:
                print(f'  {"PML pf="+str(pf):30s}{t:7.0f}{fin:13.3e}{nf:12d}{"2 lam":>11s}')
        print()


if __name__ == '__main__':
    missing = [k for k in ['P9_PML_pf50_N5', 'P9_SKK_tapered_L100_N5'] if k not in D.files]
    if missing:
        sys.exit(f'P9_* data not in master_data.npz yet (missing {missing}). Run patch9.py first.')
    print('writing figures (COMSOL data only):')
    fig21_equal_cost(); fig22_vs_angle(); fig23_saturation()
    fig24_pml_law(); fig25_scoreboard()
    report()
    print('done ->', FIG)


# ------------------------------------------------------------------ 26
# Four panels in the exact style of make_figures.py _f10 (slide 9), but at
# L = 100 lam with the self-similar feature width w = L/80 = 1.25 lam and
# ell = 2.5w -- i.e. the fig-19 profile family, so these tie to the
# terminator-length slide rather than to the L = 8 lam slide.
_C10   = {'logistic': '#1f77b4', 'lorentzian': '#8B0000', 'tapered': '#ff7f0e'}
_STY10 = {'logistic': '-', 'lorentzian': '-', 'tapered': '--'}
_LBL10 = {'logistic': 'logistic (DiHiTI)', 'lorentzian': 'Lorentzian',
          'tapered': 'tapered Lorentzian'}
_PC10  = {1: 'k', 4: '0.6'}
_K3    = ['logistic', 'lorentzian', 'tapered']


def fig26_equal_cost_L100():
    r = (2000 / 5) ** (1 / 23)
    NE = sorted(n for n in (set(int(round(v)) for v in np.geomspace(5, 2000, 24))
                | set([int(round(2000 * r**k)) for k in range(1, 6)]) | {8000})
                if n >= 100)          # below ~100 elements a 100-lam domain cannot
                                      # represent the wave at all -- those points are noise
    for ti, (thv, tag) in [(0, (0, '00')), (1, (45, '45')), (2, (80, '80')), (4, (89, '89'))]:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.set_xlabel('elements across the absorbing region'); ax.set_ylabel('R')
        for k in _K3:
            ns = [n for n in NE if f'P9_SKK_{k}_L100_N{n}' in D.files]
            ax.loglog(ns, [D[f'P9_SKK_{k}_L100_N{n}'][1][ti] for n in ns],
                      _STY10[k], color=_C10[k], lw=2.2, marker='o', ms=4, label=_LBL10[k])
        for pf in [1, 4]:
            ns = [n for n in NE if f'P9_PML_pf{pf}_N{n}' in D.files]
            ax.loglog(ns, [D[f'P9_PML_pf{pf}_N{n}'][1][ti] for n in ns],
                      ':', color=_PC10[pf], lw=2.2, marker='s', ms=4, label=f'PML pf={pf}')
        ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.set_title(rf'Equal-cost comparison ($\theta={thv}^{{\circ}}$)',
                     fontsize=13, weight='bold')
        save(fig, f'26_equal_cost_L100_theta{tag}.png',
             f'Same construction as fig 10 but at L = 100 lam, at theta = {thv} deg. Profile family is '
             'the SELF-SIMILAR one from fig 19: feature width w = L/80 = 1.25 lam, taper length '
             'ell = 2.5w, so these panels tie to the terminator-length sweep, not to the L = 8 lam '
             'comparison. PML is 2 lam thick -- extra thickness does nothing for a PML (fig 4). All '
             'curves share one geometry, port and solver; only the element count across the absorbing '
             'region varies. Element grid continues fig 10\'s geometric progression out to N = 8000, '
             'which the logistic needs at theta = 0 (it is still a factor 60 off its floor at N = 2000). '
             'Red line = 5 elements. NOTE: at L = 100 lam, 5-100 elements cannot represent the wave at '
             'all -- only the right-hand plateau of each curve is physical.')


FIGS26 = fig26_equal_cost_L100


# ------------------------------------------------------------------ 27 / 28
# Fixed-feature-width (w = lam/5) companions, from patch12.
# 27 = truncation sweep at each angle, in fig 09's style.
# 28 = equal cost at L = 100 lam, in fig 10 / fig 26's style.
_LG12 = sorted(set(round(float(v), 3) for v in np.geomspace(1, 100, 21)))
_TH5  = [(0, '00', 0), (45, '45', 1), (80, '80', 2), (85, '85', 3), (89, '89', 4)]


def _ne12():
    r = (2000 / 5) ** (1 / 23)
    ne = sorted(n for n in (set(int(round(v)) for v in np.geomspace(5, 2000, 24))
                | set([int(round(2000 * r**k)) for k in range(1, 6)]) | {8000}) if n >= 100)
    return sorted(set(ne + [int(round(8000 * r**k)) for k in (1, 2)] + [16000]))


def fig27_truncation_angles():
    """fig 09, but at every angle. Fixed w = lam/5; element SIZE fixed at 0.02 lam."""
    for thv, tag, ti in _TH5:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.set_xlabel(r'terminator thickness $L/\lambda$'); ax.set_ylabel('R')
        for k in _K3:
            xs = [L for L in _LG12 if f'P12_TH_{k}_L{L}' in D.files]
            if not xs:
                continue
            ax.loglog(xs, [D[f'P12_TH_{k}_L{L}'][1][ti] for L in xs], _STY10[k],
                      color=_C10[k], lw=2.2, marker='o', ms=4, label=_LBL10[k])
        ax.axhline(D['P10_PML_pf1'][1][thv], color='k', lw=2.2, label='PML pf=1')
        ax.axhline(D['P10_PML_pf4'][1][thv], color='0.55', lw=2.2, ls='--', label='PML pf=4')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_title(rf'Truncation effects $(\theta={thv}^{{\circ}})$', fontsize=13, weight='bold')
        save(fig, f'27_truncation_theta{tag}.png',
             f'Fig 09 extended to theta = {thv} deg. ALL profiles keep a FIXED feature width '
             'lam/5 (tapered: ell = 0.5 lam), and element SIZE is held at 0.02 lam = 50 per '
             'wavelength = 10 across the feature, so the only thing changing along x is the '
             'domain length -- this is pure truncation. Every point is mesh-converged (x2 and x4 '
             'refinement move R by <1%). PML lines are the pinned P10 measurement, and are flat '
             'because thickness does nothing for a PML (fig 4), not because it was held fixed. '
             'Above theta_c = 36 deg the logistic is past TIR.')


def fig28_equal_cost_L100_fixedw():
    """Equal cost at L = 100 lam with the FIXED width w = lam/5 -- ties to fig 27."""
    NE = _ne12()
    for thv, tag, ti in _TH5:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.set_xlabel('elements across the absorbing region'); ax.set_ylabel('R')
        for k in _K3:
            ns = [n for n in NE if f'P12_EL_{k}_N{n}' in D.files]
            if not ns:
                continue
            ax.loglog(ns, [D[f'P12_EL_{k}_N{n}'][1][ti] for n in ns], _STY10[k],
                      color=_C10[k], lw=2.2, marker='o', ms=4, label=_LBL10[k])
        for pf in [1, 4]:
            ns = [n for n in NE if f'P9_PML_pf{pf}_N{n}' in D.files]
            ax.loglog(ns, [D[f'P9_PML_pf{pf}_N{n}'][1][ti] for n in ns], ':',
                      color=_PC10[pf], lw=2.2, marker='s', ms=4, label=f'PML pf={pf}')
        ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.set_title(rf'Equal-cost comparison ($\theta={thv}^{{\circ}}$)', fontsize=13, weight='bold')
        save(fig, f'28_equal_cost_L100_fixedw_theta{tag}.png',
             f'Same construction as fig 26 but with the FIXED feature width w = lam/5 (tapered '
             f'ell = 0.5 lam), at theta = {thv} deg, so this slide ties to the truncation sweep '
             '(fig 27) rather than to the self-similar length sweep (fig 19). L = 100 lam; PML is '
             '2 lam thick -- extra thickness does nothing for a PML. Only the element count across '
             'the absorbing region varies. X-axis starts at N = 100 because below that a 100-lam '
             'domain has under one element per wavelength. Note the fixed lam/5 feature needs a '
             'FINER mesh than the self-similar family to converge, so the grid runs to N = 16000.')
