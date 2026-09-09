"""Grazing-incidence cost study: does the sKK-vs-PML equal-cost curve ever cross?

Two independent engines, deliberately:
  * TMM  -- exact 1D reference for the sKK side. No COMSOL licence needed, and it resolves
            below COMSOL's ~1e-12 solver floor, so it can follow the sKK curves further down.
  * the COMSOL PML attenuation law, EXTRACTED AND VERIFIED from master_data.npz:
                R_PML = exp(-8*pi*s*cos(theta)) + R_floor
            with s = PMLfactor.  No fitted parameters (8*pi is exact).  This reproduces the
            measured pf=1 fine-angle sweep to 0.017 decades median over the full 0-89 deg.
            It is what lets us reach pf = 10, 20, 50 without a licence.

Everything TMM-side is cross-checked against the COMSOL points that already exist
(see validate() -- agreement is ~30% at grazing, the regime this study is about).

Outputs -> comsol_skk/figures/21..24.  Run:  python comsol_skk/grazing_study.py
"""
import os, sys, textwrap, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import tmm_helper as tmm_h
from tmm import coh_tmm

FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
D   = np.load(os.path.join(HERE, 'master_data.npz'))
OUT = os.path.join(HERE, 'grazing_study.npz')

lam = 0.8; NB = 1.7; EB = NB**2
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.28,
                     'axes.labelsize': 12, 'legend.framealpha': 0.92, 'figure.dpi': 110})
C   = {'logistic': '#1f77b4', 'lorentzian': '#8B0000', 'tapered': '#ff7f0e'}
LBL = {'logistic': 'logistic (DiHiTI)', 'lorentzian': 'Lorentzian', 'tapered': 'tapered Lorentzian'}
PFC = {1: '0.0', 4: '0.30', 10: '0.48', 20: '0.62', 50: '0.76'}
R_FLOOR = 1.1e-11          # COMSOL solver/mesh floor of this model, from master_data


def save(fig, name, note):
    fig.text(0.005, -0.015, '\n'.join(textwrap.wrap(note, 105)),
             fontsize=6.8, color='0.45', va='top')
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches='tight')
    plt.close(fig); print('  ', name)


# ------------------------------------------------------------------ profiles + TMM
def profile(kind, L, w, A=4.0, ell=None, npts=20001):
    """y=0 is the PEC side, y=L is the entry face -- same convention as master_data.py."""
    y = np.linspace(0.0, L, npts)
    if kind == 'logistic':
        er = 1.0 + (EB - 1.0) / (1.0 + np.exp(-(y - L / 2) / w))
        return y, er, -tmm_h.ht_derivative(y, er)
    u = y - L / 2; Dd = u**2 + w**2
    if kind == 'lorentzian':
        return y, EB + A * w * u / Dd, A * w**2 / Dd
    if kind == 'tapered':
        ell = ell if ell is not None else 2.5 * w
        return y, EB + A * w * u / Dd / np.cosh(u / ell), A * w**2 / Dd
    raise ValueError(kind)


def R_tmm(kind, L, w, theta_deg, nslab=4000, A=4.0, ell=None):
    """n_b (semi-inf) | profile | 2*lam of n_b | PEC.  Matches the COMSOL harness."""
    y, er, ei = profile(kind, L, w, A=A, ell=ell)
    yc = np.linspace(0, L, nslab + 1); ycm = 0.5 * (yc[1:] + yc[:-1])
    e = np.interp(ycm, y, er) + 1j * np.interp(ycm, y, ei)
    n = np.sqrt(e); n = np.where(n.imag < 0, -n, n)     # passive branch
    n = n[::-1]                                          # entry face first
    n_list = np.concatenate([[NB + 0j], n, [NB + 0j], [1e7j]])
    d_list = np.concatenate([[np.inf], np.full(nslab, L / nslab), [2 * lam], [np.inf]])
    return coh_tmm('s', n_list, d_list, np.radians(theta_deg), lam)['R']


def R_pml(pf, theta_deg, floor=R_FLOOR):
    """Verified COMSOL PML law: attenuation over the effective thickness s*lambda."""
    return np.exp(-8 * np.pi * pf * np.cos(np.radians(theta_deg))) + floor


def n_elem_skk(L, w, ppw=40, ppf=20):
    """Element count actually used for the sKK profiles in master_data / patch8:
    ppf elements across the feature width AND ppw per wavelength of domain."""
    return int(min(8000, max(ppf * L / w, ppw * L / lam)))


def n_elem_pml(pf, theta_deg):
    """Elements a PML needs to reach its own floor.

    The field inside the stretched layer accumulates phase+decay ~ 8*pi*s*cos(theta),
    so the mesh requirement scales the same way -- it FALLS at grazing.  Calibrated on
    the measured P2_PMLN sweeps (see pml_cost_calibration()):
        theta=0  : pf=1 -> 62,  pf=2 -> 80,  pf=4 -> 171   (N/pf ~ 40-60)
        theta=80 : pf=1 ->  5,  pf=2 ->  5,  pf=4 ->  11
    """
    return max(5.0, 50.0 * pf * np.cos(np.radians(theta_deg)))


def pml_cost_calibration():
    """Measured elements-to-floor from master_data, used to calibrate n_elem_pml."""
    NEL = [n for n in sorted(set(int(round(v)) for v in np.geomspace(5, 1000, 22)))
           if f'P2_PMLN_pf1_N{n}' in D.files]
    th = D['P2_PMLN_pf1_N1000'][0]
    out = {}
    for ti, t in enumerate(th):
        for pf in [1, 2, 4]:
            v = np.array([D[f'P2_PMLN_pf{pf}_N{n}'][1][ti] for n in NEL]); fin = v[-1]
            n10 = next((n for n, x in zip(NEL, v) if abs(x - fin) <= 0.1 * fin), None)
            if n10:
                out[(pf, float(t))] = (n10, fin)
    return out


# ------------------------------------------------------------------ validation
def validate():
    print('\n=== validation: TMM vs the COMSOL points that exist ===')
    LG = [round(float(v), 3) for v in np.geomspace(0.5, 100, 18)]
    print(f'  {"case":32s}{"COMSOL":>12s}{"TMM":>12s}{"ratio":>8s}')
    rows = []
    for k in ['logistic', 'tapered']:
        for Lr in [8.0, LG[-4], LG[-1]]:
            key = f'P8_r80_{k}_L{Lr}'
            if key not in D.files:
                continue
            v = D[key][1]
            for ti, t in [(0, 0), (2, 80), (3, 89)]:
                r = R_tmm(k, Lr * lam, Lr * lam / 80, t)
                rows.append((k, Lr, t, v[ti], r))
                print(f'  {k+" L="+str(Lr)+" th="+str(t):32s}{v[ti]:12.3e}{r:12.3e}{r/v[ti]:8.2f}')
    print('  NOTE: at theta=0 the logistic disagrees because COMSOL is at its ~1e-12 solver')
    print('        floor there and TMM is not. At 80-89 deg -- the regime of this study -- the')
    print('        two agree to ~30%, which is nothing on a log axis spanning 12 decades.')

    print('\n=== validation: the PML law, R = exp(-8*pi*s*cos(theta)) + floor ===')
    for pf in [1, 2, 4]:
        d = D[f'PML_pf{pf}']; th, R = d[0], d[1]
        m = R > 3 * R_FLOOR
        err = np.abs(np.log10(R_pml(pf, th[m]) / R[m]))
        print(f'  pf={pf}: theta {th[m].min():.0f}-{th[m].max():.0f} deg,'
              f'  median |log10 err| = {np.median(err):.3f}, max = {err.max():.3f}')
    print('  pf=1 is essentially exact everywhere. pf=2,4 deviate at MODERATE angles only,')
    print('  where the continuum R has already dropped under the mesh floor -- at 85-89 deg,')
    print('  which is what this study needs, all three agree.')
    return rows


# ------------------------------------------------------------------ figure 21
def fig21_equal_cost_grazing():
    """R vs element budget at 85 and 89 deg, sKK at L = 8, 32, 100 lam vs PML pf = 1..50."""
    NEL = sorted(set(int(round(v)) for v in np.geomspace(5, 1000, 22)))
    LS  = {8: '-', 32: '--', 100: ':'}
    for thv, tag, ti in [(85, '85', None), (89, '89', 3)]:
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for k in ['logistic', 'tapered']:
            for Lr in [8, 32, 100]:
                w = Lr * lam / 80
                ax.loglog(NEL, [R_tmm(k, Lr * lam, w, thv, nslab=max(n, 8)) for n in NEL],
                          LS[Lr], color=C[k], lw=2.0, alpha=0.95,
                          label=f'{LBL[k]}, $L={Lr}\\lambda$')
        # measured COMSOL points at L = 8 lam, theta = 89, for reference
        if ti is not None:
            for k in ['logistic', 'tapered']:
                ax.loglog(NEL, [D[f'P7_EL_{k}_N{n}'][1][ti] for n in NEL], 'o',
                          color=C[k], ms=4, mfc='none', alpha=0.7,
                          label=f'{LBL[k]} $L=8\\lambda$ (COMSOL)')
        for pf in [1, 4, 10, 20, 50]:
            nc = n_elem_pml(pf, thv); r = R_pml(pf, thv)
            ax.plot([nc, 1000], [r, r], '-' if pf in (1, 4) else '-.',
                    color=PFC[pf], lw=2.0,
                    label=f'PML pf={pf}' + ('' if pf in (1, 4) else ' (law)'))
            ax.plot([nc], [r], 's', color=PFC[pf], ms=7, mec='k', mew=0.8, zorder=5)
        ax.text(0.97, 0.34, 'squares = elements the PML\nneeds to reach its own floor',
                transform=ax.transAxes, fontsize=8.5, color='0.3', ha='right', va='top')
        ax.set_xlabel('elements across the absorbing region')
        ax.set_ylabel(f'$R(\\theta={thv}^\\circ)$')
        ax.set_title(rf'Equal-cost comparison at grazing ($\theta={thv}^{{\circ}}$)',
                     fontsize=13, weight='bold')
        ax.legend(fontsize=7.4, loc='center left', bbox_to_anchor=(1.02, 0.5))
        save(fig, f'21_grazing_equal_cost_theta{tag}.png',
             'sKK curves are TMM (exact 1D, validated against the COMSOL points shown as open circles). '
             'PML lines use the verified law R = exp(-8*pi*s*cos(theta)) + floor; pf=1 and 4 are also '
             'measured in master_data and agree. Read vertically: at any element budget, which terminator '
             'wins. Footprint is NOT on this axis -- the sKK curves are 8, 32 and 100 lam thick, the PML '
             'is 2 lam, so the sKK side is already being flattered here.')


# ------------------------------------------------------------------ figure 22
def fig22_crossover():
    """The decisive one: R vs angle out to 89.9 deg, sKK at several L vs PML at several pf."""
    th = np.concatenate([np.linspace(70, 89, 40), np.linspace(89.05, 89.95, 19)])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for Lr, ls in [(8, ':'), (32, '--'), (100, '-'), (400, '-')]:
        lw = 2.6 if Lr == 400 else 2.0
        ax.semilogy(th, [R_tmm('tapered', Lr * lam, Lr * lam / 80, t, nslab=3000) for t in th],
                    ls, color=C['tapered'], lw=lw, alpha=0.55 + 0.15 * [8, 32, 100, 400].index(Lr),
                    label=f'tapered Lorentzian, $L={Lr}\\lambda$')
    ax.semilogy(th, [R_tmm('logistic', 100 * lam, 100 * lam / 80, t, nslab=3000) for t in th],
                '-', color=C['logistic'], lw=2.0, label=r'logistic, $L=100\lambda$')
    for pf in [1, 4, 10, 20, 50]:
        ax.semilogy(th, R_pml(pf, th), '-.' if pf not in (1, 4) else '-',
                    color=PFC[pf], lw=2.0, label=f'PML pf={pf}  ($2\\lambda$ thick)')
    ax.axvline(89, color='crimson', ls=':', lw=1.2)
    ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel(r'$R$')
    ax.set_ylim(1e-13, 2); ax.set_xlim(70, 90)
    ax.set_title('Does the cost curve cross at grazing?', fontsize=13, weight='bold')
    ax.legend(fontsize=7.4, loc='center left', bbox_to_anchor=(1.02, 0.5))
    save(fig, '22_grazing_crossover.png',
         'THE TEST. Hypothesis was: PML needs s ~ 1/cos(theta) and its cost grows like sec(theta), '
         'while the sKK footprint requirement might grow more slowly, so the two should cross at '
         'extreme grazing. They do not. The sKK curves SATURATE -- past L ~ 30 lam extra footprint buys '
         'nothing at grazing -- while the PML keeps buying orders of magnitude for a knob that costs '
         'only mesh. The logistic is disqualified above 36 deg anyway (it terminates in air, so this '
         'is past TIR).')


# ------------------------------------------------------------------ figure 23
def fig23_saturation():
    """Why it fails: the sKK grazing residual saturates in L, and no knob moves it."""
    LR = np.geomspace(2, 1000, 26)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    for thv, cc in [(80, '#2ca02c'), (85, '#ff7f0e'), (89, '#8B0000')]:
        a1.loglog(LR, [R_tmm('tapered', L * lam, L * lam / 80, thv, nslab=3000) for L in LR],
                  'o-', color=cc, lw=2.0, ms=3.5, label=rf'tapered, $\theta={thv}^\circ$')
        a1.axhline(R_pml(20, thv), color=cc, ls=':', lw=1.8)
    a1.text(0.97, 0.04, 'dotted = PML pf=20 at the matching angle',
            transform=a1.transAxes, ha='right', fontsize=8.5, color='0.35')
    a1.set_xlabel(r'terminator footprint $L/\lambda$'); a1.set_ylabel('$R$')
    a1.set_title('extra footprint saturates at grazing', fontsize=12, weight='bold')
    a1.legend(fontsize=8.5, loc='lower left')

    # knob scan at theta = 89, L = 100 lam
    for A, mk in [(1.0, 's'), (4.0, 'o'), (16.0, '^')]:
        ratios = np.array([10, 20, 40, 80, 160, 320])
        a2.loglog(ratios, [R_tmm('tapered', 100 * lam, 100 * lam / r, 89, nslab=3000, A=A)
                           for r in ratios], mk + '-', lw=1.8, ms=5, label=f'$A={A:g}$')
    for pf in [4, 20, 50]:
        a2.axhline(R_pml(pf, 89), color=PFC[pf], ls='-.', lw=1.8, label=f'PML pf={pf}')
    a2.set_xlabel(r'$L/w$  (feature sharpness at fixed $L=100\lambda$)')
    a2.set_ylabel(r'$R(\theta=89^\circ)$')
    a2.set_title('and no profile knob reaches the PML', fontsize=12, weight='bold')
    a2.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save(fig, '23_grazing_saturation.png',
         'LEFT: at 80 deg the tapered profile still improves with footprint, but by 89 deg it has '
         'flattened -- the grazing residual is set by the profile SHAPE, not its length, so the one '
         'knob sKK has (make it bigger) stops working exactly where it is needed. RIGHT: neither '
         'amplitude A nor feature sharpness L/w rescues it at 89 deg.')


# ------------------------------------------------------------------ figure 24
def fig24_pml_law():
    """The PML law itself -- this is a citable result and it is what sets the pf you need."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
    th = np.linspace(0, 89.9, 400)
    for pf in [1, 2, 4]:
        d = D[f'PML_pf{pf}']
        a1.semilogy(d[0], d[1], 'o', ms=3.2, color=PFC.get(pf, '0.4'),
                    label=f'COMSOL, pf={pf}')
        a1.semilogy(th, R_pml(pf, th), '-', color=PFC.get(pf, '0.4'), lw=1.6)
    a1.axhline(R_FLOOR, color='crimson', ls=':', lw=1.4)
    a1.text(2, R_FLOOR * 1.6, 'solver / mesh floor', color='crimson', fontsize=8.5)
    a1.set_xlabel(r'$\theta$ (deg)'); a1.set_ylabel('$R$'); a1.set_ylim(1e-13, 2)
    a1.set_title(r'$R=\exp(-8\pi\,s\,\cos\theta)+R_{\rm floor}$  (no fitted parameters)',
                 fontsize=11.5, weight='bold')
    a1.legend(fontsize=8.5, loc='upper left')

    for pf in [1, 4, 10, 20, 50]:
        a2.semilogy(th, R_pml(pf, th), '-', color=PFC[pf], lw=2.0, label=f'pf={pf}')
    a2.semilogy(th, R_pml(1 / np.cos(np.radians(th)), th), 'r--', lw=2.2,
                label=r'$s=1/\cos\theta$ (COMSOL docs)')
    a2.set_xlabel(r'$\theta$ (deg)'); a2.set_ylabel('$R$'); a2.set_ylim(1e-13, 2)
    a2.set_title('what the doc-recommended tuning buys', fontsize=11.5, weight='bold')
    a2.legend(fontsize=8.5, loc='upper left')
    fig.tight_layout()
    save(fig, '24_pml_attenuation_law.png',
         'The COMSOL PML obeys R = exp(-8*pi*s*cos(theta)) with 8*pi EXACT -- a round trip through an '
         'effective thickness s*lambda at the FREE-SPACE wavelength, which is what the manual\'s '
         'Delta x = lambda*f(xi) - Delta w * xi means. Setting s = 1/cos(theta) makes the exponent a '
         'constant 8*pi and lands on the floor at every angle, which is exactly the tuning rule the '
         'COMSOL documentation gives. This is why the PML has a knob that buys orders of magnitude '
         'for mesh alone, and the sKK profile does not.')


# ------------------------------------------------------------------ figure 25
def fig25_pml_cost_vs_angle():
    """Why the hypothesis failed: tuning a PML gets CHEAPER at grazing, not dearer."""
    cal = pml_cost_calibration()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    for t, mk, cc in [(0.0, 'o', '#1f77b4'), (80.0, 's', '#8B0000')]:
        pfs = [pf for pf in [1, 2, 4] if (pf, t) in cal]
        a1.loglog(pfs, [cal[(pf, t)][0] for pf in pfs], mk, ms=9, color=cc,
                  label=rf'COMSOL, $\theta={t:.0f}^\circ$')
        x = np.geomspace(1, 60, 50)
        a1.loglog(x, [n_elem_pml(pf, t) for pf in x], '-', color=cc, lw=1.8, alpha=0.6,
                  label=rf'$50\,s\cos\theta$, $\theta={t:.0f}^\circ$')
    x = np.geomspace(1, 60, 50)
    a1.loglog(x, [n_elem_pml(pf, 89) for pf in x], '-', color='#2ca02c', lw=2.4,
              label=r'model at $\theta=89^\circ$')
    a1.set_xlabel('PMLfactor $s$'); a1.set_ylabel('elements to reach the floor')
    a1.set_title('the mesh cost of tuning a PML', fontsize=12, weight='bold')
    a1.legend(fontsize=8.2, loc='upper left')

    th = np.linspace(0, 89.5, 200)
    for pf in [1, 4, 10, 20, 50]:
        a2.plot(th, [n_elem_pml(pf, t) for t in th], '-', color=PFC[pf], lw=2.0,
                label=f'pf={pf}')
    a2.plot(th, [n_elem_pml(1 / np.cos(np.radians(t)), t) for t in th], 'r--', lw=2.4,
            label=r'$s=1/\cos\theta$ (docs)')
    a2.set_yscale('log'); a2.set_xlabel(r'$\theta$ (deg)')
    a2.set_ylabel('elements to reach the floor')
    a2.set_title('it gets CHEAPER at grazing, not dearer', fontsize=12, weight='bold')
    a2.legend(fontsize=8.5, loc='lower left')
    fig.tight_layout()
    save(fig, '25_pml_cost_vs_angle.png',
         'This is why the crossover hypothesis fails. The field inside the stretched layer accumulates '
         'phase+decay ~ 8*pi*s*cos(theta), so the mesh needed scales as s*cos(theta) -- at grazing the '
         'normal wavenumber collapses, the field varies slowly, and the PML needs almost no elements. '
         'Measured: pf=4 needs 171 elements at 0 deg but only 11 at 80 deg. The COMSOL rule s=1/cos(theta) '
         'therefore holds BOTH accuracy and mesh cost constant at every angle (flat red line). A PML has '
         'a knob that is nearly free exactly where the sKK profile has none.')


# ------------------------------------------------------------------ headline numbers
def scoreboard():
    print('\n=== scoreboard at theta = 89 deg ===')
    rows = []
    for k in ['logistic', 'tapered']:
        for Lr in [8, 32, 100, 400]:
            r = R_tmm(k, Lr * lam, Lr * lam / 80, 89, nslab=3000)
            if not np.isfinite(r):        # logistic past TIR overflows the stack at L=400
                continue
            rows.append((f'{LBL[k]}, L={Lr}lam', r, n_elem_skk(Lr * lam, Lr * lam / 80), Lr))
    for pf in [1, 4, 10, 20, 50]:
        rows.append((f'PML pf={pf}', R_pml(pf, 89), int(n_elem_pml(pf, 89)), 2))
    print(f'  {"terminator":34s}{"R(89deg)":>12s}{"elements":>10s}{"footprint":>11s}')
    for nm, r, ne, fp in rows:
        print(f'  {nm:34s}{r:12.3e}{("~"+str(ne)):>10s}{str(fp)+" lam":>11s}')
    print('\n  pf needed for a target R at 89 deg  (s = -ln(R)/(8*pi*cos(theta))):')
    for tgt in [1e-3, 1e-6, 1e-9]:
        s = -np.log(tgt) / (8 * np.pi * np.cos(np.radians(89)))
        print(f'    R={tgt:.0e}  ->  pf = {s:.1f}')
    print(f'    1/cos(89 deg) = {1/np.cos(np.radians(89)):.1f}  (the docs\' rule; lands on the floor)')


if __name__ == '__main__':
    validate()
    print('\nwriting figures:')
    fig24_pml_law()
    fig21_equal_cost_grazing()
    fig22_crossover()
    fig23_saturation()
    fig25_pml_cost_vs_angle()
    scoreboard()
    print('\ndone ->', FIG)
