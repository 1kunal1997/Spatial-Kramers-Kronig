"""Potential-theory attack on the passivity criterion (Aug 2026).

Integrating the periodic Hilbert transform by parts converts DiHiTI into a
closed-form log-kernel superposition over "step atoms":

    eps''(x) = (1/pi) int_{-L}^{L} g(t) * ln[ cos(pi t/2L) / |sin(pi(x-t)/2L)| ] dt,
    g(t) = -d eps'/dt   (the *drop density* of the profile).

No Hilbert transform appears.  Consequences tested here:

  A. Off-center sharp step: gain depth (D/pi) ln sec(pi x0/2L) at the antipode
     -> a monotone step passes only if exactly centered.
  B. Two-step staircase at +-x0: passive iff |x0| <= L/2 (sharp threshold at
     the MIDDLE HALF of the domain), gain appears at the CENTER beyond it.
  C. THEOREM (sufficient, no HT): if eps' is non-increasing, its drop density
     is supported in |x| <= L/2, and the balance integral
         M1 = int g(t) tan(pi t/2L) dt = 0,
     then eps'' >= 0.  Monte-Carlo over random monotone profiles, including
     asymmetric balanced ones.
  D. Necessary scalar screens on profile_generator's samples:
         sigma = int g(t) ln(2 cos(pi t/2L)) dt  =  pi * mean(eps'')  >= 0
         M1 (balance)  = 0     [flat-ended profiles]
     Confusion matrix vs ground-truth labels; sigma is orientation-odd and
     breaks the mirror tie.
  E. Crossover family logistic+bump: near-threshold gain depth follows the
     endpoint expansion  depth ~ -M1^2 / (2 pi M2),  M2 = int g sec^2 dt.
  F. Moment-evasion demo: a high-frequency dressing keeps every low moment of
     eps' fixed while flipping the label -> no finite battery of smooth
     moments can be sufficient AND necessary at once.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import tmm_helper as tmm_h

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L, N = 2.5, 4096
NB = 1.7
EPS_B, EPS_AIR = NB**2, 1.0
D_FULL = EPS_B - EPS_AIR
FIGDIR = os.path.join(_ROOT, 'theory', 'figures', '2026Aug13_potential_criterion')
os.makedirs(FIGDIR, exist_ok=True)

xx = np.linspace(-L, L, N)
TH = np.pi * xx / L                       # circle angle, endpoint at +-pi
XI = np.tan(np.clip(TH / 2, -np.pi/2 + 1e-9, np.pi/2 - 1e-9))


def dihiti(e):
    v = np.imag(hilbert(np.gradient(e, xx)))
    c = cumulative_trapezoid(v, xx, initial=0)
    return c - np.linspace(c[0], c[-1], len(e))


def drop_density(e):
    return -np.gradient(e, xx)


def M1(e):
    """Balance integral int g tan(pi t/2L) dt (endpoint-slope moment)."""
    g = drop_density(e)
    return np.trapezoid((g * np.tan(TH / 2))[1:-1], xx[1:-1])


def M2(e):
    g = drop_density(e)
    return np.trapezoid((g / np.cos(TH / 2)**2)[1:-1], xx[1:-1])


def sigma_mean(e):
    """sigma = int g ln(2 cos(pi t/2L)) dt = pi * mean(eps'').  No HT needed."""
    g = drop_density(e)
    kern = np.log(2 * np.cos(TH / 2) + 1e-300)
    return np.trapezoid((g * kern)[1:-1], xx[1:-1])


# ----------------------------------------------------------------- A
def part_A():
    print('=' * 78)
    print('A. Off-center sharp step: gain = (D/pi) ln sec(pi x0/2L) at the antipode')
    print('=' * 78)
    print(f"{'x0':>6}{'min eps'' num':>14}{'predicted':>12}{'x_min num':>11}{'antipode':>10}")
    x0s = np.linspace(0, 1.8, 10)
    depths, preds = [], []
    for x0 in x0s:
        e = tmm_h.logistic(xx - x0, 60.0, NB)
        ei = dihiti(e)
        th0 = np.pi * x0 / L
        pred = (D_FULL / np.pi) * np.log(np.cos(th0 / 2))
        anti = x0 - L if x0 > 0 else x0 + L
        depths.append(ei.min()); preds.append(pred)
        print(f"{x0:>6.2f}{ei.min():>14.5f}{pred:>12.5f}"
              f"{xx[np.argmin(ei)]:>11.3f}{anti:>10.3f}")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(x0s / L, depths, 'o', label='DiHiTI numeric')
    ax.plot(x0s / L, preds, '-', label=r'$(D/\pi)\ln\cos(\pi x_0/2L)$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'step position $x_0/L$'); ax.set_ylabel(r"min $\epsilon''$")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'A_offcenter_step.png'), dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- B
def part_B():
    print('\n' + '=' * 78)
    print('B. Symmetric two-step staircase at +-x0: passive iff x0 <= L/2')
    print('=' * 78)
    print(f"{'x0/L':>7}{'min eps'' num':>14}{'pred pair kernel':>17}{'x at min':>10}")
    r = np.linspace(0.25, 0.75, 11)
    mins = []
    for f in r:
        x0 = f * L
        # staircase: eps_b for x<-x0, midway value between, eps_air for x>x0
        e = EPS_B - (D_FULL / 2) * ((1 + np.tanh(30 * (xx + x0))) / 2) \
                  - (D_FULL / 2) * ((1 + np.tanh(30 * (xx - x0))) / 2)
        ei = dihiti(e)
        th0 = np.pi * x0 / L
        # pair kernel at center theta=0: (D/2pi) ln[(1+cos th0)/(1 - cos th0)] ...
        # exact pair value at gamma=1 (center): (D/2/pi)*ln[(1+c0)/(1-c0)] with c0=cos th0
        c0 = np.cos(th0)
        pred_center = (D_FULL / (2 * np.pi)) * np.log((1 + c0) / (1 - c0)) if abs(c0) < 1 else np.inf
        mins.append(ei.min())
        print(f"{f:>7.3f}{ei.min():>14.5f}{min(pred_center,9.99):>17.5f}"
              f"{xx[np.argmin(ei)]:>10.3f}")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(r, mins, 'o-')
    ax.axvline(0.5, color='r', ls='--', label=r'$x_0 = L/2$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$x_0/L$'); ax.set_ylabel(r"min $\epsilon''$")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'B_pair_threshold.png'), dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- C
def _random_monotone(rng, n_atoms, c_supp, balanced):
    """Random monotone profile: drop density = positive Gaussians in |x|<=c_supp."""
    centers = rng.uniform(-c_supp, c_supp, n_atoms)
    widths = rng.uniform(0.04, 0.25, n_atoms) * L
    weights = rng.uniform(0.2, 1.0, n_atoms)
    g = np.zeros_like(xx)
    for c, s, w in zip(centers, widths, weights):
        g += w * np.exp(-((xx - c) / s)**2)
    g[np.abs(xx) > c_supp] = 0.0          # hard truncation keeps support exact
    if balanced:
        # add a compensator atom inside the support to zero the balance integral
        S = np.trapezoid((g * np.tan(TH / 2))[1:-1], xx[1:-1])
        pos = -0.7 * c_supp * np.sign(S)
        comp = np.exp(-((xx - pos) / (0.05 * L))**2)
        comp[np.abs(xx) > c_supp] = 0.0
        Sc = np.trapezoid((comp * np.tan(TH / 2))[1:-1], xx[1:-1])
        if abs(Sc) > 1e-12:
            g += (-S / Sc) * comp
        if g.min() < 0:                    # compensator overshoot -> reject
            return None
    tot = np.trapezoid(g, xx)
    g *= D_FULL / tot
    e = EPS_B - cumulative_trapezoid(g, xx, initial=0)
    return e


def part_C(n_trials=400):
    print('\n' + '=' * 78)
    print('C. THEOREM check: monotone + support in |x|<=L/2 + balance=0  =>  passive')
    print('=' * 78)
    rng = np.random.default_rng(7)
    res = {'thm': [], 'sym_out': [], 'unbal_in': []}
    for _ in range(n_trials):
        # (i) theorem case: balanced, middle-half support (asymmetric allowed)
        e = _random_monotone(rng, rng.integers(1, 5), 0.5 * L, balanced=True)
        if e is not None:
            res['thm'].append(dihiti(e).min())
        # (ii) symmetric-in-law but unbalanced, middle-half support
        e = _random_monotone(rng, rng.integers(1, 5), 0.5 * L, balanced=False)
        if e is not None:
            res['unbal_in'].append(dihiti(e).min())
        # (iii) support beyond the middle half (unbalanced)
        e = _random_monotone(rng, rng.integers(1, 5), 0.92 * L, balanced=False)
        if e is not None:
            res['sym_out'].append(dihiti(e).min())
    for k, lab in (('thm', 'balanced, support<=L/2   (theorem: MUST all pass)'),
                   ('unbal_in', 'unbalanced, support<=L/2 (theorem silent)'),
                   ('sym_out', 'support up to 0.92L      (theorem silent)')):
        v = np.array(res[k])
        nfail = int((v < -1e-4 * D_FULL).sum())
        print(f"  {lab}:  n={len(v)}, worst min eps'' = {v.min():+.6f}, "
              f"failures = {nfail}")
    return res


# ----------------------------------------------------------------- D
def part_D(n_samples=400):
    print('\n' + '=' * 78)
    print('D. Necessary screens (sigma >= 0, M1 = 0) vs profile_generator labels')
    print('=' * 78)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from profile_generator import sample
    rows = []
    for s in sample(n_samples):
        e = s['e_re']
        g = -np.gradient(e, s['x'])
        th = np.pi * s['x'] / s['x'][-1]
        sig = np.trapezoid((g * np.log(2 * np.cos(th / 2) + 1e-300))[1:-1],
                           s['x'][1:-1])
        m1 = np.trapezoid((g * np.tan(th / 2))[1:-1], s['x'][1:-1])
        scale1 = np.trapezoid((np.abs(g * np.tan(th / 2)))[1:-1], s['x'][1:-1])
        scale0 = np.trapezoid(np.abs(g), s['x']) * np.log(2)
        rows.append(dict(name=s['name'], passive=s['passive'],
                         min_eim=s['min_eim'],
                         sig=sig, sig_rel=sig / max(scale0, 1e-12),
                         m1=m1, m1_rel=abs(m1) / max(scale1, 1e-12),
                         mean_eim=s['e_im'].mean()))
    # verify sigma identity on generator output.  Families with nonzero edge
    # slope (triangle, tanh_slopes, sine) have a log-singular sigma integrand
    # at x=+-L, so trapezoid quadrature converges slowly there; split them out.
    edgy = {'triangle', 'tanh_slopes', 'sine'}
    err_flat = max((abs(r['sig'] - np.pi * r['mean_eim']) for r in rows
                    if r['name'] not in edgy), default=0.0)
    err_edgy = max((abs(r['sig'] - np.pi * r['mean_eim']) for r in rows
                    if r['name'] in edgy), default=0.0)
    print(f"  identity |sigma - pi*mean(eps'')|: flat-ended families max = {err_flat:.2e},"
          f" edge-sloped families max = {err_edgy:.2e} (quadrature, see comment)")

    TOL_S, TOL_M = -1e-3, 3e-2
    flag = [(r['sig_rel'] < TOL_S) or (r['m1_rel'] > TOL_M) for r in rows]
    gain = [not r['passive'] for r in rows]
    m1_pass = [r['m1_rel'] for r, g_ in zip(rows, gain) if not g_]
    m1_gain = [r['m1_rel'] for r, g_ in zip(rows, gain) if g_]
    print(f"  |M1| rel among passive: max = {max(m1_pass):.2e}, "
          f"median = {np.median(m1_pass):.2e}")
    print(f"  |M1| rel among gain:    median = {np.median(m1_gain):.2e}")
    caught = sum(1 for f, gn in zip(flag, gain) if f and gn)
    missed = sum(1 for f, gn in zip(flag, gain) if not f and gn)
    false_alarm = sum(1 for f, gn in zip(flag, gain) if f and not gn)
    clean = sum(1 for f, gn in zip(flag, gain) if not f and not gn)
    print(f"  gain caught: {caught}/{caught + missed}   missed: {missed}")
    print(f"  passive clean: {clean}/{clean + false_alarm}   false alarms: {false_alarm}")
    if false_alarm:
        print('  FALSE ALARM details (would falsify necessity!):')
        for r, f in zip(rows, flag):
            if f and r['passive']:
                print(f"    {r['name']}: sig_rel={r['sig_rel']:+.4f} "
                      f"m1_rel={r['m1_rel']:.4f} min_eim={r['min_eim']:+.5f}")
    # scatter figure
    fig, ax = plt.subplots(figsize=(5.4, 4))
    for gn, col, lab in ((False, 'tab:blue', 'passive'), (True, 'tab:red', 'gain')):
        xs = [r['m1_rel'] for r, g_ in zip(rows, gain) if g_ == gn]
        ys = [r['sig_rel'] for r, g_ in zip(rows, gain) if g_ == gn]
        ax.scatter(xs, ys, s=10, c=col, label=lab, alpha=0.6)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(TOL_M, color='k', ls=':', lw=0.8)
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_xlabel(r'$|M_1|$ (relative)   [balance violation]')
    ax.set_ylabel(r'$\sigma$ (relative)   [$= \pi \cdot$ mean loss]')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'D_screen_scatter.png'), dpi=160)
    plt.close(fig)

    # -------- probe extension: evaluate eps'' AT A FEW FIXED POINTS via the
    # closed-form kernel (one O(N) integral per probe, still no HT), and flag
    # any probe < 0.  This is a *pointwise-necessary* battery.
    probes_x = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * L
    caught2 = missed2 = fa2 = clean2 = 0
    for r_, s_ in zip(rows, sample(n_samples)):
        x = s_['x']; e = s_['e_re']
        gg = -np.gradient(e, x)
        th_t = np.pi * x / x[-1]
        bad = False
        for xp in probes_x + 0.5 * (x[1] - x[0]):   # stagger off grid nodes
            th = np.pi * xp / x[-1]
            integ = gg * np.log(np.abs(np.cos(th_t / 2))
                                / np.maximum(np.abs(np.sin((th - th_t) / 2)), 1e-300))
            val = np.trapezoid(integ, x) / np.pi
            # probe quadrature is ~2% accurate near the log singularity, so
            # the flag threshold must sit above that noise floor
            if val < -3e-2 * max(abs(np.pi * r_['mean_eim']), 1e-6):
                bad = True
                break
        f2 = bad or (r_['sig_rel'] < TOL_S) or (r_['m1_rel'] > TOL_M)
        gn = not r_['passive']
        caught2 += f2 and gn; missed2 += (not f2) and gn
        fa2 += f2 and (not gn); clean2 += (not f2) and (not gn)
    print(f"  + 7-point kernel probes: gain caught {caught2}/{caught2 + missed2}, "
          f"false alarms {fa2}/{fa2 + clean2}")

    # mirror tie-break demo
    e = tmm_h.logistic(xx, 4.0, NB)
    for lab, prof in (('logistic', e), ('mirrored logistic', e[::-1])):
        g = -np.gradient(prof, xx)
        sig = np.trapezoid((g * np.log(2 * np.cos(TH / 2) + 1e-300))[1:-1], xx[1:-1])
        print(f"  mirror demo: {lab:<18} sigma = {sig:+.4f}   "
              f"min eps'' = {dihiti(prof).min():+.4f}")
    return rows


# ----------------------------------------------------------------- E
def part_E():
    print('\n' + '=' * 78)
    print('E. logistic+bump crossover: endpoint expansion depth = -M1^2/(2 pi M2)')
    print('=' * 78)
    print(f"{'alpha':>7}{'min eps''':>11}{'pred (endpoint)':>16}{'x at min':>10}{'M1':>9}")
    als = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    bump = D_FULL * np.exp(-((xx - 0.0) / 0.6)**2)
    e0 = tmm_h.logistic(xx, 4.0, NB)
    for al in als:
        e = e0 + al * bump
        ei = dihiti(e)
        m1, m2 = M1(e), M2(e)
        pred = -m1**2 / (2 * np.pi * m2)
        print(f"{al:>7.3f}{ei.min():>11.6f}{pred:>16.6f}"
              f"{xx[np.argmin(ei)]:>10.3f}{m1:>9.4f}")


# ----------------------------------------------------------------- F
def part_F():
    print('\n' + '=' * 78)
    print('F. Moment evasion: high-freq dressing invisible to any fixed moment battery')
    print('=' * 78)
    e0 = tmm_h.logistic(xx, 4.0, NB)
    # place the dressing where the base loss is thin (near, not at, the edge):
    # a centered dressing is harmless because the logistic's own loss dominates
    win = np.exp(-((np.abs(xx) - 1.9) / 0.25)**2)
    # smooth C1 cutoff to EXACT zero before the edge, so the tan-singular
    # balance integral is not polluted by an m-amplified window tail
    win *= 0.5 * (1 + np.cos(np.pi * np.clip((np.abs(xx) - 2.15) / 0.2, 0, 1)))
    m10, sig0 = M1(e0), sigma_mean(e0)
    comp = np.exp(-(xx / 0.6)**2)         # even bump: shifts M1 linearly

    def legendre_moments(e, K=6):
        u = xx / L
        return np.array([np.trapezoid(e * np.polynomial.legendre.Legendre.basis(k)(u), xx)
                         for k in range(K)])

    lm0 = legendre_moments(e0)
    for m in (8, 16, 32, 64):
        for a in (0.25,):
            pert = a * np.cos(m * np.pi * xx / L) * win
            e = e0 + pert
            # re-balance M1 exactly with the smooth centered compensator
            e = e - (M1(e) - m10) / (M1(e0 + comp) - m10) * comp
            ei = dihiti(e)
            dlm = np.abs(legendre_moments(e) - lm0).max()
            print(f"  m={m:>3} a={a:.2f}: min eps''={ei.min():+.5f}  "
                  f"dM1={M1(e) - m10:+.1e}  dsigma={sigma_mean(e) - sig0:+.1e}  "
                  f"max|dLegendre 0..5|={dlm:.1e}")

    # Monotonicity is an essential hypothesis: a SYMMETRIC (hence balanced),
    # middle-half-supported drop density with a negative dip at the center.
    # A mild central dip in g is TOLERATED (log kernel is only logarithmically
    # singular); a deep one is not.  Both are symmetric+balanced, support<L/2.
    for lab, dip_amp, dip_w in (('mild dip', 0.9, 0.08), ('deep dip', 2.0, 0.10)):
        g_nm = (np.exp(-((xx - 0.5) / 0.12)**2) + np.exp(-((xx + 0.5) / 0.12)**2)
                - dip_amp * np.exp(-(xx / dip_w)**2))
        g_nm *= D_FULL / np.trapezoid(g_nm, xx)
        e_nm = EPS_B - cumulative_trapezoid(g_nm, xx, initial=0)
        print(f"  non-monotone control ({lab}, symmetric, support<L/2, balanced):"
              f" min eps'' = {dihiti(e_nm).min():+.4f}")
    print('  -> monotonicity cannot simply be dropped from the theorem; the')
    print('     tolerance to interior dips is finite and set by the log kernel.')


if __name__ == '__main__':
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()
    part_F()
    print(f"\nfigures -> {FIGDIR}")
