"""
Independent audit of C28 / theorist's diagnosis (research/theory/h9_potential_domination.md,
h9_domination_check.py block C, h9_fixes.py block C).

Claim under test: (L1)'s 7/240 disagreements with the labelled corpus are caused by the
0.97L truncation in the eta-space (equivalently t-space) quadrature cutting off the
near-seam region where the minimum of the domination gap lives, NOT a defect of (L1) itself.

Independent test: if the diagnosis is correct, the disagreement should be a *quadrature
convergence* phenomenon -- it should shrink and eventually vanish as the truncation
window m = |x| < frac*L is pushed frac -> 1, on some plausible convergence law (log
divergence expected for a conditionally-convergent integral / edge effect). If disagreement
persists at fixed nonzero margin as frac->1 (or grows), the disagreement is not a truncation
artifact -- it is a defect of the law itself (or of the *finite* grid discretization, a
different explanation not proposed by the theorist).

We reuse profile_generator.sample(240, rng=7) exactly as h9_fixes.py does, so the same 240
profiles / the reported same 7 disagreers are examined. We do NOT import or call any
theorist code -- gap() is reimplemented here from the paper definition, independently.
"""
import sys, os
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'theory'))
sys.path.insert(0, os.path.join(_ROOT, 'theory', 'passivity_2026Aug'))

import profile_generator as pg


def logln_naive(eta, xg):
    """Plain ln|eta-xi|, no singular-cell correction -- the textbook Riemann sum."""
    d = np.abs(eta - xg)
    d = np.where(d < 1e-300, 1e-300, d)
    return np.log(d)


def gap_naive(g, t, eta, L):
    xg = np.tan(np.pi * t / (2 * L))
    D = np.trapezoid(g, t)
    eta = np.atleast_1d(np.asarray(eta, float))
    out = np.array([np.trapezoid(logln_naive(e, xg) * g, t) for e in eta])
    return out - 0.5 * D * np.log(1 + eta ** 2)


def dom_min_for_frac(xs, gs, Ls, frac):
    """Domination-gap based min eps'' estimate, using window |x| < frac*L for BOTH the
    quadrature support (t-integral truncation) and the eta evaluation points -- exactly
    the two roles 0.97L plays in the theorist's block C."""
    mm = np.abs(xs) < frac * Ls
    if mm.sum() < 5:
        return np.nan
    eta_eval = np.tan(np.pi * xs[mm] / (2 * Ls))
    dl = gap_naive(gs, xs, eta_eval, Ls)
    return -dl.max() / np.pi


def main():
    rng = np.random.default_rng(7)
    samples = list(pg.sample(240, rng=rng))
    print(f"Loaded {len(samples)} profiles (same seed=7 as h9_fixes.py)")

    fracs = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9999, 0.99999]

    # First pass at frac=0.97 to identify which profiles disagree (reproduce 233/240 claim)
    results_97 = []
    for smp in samples:
        xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
        dm = dom_min_for_frac(xs, gs, Ls, 0.97)
        scale = max(smp['e_im'].max(), 1e-12)
        v_dom = dm >= -1e-3 * scale
        agree = (v_dom == smp['passive'])
        results_97.append((smp['name'], smp['passive'], smp['min_eim'], dm, agree))

    n_agree = sum(r[4] for r in results_97)
    print(f"\n== Reproduction at frac=0.97 (independent gap implementation, no singular-cell fix) ==")
    print(f"agreement: {n_agree}/{len(samples)}")
    disagreers = [r for r in results_97 if not r[4]]
    print(f"disagreers ({len(disagreers)}):")
    for name, passive, min_eim, dm, agree in disagreers:
        print(f"  {name!r:40s} labelled_passive={passive!s:5s} DiHiTI_min_eim={min_eim:+.5f} dom_min@0.97={dm:+.5f}")

    # Now sweep frac for exactly the disagreeing profiles and see if the sign flips
    # (agreement recovered) or the margin trend, as frac -> 1.
    print(f"\n== Truncation sweep on the {len(disagreers)} disagreeing profiles ==")
    print(f"{'name':40s} " + " ".join(f"f={f:<9g}" for f in fracs))
    disagree_names = {r[0] for r in disagreers}
    sweep_data = {}
    for smp in samples:
        if smp['name'] not in disagree_names:
            continue
        xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
        row = []
        for f in fracs:
            dm = dom_min_for_frac(xs, gs, Ls, f)
            row.append(dm)
        sweep_data[smp['name']] = (smp['passive'], smp['min_eim'], row)
        row_str = " ".join(f"{v:+10.5f}" for v in row)
        print(f"{smp['name']!r:40s} {row_str}   [labelled_passive={smp['passive']}, DiHiTI_min={smp['min_eim']:+.5f}]")

    # Also sweep frac on a random subset of AGREEING profiles as a control -- to see
    # whether pushing frac->1 breaks anything that currently works (would indicate the
    # 0.97 cutoff was doing something protective, not just lossy).
    print(f"\n== Control: sweep on 10 currently-agreeing profiles ==")
    agree_samples = [r for r in results_97 if r[4]][:10]
    agree_names = {r[0] for r in agree_samples}
    for smp in samples:
        if smp['name'] not in agree_names:
            continue
        if smp['name'] not in {n for n, *_ in agree_samples}:
            continue
        xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
        row = []
        for f in fracs:
            dm = dom_min_for_frac(xs, gs, Ls, f)
            row.append(dm)
        row_str = " ".join(f"{v:+10.5f}" for v in row)
        print(f"{smp['name']!r:40s} {row_str}   [labelled_passive={smp['passive']}, DiHiTI_min={smp['min_eim']:+.5f}]")
        agree_names.discard(smp['name'])

    # Full-corpus agreement count at each frac
    print(f"\n== Full corpus agreement count vs frac ==")
    for f in fracs:
        n_ok = 0
        for smp in samples:
            xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
            dm = dom_min_for_frac(xs, gs, Ls, f)
            scale = max(smp['e_im'].max(), 1e-12)
            v_dom = dm >= -1e-3 * scale
            if v_dom == smp['passive']:
                n_ok += 1
        print(f"  frac={f:<10g} agreement {n_ok}/{len(samples)}")


if __name__ == '__main__':
    main()
