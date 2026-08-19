"""
C28 audit, take 2: use the singular-cell-corrected kernel (as h9_fixes.py does) so the
diagonal ln(0) blow-up doesn't swamp the truncation-sweep signal. Independent reimplementation,
not an import of theorist code, but same numerical technique (cell-average regularization of
ln|eta-xi| on the diagonal) since that is not the object under test -- the 0.97L truncation is.
"""
import sys, os
import numpy as np

_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'theory'))
sys.path.insert(0, os.path.join(_ROOT, 'theory', 'passivity_2026Aug'))
import profile_generator as pg


def logln_reg(eta, xg, dxi):
    d = np.abs(eta - xg)
    out = np.empty_like(d)
    bad = d < 0.5 * dxi
    out[~bad] = np.log(d[~bad])
    out[bad] = np.log(np.maximum(dxi[bad], 1e-300) / 2.0) - 1.0
    return out


def gap_reg(g, t, eta, L):
    xg = np.tan(np.pi * t / (2 * L))
    dxi = np.gradient(xg)
    D = np.trapezoid(g, t)
    eta = np.atleast_1d(np.asarray(eta, float))
    out = np.array([np.trapezoid(logln_reg(e, xg, dxi) * g, t) for e in eta])
    return out - 0.5 * D * np.log(1 + eta ** 2)


def dom_min_for_frac(xs, gs, Ls, frac):
    mm = np.abs(xs) < frac * Ls
    if mm.sum() < 5:
        return np.nan
    eta_eval = np.tan(np.pi * xs[mm] / (2 * Ls))
    dl = gap_reg(gs, xs, eta_eval, Ls)
    return -dl.max() / np.pi


def main():
    rng = np.random.default_rng(7)
    samples = list(pg.sample(240, rng=rng))
    print(f"Loaded {len(samples)} profiles (same seed=7 as h9_fixes.py)")

    # Reproduce reported 233/240 at frac=0.97
    results_97 = []
    for smp in samples:
        xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
        dm = dom_min_for_frac(xs, gs, Ls, 0.97)
        scale = max(smp['e_im'].max(), 1e-12)
        v_dom = dm >= -1e-3 * scale
        agree = (v_dom == smp['passive'])
        results_97.append((smp['name'], smp['passive'], smp['min_eim'], dm, agree))
    n_agree = sum(r[4] for r in results_97)
    print(f"agreement @ frac=0.97 (regularized kernel): {n_agree}/{len(samples)}")
    disagreers = [r for r in results_97 if not r[4]]
    print(f"disagreers ({len(disagreers)}):")
    for name, passive, min_eim, dm, agree in disagreers:
        print(f"  {name!r:20s} labelled_passive={passive!s:5s} DiHiTI_min={min_eim:+.5f} dom_min@0.97={dm:+.5f}")

    fracs = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9999, 0.99999]
    print(f"\n== Truncation sweep on disagreeing profiles ==")
    disagree_names_idx = [i for i, r in enumerate(results_97) if not r[4]]
    for i in disagree_names_idx:
        smp = samples[i]
        xs = smp['x']; Ls = xs[-1]; gs = -np.gradient(smp['e_re'], xs)
        row = [dom_min_for_frac(xs, gs, Ls, f) for f in fracs]
        row_str = " ".join(f"{v:+10.5f}" for v in row)
        print(f"{smp['name']!r:20s} {row_str}  [passive={smp['passive']}, DiHiTI_min={smp['min_eim']:+.5f}]")

    print(f"\n== Full corpus agreement vs frac (regularized kernel) ==")
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
