"""
Part 3 — proportionality (a)/(b): is the depression-per-unit-amplitude coefficient
kappa = -1/(pi * slope) UNIVERSAL across x0 (interior locations) and across base profiles
(headroom spanning decades), at FIXED delta?

Also: log-log slope check on the "centered" construction's delta-scaling (from part 2), to state
precisely what power law it follows (predicted ~delta^1 from the boundary-area argument).

Uses the "offset" (dilation-covariant, x0 = bump_center - delta/2) construction from part 2, since
that is the one that showed an approximate plateau. Exploits exact LINEARITY of ht_derivative in
eps' (finite-difference slope from two amplitudes a1,a2 gives the exact linear coefficient, no need
for a bisection search for a_min).
"""
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import json

def ht_derivative(xx, e_re):
    N = len(e_re)
    u = np.gradient(e_re, xx)
    v = np.imag(hilbert(u))
    e_im = cumulative_trapezoid(v, xx, initial=0)
    e_im -= np.linspace(e_im[0], e_im[-1], N)
    return e_im

def raised_cosine_bump(x, center, delta, a):
    s = (x - center) / (3 * delta)
    out = np.zeros_like(x)
    mask = np.abs(s) < 1
    out[mask] = a * 0.5 * (1 + np.cos(np.pi * s[mask]))
    return out

def base_profile(x, L, cs, phis, ks, scale=1.0):
    theta = np.pi * x / L
    out = np.zeros_like(x)
    for c, phi, k in zip(cs, phis, ks):
        out += c * np.cos(k * theta + phi)
    return scale * out

def slope_at(x, i0, ep0, delta, placement="offset", a_probe=1e-3):
    """Exact linear slope d(eps''(x0))/da via symmetric finite difference in a (linearity check
    included: verify two different a_probe values agree to confirm we're in the linear regime,
    which for this exactly-linear operator should hold to numerical precision)."""
    x0 = x[i0]
    if placement == "offset":
        center = x0 + delta / 2.0
    else:
        center = x0
    base_val = ht_derivative(x, ep0)[i0]
    vals = []
    for a in [a_probe, 2 * a_probe]:
        bump = raised_cosine_bump(x, center, delta, a)
        eps2 = ht_derivative(x, ep0 + bump)
        vals.append((eps2[i0] - base_val) / a)
    # vals[0] and vals[1] should agree (exact linearity of the operator); report their mean
    lin_check_rel_diff = abs(vals[0] - vals[1]) / (abs(vals[0]) + 1e-300)
    return np.mean(vals), lin_check_rel_diff

def run(N=8192, L=1.0):
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    delta = 0.02 * L  # fixed, well-resolved (delta/dx ~ 82 at N=8192), well inside part-2 plateau

    # Independent profiles (different seeds/K) AND, within each, one representative scale — this
    # spans headroom decades via GENUINELY DIFFERENT profile shapes/locations, not just a trivial
    # amplitude rescaling of one profile (which cannot test x0/profile universality, only linearity).
    scales = [1.0, 0.1, 0.01]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    x0_candidates = np.linspace(-0.75, 0.75, 21)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        K = rng.integers(3, 7)
        ks = np.arange(1, K + 1)
        cs = rng.uniform(0.3, 1.0, K) / ks
        phis = rng.uniform(0, 2 * np.pi, K)
        for scale in scales:
            ep0 = base_profile(x, L, cs, phis, ks, scale=scale)
            eps2_0 = ht_derivative(x, ep0)
            for x0 in x0_candidates:
                i0 = np.argmin(np.abs(x - x0))
                headroom = eps2_0[i0]
                if headroom <= 1e-6 * scale:
                    continue  # skip non-passive / near-zero-headroom points
                slope, lincheck = slope_at(x, i0, ep0, delta, placement="offset")
                C_phi = -1.0 / (np.pi * slope)
                a_min = -headroom / slope
                kappa = a_min / headroom
                rows.append(dict(seed=seed, scale=scale, x0=float(x[i0]), headroom=float(headroom),
                                  slope=float(slope), C_phi=float(C_phi), kappa=float(kappa),
                                  a_min=float(a_min), lin_check_rel_diff=float(lincheck)))
    return dict(N=N, L=L, delta=delta, dx=float(dx), rows=rows)

if __name__ == "__main__":
    out = run()
    print(f"N={out['N']}  delta={out['delta']:.4f}  dx={out['dx']:.3e}  delta/dx={out['delta']/out['dx']:.1f}")
    print(f"total (seed,scale,x0) rows: {len(out['rows'])}")

    # dedupe by (seed, x0) — C_phi/kappa are scale-independent by construction (linearity), so use
    # only scale=1.0 rows to test genuine x0/profile universality (avoid trivial repeats).
    uniq_rows = [r for r in out["rows"] if r["scale"] == 1.0]
    print(f"unique (seed,x0) rows used for universality stat: {len(uniq_rows)}")
    print(f"{'seed':>5} {'x0':>7} {'headroom':>12} {'slope':>12} {'C_phi':>10} {'kappa':>10} {'lin_chk':>10}")
    for r in uniq_rows:
        print(f"{r['seed']:>5} {r['x0']:>7.3f} {r['headroom']:>12.4e} {r['slope']:>12.4e} "
              f"{r['C_phi']:>10.4f} {r['kappa']:>10.4f} {r['lin_check_rel_diff']:>10.2e}")

    C_phis = np.array([r["C_phi"] for r in uniq_rows])
    kappas = np.array([r["kappa"] for r in uniq_rows])
    headrooms_all = np.array([r["headroom"] for r in out["rows"]])  # includes scale variation
    print(f"\nheadroom range (all rows, incl. scale variation): {headrooms_all.min():.3e} to "
          f"{headrooms_all.max():.3e} ({np.log10(headrooms_all.max()/headrooms_all.min()):.2f} decades)")
    print(f"C_phi across {len(uniq_rows)} independent (seed,x0) points: "
          f"mean={C_phis.mean():.4f}  std={C_phis.std():.4f}  "
          f"rel_std={C_phis.std()/C_phis.mean()*100:.2f}%  min={C_phis.min():.4f}  max={C_phis.max():.4f}")
    print(f"kappa: mean={kappas.mean():.4f}  std={kappas.std():.4f}  "
          f"rel_std={kappas.std()/kappas.mean()*100:.2f}%")
    print(f"kappa/pi: mean={kappas.mean()/np.pi:.4f}  std={kappas.std()/np.pi:.4f}  (claim: 0.418 +/-3%)")

    with open("research/results/cycle3_v2a_replication/part3_proportionality_data.json", "w") as f:
        json.dump(out, f, indent=2)
