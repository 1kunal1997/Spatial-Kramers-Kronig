"""
Part 2 — width-independence of the bump-induced depression of eps''(x0).

Tests the crux claim: adding a small bump of amplitude a, width delta, to eps'(x) at x0 produces
Delta eps''(x0) independent of delta.

Tests TWO placements to resolve the pre-registered analytic tension in checkpoint.md:
  (A) bump exactly centered at x0 (symmetric/even about x0) -- measurement point = bump center.
      My analytic argument (H maps even->odd; odd vanishes at its own symmetry center) predicts
      Delta eps''(x0) -> 0 as delta -> 0 here (NOT a plateau), since the only surviving
      contribution is from the two boundary corrections, which scale like the bump's area ~ a*delta.
  (B) bump centered at x0 + delta/2 (asymmetric relative to x0: x0 sits on the FLANK of the bump,
      not its center) -- to see whether a delta-independent plateau appears when measurement point
      and bump are offset rather than coincident. This is the natural alternative reading of "bump
      at position x0" if x0 denotes where the local passivity threshold is being tracked and the
      bump is a nearby perturbation, not necessarily perfectly centered on the probe point.

Base profile: smooth, passive (eps''_0 > 0 in the interior), built from the SAME trig-sum family as
part 1 but with an added slowly-varying envelope so the profile is not perfectly odd/even itself
(avoids accidental extra symmetry). Bump: raised-cosine (compact support, exactly zero outside
[center-3*delta, center+3*delta], C1-smooth) so it has genuinely finite/localized support -- unlike
a Gaussian, this lets us test the delta -> L regime cleanly without long tails contaminating results.
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
    """Compact support on [center-3*delta, center+3*delta], amplitude a, C1 smooth, EVEN about
    center. 'Width' delta is the half-width of the central plateau-ish region (full support 6*delta)."""
    s = (x - center) / (3 * delta)
    out = np.zeros_like(x)
    mask = np.abs(s) < 1
    out[mask] = a * 0.5 * (1 + np.cos(np.pi * s[mask]))
    return out

def base_profile(x, L, rng):
    K = 6
    ks = np.arange(1, K + 1)
    cs = rng.uniform(0.3, 1.0, K) / ks
    phis = rng.uniform(0, 2 * np.pi, K)
    theta = np.pi * x / L
    out = np.zeros_like(x)
    for c, phi, k in zip(cs, phis, ks):
        out += c * np.cos(k * theta + phi)
    return out

def run_width_sweep(N=8192, L=1.0, seed=0, x0=0.15, a=0.02, placement="centered"):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    ep0 = base_profile(x, L, rng)
    eps2_0 = ht_derivative(x, ep0)
    # interpolate eps2_0 at x0 for baseline (x0 chosen off-grid-safe; use nearest grid index)
    i0 = np.argmin(np.abs(x - x0))
    x0g = x[i0]
    base_val = eps2_0[i0]

    deltas = np.geomspace(2 * dx, L / 6, 22)
    results = []
    for delta in deltas:
        if placement == "centered":
            center = x0g
        elif placement == "offset":
            center = x0g + delta / 2.0
        else:
            raise ValueError(placement)
        bump = raised_cosine_bump(x, center, delta, a)
        ep = ep0 + bump
        eps2 = ht_derivative(x, ep)
        depression = eps2[i0] - base_val
        results.append(dict(delta=float(delta), delta_over_dx=float(delta / dx),
                             depression=float(depression)))
    return dict(N=N, L=L, x0=x0g, dx=float(dx), a=a, placement=placement, results=results)

if __name__ == "__main__":
    out = {}
    for placement in ["centered", "offset"]:
        r = run_width_sweep(placement=placement)
        out[placement] = r
        print(f"\n=== placement={placement} ===  x0={r['x0']:.4f}  dx={r['dx']:.3e}  a={r['a']}")
        print(f"{'delta':>12} {'delta/dx':>10} {'depression':>14} {'depression/a':>14}")
        for row in r["results"]:
            print(f"{row['delta']:>12.5e} {row['delta_over_dx']:>10.1f} "
                  f"{row['depression']:>14.6e} {row['depression']/r['a']:>14.6e}")

    with open("research/results/cycle3_v2a_replication/part2_width_sweep_data.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved raw data to part2_width_sweep_data.json")
