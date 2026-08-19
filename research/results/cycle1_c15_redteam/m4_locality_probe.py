"""M4: locality / query-lower-bound probe.

Base profile: logistic w=0.05 (k=20), L=2.5, N=2**14 -- comfortably passive
(interior margin m = 0.00187 at |x|<=0.95L, from M1's own record for this
exact profile).

For bump widths sigma/L in {0.5,...,0.005}: bisect on amplitude a to find the
smallest |a| such that u + a*exp(-((x-x0)/sigma)^2) has interior min < 0,
scanning x0 over a grid and taking the EASIEST (smallest required |a|) x0.
"""
import numpy as np
from common import pg, tmm_h, grid, dihiti, OUTDIR
import os

L = 2.5
N = 2**14
NB = 1.7
k = 1.0 / 0.05

x = grid(L, N)
u0 = tmm_h.logistic(x, k, NB)
D = u0[0] - u0[-1]

e_im0 = dihiti(u0, x)
mask0 = np.abs(x) <= 0.95 * L
m0 = float(e_im0[mask0].min())
print(f"base profile: D = {D:.6f}   interior margin m0 = {m0:.6e}")


def interior_min_for(a, x0, sigma):
    pert = a * np.exp(-((x - x0) / sigma)**2)
    u = u0 + pert
    e_im = dihiti(u, x)
    return float(e_im[mask0].min())


def a_min_for_x0(x0, sigma, a_hi_start=None, tol_rel=1e-4):
    """Bisect on |a| (signed direction chosen to reduce eps'', i.e. push down)
    to find the smallest |a| that flips interior min negative. We must search
    BOTH signs of a since it's not obvious a priori which sign of a localized
    bump is more dangerous (a bump changes eps' -> changes the HT-transform
    partner eps'' nonlocally / non-monotonically), so try both and keep the
    cheaper one."""
    best = None
    for sign in (+1.0, -1.0):
        a_hi = sign * (a_hi_start or 0.05)
        # grow a_hi until it flips, cap search
        tries = 0
        while interior_min_for(a_hi, x0, sigma) >= 0 and tries < 40:
            a_hi *= 1.6
            tries += 1
        if interior_min_for(a_hi, x0, sigma) >= 0:
            continue  # didn't flip within search range for this sign
        a_lo = 0.0
        # bisection between a_lo (passive) and a_hi (flipped)
        for _ in range(60):
            a_mid = 0.5 * (a_lo + a_hi)
            if interior_min_for(a_mid, x0, sigma) >= 0:
                a_lo = a_mid
            else:
                a_hi = a_mid
            if abs(a_hi - a_lo) < tol_rel * abs(a_hi if a_hi != 0 else 1.0):
                break
        cand = abs(a_hi)
        if best is None or cand < best:
            best = cand
    return best


sigmas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
# The base profile's own interior-min sits near |x|/L = 0.95 (the interior
# cutoff itself, per M1: x_argmin_interior/L = -0.9499 for this exact
# profile) -- so a dense scan near the edges of the interior region is
# essential; a scan that stops at |x0|<2.2 (0.88L) undersamples exactly
# where a bump is cheapest to place. Combine a coarse global scan with a
# dense one covering |x0/L| in [0.85, 0.98].
x0_coarse = np.linspace(-2.2, 2.2, 23)
x0_edge = np.concatenate([np.linspace(-0.98 * L, -0.85 * L, 40),
                           np.linspace(0.85 * L, 0.98 * L, 40)])
x0_grid = np.concatenate([x0_coarse, x0_edge])

rows = []
for sigma_frac in sigmas:
    sigma = sigma_frac * L
    best_a = None
    best_x0 = None
    for x0 in x0_grid:
        a = a_min_for_x0(x0, sigma)
        if a is None:
            continue
        if best_a is None or a < best_a:
            best_a = a
            best_x0 = x0
    pert = best_a * np.exp(-((x - best_x0) / sigma)**2)
    linf = np.max(np.abs(pert))
    l1 = np.trapz(np.abs(pert), x)
    rows.append(dict(sigma_over_L=sigma_frac, a_min=best_a, x0_easiest=best_x0,
                      linf_over_D=linf / D, l1_over_DL=l1 / (D * L)))
    print(f"sigma/L={sigma_frac:<6}  a_min={best_a:.6e}  x0*={best_x0:+.3f}  "
          f"||pert||_inf/D={linf/D:.6e}  ||pert||_1/(D L)={l1/(D*L):.6e}")

with open(os.path.join(OUTDIR, 'm4_summary.txt'), 'w') as f:
    f.write(f"base: D={D:.6f}  interior margin m0={m0:.6e}  N={N}  L={L}\n")
    f.write(f"{'sigma/L':>10}{'a_min':>16}{'x0_easiest':>14}{'linf/D':>16}{'l1/(D L)':>16}\n")
    for r in rows:
        f.write(f"{r['sigma_over_L']:>10}{r['a_min']:>16.6e}{r['x0_easiest']:>14.3f}"
                f"{r['linf_over_D']:>16.6e}{r['l1_over_DL']:>16.6e}\n")

# trend check
a_vals = [r['a_min'] for r in rows]
print("\na_min sequence (sigma/L descending 0.5 -> 0.005):", a_vals)
if a_vals[-1] < a_vals[0]:
    print("TREND: a_min DECREASES as sigma shrinks -> narrow perturbations get CHEAPER")
    print("  -> localized label-flips are cheap; consistent with an Omega(N) query lower bound.")
else:
    print("TREND: a_min GROWS as sigma shrinks -> narrow perturbations are EXPENSIVE")
    print("  -> problem looks robust to narrow probes; sub-linear-query test not obviously blocked.")

with open(os.path.join(OUTDIR, 'm4_summary.txt'), 'a') as f:
    f.write(f"\na_min sequence: {a_vals}\n")

print("\nwrote m4_summary.txt")
